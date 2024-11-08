import sys
sys.path.append("/root/projects/rl-enzyme-engineering/single_agent/env/models/BIND/")
sys.path.append("/root/projects/rl-enzyme-engineering/single_agent/env/models")
sys.path.append("/root/projects/rl-enzyme-engineering/single_agent/env")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")

import signal
import random
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import pandas as pd
import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Distribution
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils import TensorboardLogger
from agent.crossattention_graph_net import CustomGraphNet

import gymnasium
from agent.gumbel_softmax_distribution import GumbelSoftmaxDistribution
from agent.plm_ppo_policy import pLMPPOPolicy
from gymnasium.envs.registration import register
from env.protein_ligand_gym_env import ProteinLigandInteractionEnv, encode_aa_sequence

from top_sequences_tracker import TopSequencesTracker

logger = logging.getLogger(__name__)


def run(cfg: DictConfig):
    
    # Logger
    log_path = os.path.join(os.getcwd(), 'rl-loop')
    writer = SummaryWriter(log_path)
    tb_logger = TensorboardLogger(writer, train_interval=10,update_interval=1)
    
    device = cfg.experiment.device

    logger.debug("Loading PettingZoo environment...")

    # Signal Handler
    def signal_handler(signum, frame):
        if 'env' in locals():
            logger.info("Closing environment...")
            train_envs.close()
        if 'write' in locals():
            logger.info("Closing tensorboard writer...")
            writer.close() # Tensorboard writer
        if device == 'cuda':
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
        logger.info("Exiting...")
        sys.exit(0)
            
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set Seed
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Steps per episode: {cfg.on_policy_trainer.steps_per_epoch}")
    
    # Tracker for highest performing mutants
    top_sequences_tracker = TopSequencesTracker()

    # Setup Environment
    def init_env():
        return ProteinLigandInteractionEnv(
            seed=seed,
            top_sequences_tracker=top_sequences_tracker,
            render_mode="human",
            wildtype_aa_seq=cfg.experiment.wildtype_AA_seq,
            ligand_smiles=cfg.experiment.ligand_smiles,
            max_steps=cfg.on_policy_trainer.steps_per_epoch,
            device=device,
            config=cfg
        )
    
    train_envs = DummyVectorEnv([lambda: init_env() for _ in range(8)])

    action_space = train_envs.action_space[0]
    observation_space = train_envs.observation_space[0]
    seq_encoder = encode_aa_sequence
    
    # Setup PPO Agent
    net = CustomGraphNet(
        state_shape=observation_space,
        action_shape=action_space.shape,
        device=device
    )
    actor = Actor(preprocess_net=net, action_shape=action_space.shape, softmax_output=False, hidden_sizes=[256, 256, 256], device=device)
    critic = Critic(net, hidden_sizes=[256, 256, 256] ,device=device)
    
    def actor_init(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            torch.nn.init.constant_(layer.bias, 0.0)

    def critic_init(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1)
            torch.nn.init.constant_(layer.bias, 0.0)
            
    actor.last.apply(actor_init)
    critic.last.apply(critic_init)

    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=cfg.agents.picker_ppo.adam.learning_rate, eps=cfg.agents.picker_ppo.adam.epsilon
    )
    
    def gumbel_dist(logits: torch.Tensor) -> Distribution:
        return GumbelSoftmaxDistribution(logits)
    
    plm_ppo_policy: pLMPPOPolicy = pLMPPOPolicy(
        model_size_parameters = cfg.agents.filler_plm.evodiff_model_size_parameters,
        sequence_encoder=seq_encoder,
        device=device,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=gumbel_dist,
        action_space=action_space,
        eps_clip=cfg.agents.picker_ppo.policy.eps,
        dual_clip=None,
        value_clip=cfg.agents.picker_ppo.policy.value_clip,
        advantage_normalization=cfg.agents.picker_ppo.policy.advantage_normalization,
        recompute_advantage=cfg.agents.picker_ppo.policy.recompute_advantage,
        vf_coef=cfg.agents.picker_ppo.policy.vf_coef,
        ent_coef=cfg.agents.picker_ppo.policy.ent_coef,
        max_grad_norm= None, # Don't change!
        gae_lambda=cfg.agents.picker_ppo.policy.gae_lambda,
        discount_factor=cfg.agents.picker_ppo.policy.discount_factor,
        reward_normalization=cfg.agents.picker_ppo.policy.reward_normalization, # 5.1 Value Normalization
        deterministic_eval=False,
        action_scaling=False,
        lr_scheduler=None,
    ).to(device)
    
    # Setup Trainer
    buffer = VectorReplayBuffer(
        total_size=cfg.on_policy_trainer.replayBuffer.total_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=False,
        stack_num=1,
    )

    collector = Collector(
        policy=plm_ppo_policy,
        env=train_envs,
        buffer=buffer,
        exploration_noise=False,
    )

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # Saves after every epoch
        logger.info("Saving models and buffer.")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save(
            {
                "model": plm_ppo_policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        # TODO Renable
        #buffer_path = os.path.join(log_path, "train_buffer.pkl")
        #with open(buffer_path, "wb") as f:
        #    pickle.dump(collector.buffer, f)
        return ckpt_path

    def time_limit_reached(elapsed_time, limit_seconds):
        return elapsed_time >= limit_seconds

    start_time = time.time()
    time_limit = cfg.experiment.time_limit_h * 60 * 60

    def stop_fn(epoch, env_step):
        elapsed_time = time.time() - start_time
        return time_limit_reached(elapsed_time, time_limit)
    
    result = OnpolicyTrainer(
        policy=plm_ppo_policy,
        max_epoch=cfg.on_policy_trainer.epochs,
        batch_size=cfg.on_policy_trainer.batch_size,
        train_collector=collector,
        test_collector=None,
        buffer=None,
        step_per_epoch=cfg.on_policy_trainer.steps_per_epoch,
        repeat_per_collect=cfg.on_policy_trainer.repeat_per_collect,
        episode_per_test=0,
        update_per_step=1.0,
        step_per_collect=cfg.on_policy_trainer.steps_per_collect * 2, # TODO find out why this factor is necessary
        episode_per_collect=None,
        train_fn=None,
        test_fn=None,
        stop_fn=stop_fn,
        save_best_fn=None,
        save_checkpoint_fn=None, # TODO: Reeanble save_checkpoint_fn
        resume_from_log=False,
        reward_metric=None,
        logger=tb_logger,
        verbose=True,
        show_progress=True,
        test_in_train=True
    ).run()

@hydra.main(version_base=None, config_path="../", config_name='conf')
def main(cfg: DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
