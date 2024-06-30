import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/AlphaFlow")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/FABind/FABind_plus/fabind")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/DSMBind")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/BIND/")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env")

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
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Distribution, Independent, Bernoulli
from torch.optim.lr_scheduler import LambdaLR
from ProteinSequencePolicy.policy import ProteinSequencePolicy
from ProteinLigandGym import protein_ligand_gym_v0
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net, MLP
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils import TensorboardLogger

logger = logging.getLogger(__name__)

class CustomNet(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_sizes, device):
        super().__init__()
        self._device = device
        self.model = MLP(
            int(np.prod(state_shape)),
            int(np.prod(action_shape)),
            hidden_sizes,
            device=device
        )
        self.action_shape = action_shape
        self.output_dim = self.model.output_dim
        self.input_dim = int(np.prod(state_shape))

    def forward(self, obs, state=None, info=None):
        obs = obs.protein_ligand_conformation_latent
        #logger.info(f"Obs Preprocess Network: {obs.shape}")
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        logits = self.model(obs)
        return logits, state
    
    def get_output_dim(self):
        return self.output_dim

    def get_input_dim(self):
        return self.input_dim


@hydra.main(version_base=None, config_path="../conf/", config_name='conf_dev')
def main(cfg: DictConfig):
    
    # Logger
    log_path = os.path.join(os.getcwd(), 'rl-loop')
    writer = SummaryWriter(log_path)
    tb_logger = TensorboardLogger(writer, train_interval=10)
    
    device = cfg.experiment.device

    print(OmegaConf.to_yaml(cfg))

    logger.debug("Loading PettingZoo environment...")

    # Set Seed
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    tb_logger.write("config", 0, {"seed": seed})

    env = protein_ligand_gym_v0.env(
        render_mode="human",
        wildtype_aa_seq=cfg.experiment.wildtype_AA_seq,
        ligand_smile=cfg.experiment.ligand_smile,
        max_steps=cfg.agents.steps_per_epoch,
        device=device,
        config=cfg
    )
    seq_encoder = env.encode_aa_sequence
    env = PettingZooEnv(env)
    
    # Model PPO
    state_shape = env.observation_space['protein_ligand_conformation_latent'].shape
    action_shape = env.action_space.shape

    #logger.info(f"Action Space shape: {action_shape}")
    net = CustomNet(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128]*4, device=device)
    #logger.info(f"Net: {net}")
    actor = Actor(preprocess_net=net, action_shape=action_shape, softmax_output=False, device=device)
    critic = Critic(net, device=device)
    
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
        ActorCritic(actor, critic).parameters(), lr=cfg.agents.adam.learning_rate, eps=cfg.agents.adam.epsilon
    )

    def dist(logits: torch.Tensor) -> Distribution:
        target_ratio = cfg.agents.sequence_edit_target_ratio
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Calculate current mean probability
        current_ratio = probs.mean()
        # Adjust probabilities to meet target ratio
        adjusted_probs = probs * (target_ratio / current_ratio)
        # Clip probabilities to valid range [0, 1]
        adjusted_probs = torch.clamp(adjusted_probs, 0, 1)

        return Bernoulli(probs=adjusted_probs)
    
    # decay learning rate to 0 linearly
    #lr_scheduler = LambdaLR(optim, lr_lambda=lambda e: 1 - e / epoch)
    
    # PPO policy
    #logger.info(f"Observation Space: {env.observation_space['protein_ligand_conformation_latent']}")
    #logger.info(f"Action Space Picker: {env.action_space}")
    ppo_policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_space=env.action_space,
        eps_clip=cfg.agents.ppo.eps,
        dual_clip=None,
        value_clip=cfg.agents.ppo.value_clip,
        advantage_normalization=cfg.agents.ppo.advantage_normalization,
        recompute_advantage=cfg.agents.ppo.recompute_advantage,
        vf_coef=cfg.agents.ppo.vf_coef,
        ent_coef=cfg.agents.ppo.ent_coef,
        max_grad_norm=cfg.agents.ppo.max_grad_norm,
        gae_lambda=cfg.agents.ppo.gae_lambda,
        discount_factor=cfg.agents.ppo.discount_factor,
        reward_normalization=cfg.agents.ppo.reward_normalization, # 5.1 Value Normalization
        deterministic_eval=False,
        observation_space=env.observation_space['protein_ligand_conformation_latent'],
        action_scaling=False,
        lr_scheduler=None,
        #lr_scheduler=lr_scheduler,
    ).to(device)
    
    buffer = VectorReplayBuffer(
        total_size=cfg.agents.replayBuffer.total_size,
        buffer_num=1,
        ignore_obs_next=True,
        save_only_last_obs=False,
        stack_num=1,
    )

    policy = MultiAgentPolicyManager(
        [
            ppo_policy,
            ProteinSequencePolicy(
                model_size_parameters = cfg.evodiff.model_size_parameters,
                sequence_encoder=seq_encoder,
                action_space=env.action_space,
                device=device
            )
        ],
        env
    )

    env = DummyVectorEnv([lambda: env])

    collector = Collector(
        policy=policy,
        env=env,
        buffer=buffer,
        exploration_noise=False,
    )

    #def train_fn(epoch, env_step):
    #    policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # Saves after every epoch
        logger.info("Saving models and buffer.")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        # TODO Renable
        #buffer_path = os.path.join(log_path, "train_buffer.pkl")
        #with open(buffer_path, "wb") as f:
        #    pickle.dump(collector.buffer, f)
        return ckpt_path

    result = OnpolicyTrainer(
        policy=policy,
        max_epoch=cfg.agents.epochs,
        batch_size=cfg.agents.batch_size,
        train_collector=collector,
        test_collector=None,
        buffer=None,
        step_per_epoch=cfg.agents.steps_per_epoch,
        repeat_per_collect=cfg.agents.repeat_per_collect,
        episode_per_test=0,
        update_per_step=1.0,
        step_per_collect=None,
        episode_per_collect=cfg.agents.episode_per_collect,
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_best_fn=None,
        save_checkpoint_fn=save_checkpoint_fn,
        resume_from_log=False,
        reward_metric=None,
        logger=tb_logger,
        verbose=True,
        show_progress=True,
        test_in_train=False,
        save_fn=None,
    ).run()
    
    # This does not get executed afaik
    #collector = collector.collect(n_episode=1, render=0.1)

    #args.eval_mean_reward = result.returns_stat.mean
    #args.training_time_h = ((train_end_time - start_time) / 60) / 60
    #args.total_time_h = ((eval_end_time - start_time) / 60) / 60


if __name__ == "__main__":
    main()