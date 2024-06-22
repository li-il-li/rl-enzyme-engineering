import sys
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/AlphaFlow")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/FABind/FABind_plus/fabind")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models/DSMBind")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env/models")
sys.path.append("/root/projects/rl-enzyme-engineering/src/ProteinLigandGym/env")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
from ProteinLigandGym import protein_ligand_gym_v0
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from ProteinSequencePolicy.policy import ProteinSequencePolicy
from supersuit import pad_action_space_v0

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf/", config_name='conf_dev')
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    logger.debug("Loading PettingZoo environment...")
    env = protein_ligand_gym_v0.env(render_mode="human",
                                    wildtype_aa_seq=cfg.experiment.wildtype_AA_seq,
                                    ligand_smile=cfg.experiment.ligand_smile,
                                    device=cfg.experiment.device,
                                    config=cfg)
    env = PettingZooEnv(env)

    policies = MultiAgentPolicyManager(
        [
            RandomPolicy(),
            ProteinSequencePolicy(
                action_space=env.action_space,
                device=cfg.experiment.device
            )
        ],
        env
    )

    env = DummyVectorEnv([lambda: env])

    collector = Collector(policies, env)

    result = collector.collect(n_episode=1, render=0.1)


if __name__ == "__main__":
    main()