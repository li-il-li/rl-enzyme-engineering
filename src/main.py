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

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf/", config_name='conf_dev')
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    logger.debug("Loading PettingZoo environment...")
    env = protein_ligand_gym_v0.env(render_mode=None,
                                    wildtype_aa_seq=cfg.experiment.wildtype_AA_seq,
                                    ligand_smile=cfg.experiment.ligand_smile,
                                    device=cfg.experiment.device,
                                    config=cfg)

    ## Step 2: Wrap the environment for Tianshou interfacing
    #env = PettingZooEnv(env)

    ## Step 3: Define policies for each agent
    #policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    ## Step 4: Convert the env to vector format
    #env = DummyVectorEnv([lambda: env])

    ## Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    #collector = Collector(policies, env)

    ## Step 6: Execute the environment with the agents playing for 1 epis


if __name__ == "__main__":
    main()