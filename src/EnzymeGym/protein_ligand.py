import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra
# EvoDiff
from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.conditional_generation import inpaint_simple
# AlphaFlow
from alphaflow_inference import init_esmflow, generate_conformation_ensemble
# FABind+

# DSMBind

class ProteinLigandInteractionEnv(gym.Env):

    def __init__(self, render_mode=None,
                 wildtype_aa_seq: str = 'AA',
                 ligand_smile: str = 'SMILE',
                 device = 'cuda',
                 config={}):
        
        # Hydra Config
        self.config = config

        # Models
        self.device = device
        self.sequence_model, self.sequenze_tokenizer = self._init_evodiff()
        self.folding_model = init_esmflow(ckpt = config.alphaflow.ckpt, device=device)

        # Protein Ligand
        self.wildtype_aa_seq = wildtype_aa_seq
        self.aa_seq_len = len(wildtype_aa_seq)
        self.ligand_smile = ligand_smile
        self.mutant_aa_seq = wildtype_aa_seq
        self.conformation_structures = []

        # RL Environment
        # Observations: Dictionary with the fittest mutant and the proteinligand conformation
        # encoded in latent variable Z.
        self.observation_space = spaces.Dict(
            {
                "fittest_mutation_aa_seq": spaces.Text(min_length=self.aa_seq_len, max_length=self.aa_seq_len),
                "protein_ligand_conformation_Z": spaces.Box(low=-100.0, high=100.0, shape=(2,2), dtype=np.float32)
            }
        )

        # Action: Pick location [idx_start, idx_end] of hole in AA sequence
        self.action_space = spaces.Box(
            low = 0,
            high = self.aa_seq_len,
            shape = (2,),
            dtype=np.int32
        )

        # Render: Not used for now!
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        return {
            "fittest_mutation_aa_seq": self.mutant_aa_seq,
            "protein_ligand_conformation_Z": np.zeros((2,2), dtype=np.float32)
        }

    def _get_info(self):
        return { 
            "binding_affinity": 10,
            "protein_conformations": self.conformation_structures
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initial wildtype protein ligand conformation
        self._protein_ligand_conformation = np.zeros((2,2), dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        aa_seq_hole_start_idx, aa_seq_hole_end_idx = np.sort(action)
        
        # Mutate Sequenze
        sample, entire_sequence, generated_idr = inpaint_simple(
            self.sequence_model,
            self.mutant_aa_seq,
            aa_seq_hole_start_idx,
            aa_seq_hole_end_idx,
            tokenizer=self.sequenze_tokenizer,
            device=self.device
        )
        self.mutant_aa_seq = entire_sequence
        
        # Genereate Protein Conformation Ensemble (Folding + MD)
        conformation_structures = generate_conformation_ensemble(self.folding_model,
                                                                 self.config,
                                                                 self.mutant_aa_seq)
        self.conformation_structures = conformation_structures
        
        # Protein Ligand Docking

        terminated = False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def _init_evodiff(self):
        model, collater, tokenizer, scheme = OA_DM_640M()
        model.to(device=self.device)

        return model, tokenizer