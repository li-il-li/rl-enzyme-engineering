import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ProteinLigandInteractionEnv(gym.Env):

    def __init__(self, render_mode=None, wildtype: str = 'AA'):
        self.aa_seq_len = len(wildtype) # The length of the wildtype AA sequnce

        # Observations are dictionaries with the fittest location and the proteinligand conformation
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
            "fittest_mutation_aa_seq": "ABSCD",
            "protein_ligand_conformation_Z": np.zeros((2,2), dtype=np.float32)
        }

    def _get_info(self):
        return { "binding_affinity": 10 }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initial wildtype protein ligand conformation
        self._protein_ligand_conformation = np.zeros((2,2), dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        terminated = False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info