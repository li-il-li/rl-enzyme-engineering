from gymnasium.envs.registration import register

register(
     id="EnzymeGym/ProteinLigandInteraction-v0",
     entry_point="protein_ligand:ProteinLigandInteractionEnv",
     max_episode_steps=20,
)