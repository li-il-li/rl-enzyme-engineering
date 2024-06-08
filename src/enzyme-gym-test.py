# %%
import EnzymeGym
import gymnasium
import numpy as np
from pprint import pprint

# %%
env = gymnasium.make('EnzymeGym/ProteinLigandInteraction-v0',
                     wildtype="MAPLRKTYVLKLYVAGNTPNSVRALKTLNNILEKEFKGVYALKVIDVLKNPQLAEEDKILATPTLAKVLPPPVRRIIGDLSNREKVLIGLDLLYEEIGDQAEDDLGLE")

# %%
observation, info = env.reset()
pprint(observation)
pprint(info)
# %%
action = np.array([3, 5])
observation, reward, terminated, truncated , info = env.step(action)
pprint(observation)
pprint(info)