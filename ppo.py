import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

from rllib_wrapper.flatland_wrapper import FlatlandEnv
from rllib_wrapper.custom_preprocessor import TreeObsPreprocessor

ModelCatalog.register_custom_preprocessor("tree_obs_prep", TreeObsPreprocessor)
ray.init()

# Configurations https://ray.readthedocs.io/en/latest/rllib-training.html
trainer = PPOTrainer(env=FlatlandEnv, config={
    # "num_workers": 2,
    "train_batch_size": 4000,
    "model": {
        "custom_preprocessor": "tree_obs_prep"
    }
    # "callbacks": {
    #     "on_episode_start": None,     # arg: {"env": .., "episode": ...}
    #     "on_episode_end": None,       # arg: {"env": .., "episode": ...}
    #     "on_train_result": None,      # arg: {"trainer": ..., "result": ...}
    # }
})

for i in range(10000):
    print("=========== Iteration ", i, " ===========")
    print("ITE ", i, " --", trainer.train())

print("TRAINING COMPLETED SUCCESSFULLY")