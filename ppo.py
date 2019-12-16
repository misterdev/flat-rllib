import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

import rllib_wrapper.callbacks as cb
from rllib_wrapper.flatland_wrapper import FlatlandEnv
from rllib_wrapper.custom_preprocessor import TreeObsPreprocessor

ModelCatalog.register_custom_preprocessor("tree_obs_prep", TreeObsPreprocessor)
ray.init()

# Configurations https://ray.readthedocs.io/en/latest/rllib-training.html
trainer = PPOTrainer(env=FlatlandEnv, config={
    "num_workers": 1,
    "train_batch_size": 4000,
    "model": {
        "custom_preprocessor": "tree_obs_prep"
    },
    "callbacks": {
        # "on_episode_start": cb.on_episode_start,     # arg: {"env": .., "episode": ...}
        "on_episode_end": cb.on_episode_end,       # arg: {"env": .., "episode": ...}
        "on_train_result": cb.on_train_result,      # arg: {"trainer": ..., "result": ...}
    },
    "log_level": "ERROR"
})

for i in range(100000 + 2):
    # print("=========== Iteration ", i, " ===========")
    trainer.train()
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

print("TRAINING COMPLETED SUCCESSFULLY")