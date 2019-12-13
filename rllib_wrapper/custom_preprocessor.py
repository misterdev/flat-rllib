import numpy as np
from ray.rllib.models.preprocessors import Preprocessor

from rllib_wrapper.observation_utils import normalize_observation
class TreeObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.step_memory = 2 # TODO options["custom_options"]["step_memory"]
        self.tree_depth = 2
        return sum([space.shape[0] for space in obs_space]),

    def transform(self, obs):
        if obs:
            ret = normalize_observation(obs, self.tree_depth, observation_radius=10)
        else:
            ret = np.zeros(231)
        
        return ret
