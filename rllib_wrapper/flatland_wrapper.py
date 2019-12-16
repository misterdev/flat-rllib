import gym
import numpy as np
from ray import rllib

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.malfunction_generators import malfunction_from_params

from flatland.utils.rendertools import RenderTool

class FlatlandEnv(rllib.env.MultiAgentEnv):
    def __init__(self, env_config):
        width = 35
        height = 35
        self.n_agents = 2
        self.tree_depth = 3

        TreeObservation = TreeObsForRailEnv(max_depth=2)

        # Use a the malfunction generator to break agents from time to time
        stochastic_data = {
            'malfunction_rate': 8000,  # Rate of malfunction occurence of single agent
            'min_duration': 15,  # Minimal duration of malfunction
            'max_duration': 50  # Max duration of malfunction
        }

        # Different agent types (trains) with different speeds.
        speed_ration_map = {
            1.: 0.,  # Fast passenger train
            1. / 2.: 1.0,  # Fast freight train
            1. / 3.: 0.0,  # Slow commuter train
            1. / 4.: 0.0   # Slow freight train
        }

        self.env = RailEnv(
            width=width,
            height=height,
            rail_generator=sparse_rail_generator(
                max_num_cities=3,
                # Number of cities in map (where train stations are)
                seed=1,  # Random seed
                grid_mode=False,
                max_rails_between_cities=2,
                max_rails_in_city=3,
            ),
            schedule_generator=sparse_schedule_generator(speed_ration_map),
            number_of_agents=self.n_agents,
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            obs_builder_object=TreeObservation
        )

        # self.env_renderer = RenderTool(self.env, gl="PILSVG", )

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = np.zeros((1, 231))

    def reset(self):
        self.agents_done = []
        obs = self.env.reset()
        # self.env_renderer.reset()
        # print('================= RESET', obs)
        return obs[0]

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        # self.env_renderer.render_env(show=True, show_predictions=False, show_observations=True)
        # return <obs>, <reward: float>, <done: bool>, <info: dict>
        d = dict()
        r = dict()
        o = dict()
        i = dict()
        for a in range(len(self.env.agents)):
            if a not in self.agents_done:            
                o[a] = obs[a]
                r[a] = rewards[a]
                d[a] = dones[a]
                i[a] = dict()
                for info in infos:
                    i[a][info] = infos[info][a]
                # if dones[a]:
                    # print('DONE', a, infos)
        d['__all__'] = dones['__all__']

        # print('================= STEP', action_dict, d)
        for agent, done in dones.items():
            if done and agent != '__all__':
                self.agents_done.append(agent)

        # print('STEP', action_dict, r, d)
        return  o, r, d, i
    
    def get_num_agents():
        return self.n_agents