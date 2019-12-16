import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
# from ray.tune.logger import pretty_print

# def on_episode_start(info):
#     print('')

def on_episode_end(info):
    episode = info['episode']
    episode.custom_metrics["score"] = episode.total_reward

def on_train_result(info):
    # print('on_train_result', pretty_print(info))
    n_agents = 10
    x_dim = y_dim = 35
    print(
    '\rIteration {} - {} Episodes\tTraining {} Agents on ({},{}).\t Average Score: {:.3f}'.format(
        info['result']['training_iteration'],
        info['result']['episodes_total'],
        n_agents, x_dim, y_dim,
        info['result']['custom_metrics']['score_mean']
    ))