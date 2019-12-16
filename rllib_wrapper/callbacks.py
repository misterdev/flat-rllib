import numpy as np
from flatland.envs.agent_utils import RailAgentStatus

# scores_window = deque(maxlen=100)
# done_window = deque(maxlen=100)
# scores = []
# dones_list = []

# def on_episode_start(info):
#     print('')

def on_episode_end(info):
    episode = info['episode']

    # Calculation of a custom score metric: cum of all accumulated rewards, divided by the number of agents
    # and the number of the maximum time steps of the episode.
    score = 0
    for k, v in episode._agent_reward_history.items():
        score += np.sum(v)
    score /= len(episode._agent_reward_history) # TODO * episode.horizon) AND WTF

    # Calculation of the proportion of solved episodes before the maximum time step
    done = 1
    # if len(episode._agent_reward_history[0]) == episode.horizon:
    #     done = 0
    episode.custom_metrics["score"] = score
    episode.custom_metrics["proportion_episode_solved"] = done

def on_train_result(info):
    n_agents = 10
    x_dim = y_dim = 35
    print(
    '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}'.format(
        n_agents, x_dim, y_dim,
        info['result']['episodes_total'],
        info['result']['custom_metrics']['score_mean']
    ))