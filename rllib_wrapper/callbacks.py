import numpy as np
from flatland.envs.agent_utils import RailAgentStatus

# scores_window = deque(maxlen=100)
# done_window = deque(maxlen=100)
# scores = []
# dones_list = []

def on_episode_start(info):
    env = info['env']
    # Collection information about training
    tasks_finished = 0
    for current_agent in env.agents:
        if current_agent.status == RailAgentStatus.DONE_REMOVED:
            tasks_finished += 1
    # done_window.append(tasks_finished / max(1, env.get_num_agents()))
    # # scores_window.append(score)  # TODO / max_steps save most recent score
    # scores.append(np.mean(scores_window))
    # dones_list.append((np.mean(done_window)))
    print('HERMANO', env)
    # print(
    #     '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
    #         env.get_num_agents(), x_dim, y_dim,
    #         trials,
    #         np.mean(scores_window),
    #         100 * np.mean(done_window),
    #         eps, action_prob / np.sum(action_prob)), end=" ")

def on_episode_end(info):
    print("ciao")

def on_train_result(info):
    print("on_train_result")