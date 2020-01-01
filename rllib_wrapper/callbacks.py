import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
# from ray.tune.logger import pretty_print

def on_episode_end(info):
    episode = info['episode']
    env = info['env'].envs[0]
    agents_done = len(env.agents_done)
    episode_done = agents_done == env.n_agents
    
    episode.custom_metrics['score'] = episode.total_reward
    episode.custom_metrics['done'] = episode_done
    episode.custom_metrics['agents_done'] = agents_done

    completed = 'done {:>3.0f}% ({}/{})'.format(
        agents_done / env.n_agents * 100,
        agents_done,
        env.n_agents
    )
    print('\rEPISODE {:>7} -- {:^19} -- score {:>7.1f} -- length {:>3}'.format(
        env.n_episode,
        completed,
        episode.total_reward,
        episode.length
    ))

def on_train_result(info):
    result = info['result']
    custom_metrics = result['custom_metrics']
    print(
    '\r\n==> Iteration {:,} - {:,} episodes -- avg score: {:.0f} -- avg length: {:.0f} \n'.format(
        result['training_iteration'],
        result['episodes_total'],
        custom_metrics['score_mean'],
        result['episode_len_mean']
    ))