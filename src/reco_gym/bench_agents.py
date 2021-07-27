import multiprocessing
import time
from multiprocessing import Pool
from copy import deepcopy

from scipy.stats.distributions import beta

from reco_gym import AgentStats

from tqdm import tqdm

import numpy as np

def _collect_stats(args):
    env = args['env']
    agent = args['agent']
    num_offline_users = args['num_offline_users']
    num_online_users = args['num_online_users']
    num_organic_offline_users = args['num_organic_offline_users']
    epoch_with_random_reset = args['epoch_with_random_reset']
    epoch = args['epoch']
    num_epochs = args['num_epochs']

    start = time.time()
    #print(f"Start: Agent Training #{epoch}")

    successes = 0
    failures = 0

    unique_user_id = 0
    new_agent = deepcopy(agent)

    if epoch_with_random_reset:
        env = deepcopy(env)
        env.reset_random_seed(epoch)

    # Offline organic Training.
    for u in range(num_organic_offline_users):
        env.reset(unique_user_id + u)
        unique_user_id += 1
        observation, _, _, _ = env.step(None)
        new_agent.train(observation, None, None, True)
    unique_user_id += num_organic_offline_users

    # Offline Training.
    for u in range(num_offline_users):
        env.reset(unique_user_id + u)
        new_observation, _, done, _ = env.step(None)
        while not done:
            old_observation = new_observation
            action, new_observation, reward, done, info = env.step_offline(old_observation, 0, False)
            new_agent.train(old_observation, action, reward, done)
    unique_user_id += num_offline_users
    
    # Set process-id for agent
    if hasattr(new_agent, 'pid'):
        new_agent.pid = 1 + epoch

    # Online Testing.
    #print(f"Start: Agent Testing #{epoch}")
    disappointment = 0
    with tqdm(total=num_online_users, desc='Evaluate {0}'.format(epoch)) as pbar:
        for u in range(num_online_users):
            env.reset(unique_user_id + u)
            new_agent.reset()
            new_observation, _, done, _ = env.step(None)
            reward = None
            done = None
            while not done:
                if hasattr(new_agent, 'skyline'):
                    pclick = env._get_true_pclick()
                    a = np.argmax(pclick)
                    ps_all = np.zeros(pclick.shape)
                    ps_all[a] = 1.0
                    action = {
                        'a': a,
                        'ps': 1.0,
                        'ps-a': ps_all
                        }
                else:
                    action = new_agent.act(new_observation, reward, done)

                # If we have a reward estimate, we want to measure post-decision disappointment
                if 'r-est' in action:
                    disappointment += action['r-est'] - env._get_true_pclick()[action['a']]

                new_observation, reward, done, info = env.step(action['a'])

                if reward:
                    successes += 1
                else:
                    failures += 1
            # Update progres-bar
            pbar.update(1)
    # Update counters for unique simulation users
    unique_user_id += num_online_users

    # Normalise disappointment to get the mean
    disappointment /= (successes + failures)
    extra_info = f" (Mean disappointment: {disappointment}, N = {successes + failures})"
    print(f"End: Agent Testing #{epoch} ({time.time() - start}s)" + extra_info)

    return {
        AgentStats.SUCCESSES: successes,
        AgentStats.FAILURES: failures,
        AgentStats.MEAN_DISAPPOINTMENT: disappointment,
    }


def test_agent(
        env,
        agent,
        num_offline_users = 1000,
        num_online_users = 100,
        num_organic_offline_users = 100,
        num_epochs = 1,
        epoch_with_random_reset = False
):
    successes = 0
    failures = 0
    disappointment = 0

    # Don't spawn too many processes when P is large
    # Acting needs all the parallel CPU power it can get in this case
    n_processes = 2 if (env.config.num_products >= 150) else multiprocessing.cpu_count()
    with Pool(processes = n_processes) as pool:
        argss = [
            {
                'env': env,
                'agent': agent,
                'num_offline_users': num_offline_users,
                'num_online_users': num_online_users,
                'num_organic_offline_users': num_organic_offline_users,
                'epoch_with_random_reset': epoch_with_random_reset,
                'epoch': epoch,
                'num_epochs': num_epochs
            }
            for epoch in range(num_epochs)
        ]

        for result in [_collect_stats(args) for args in argss] if num_epochs == 1 else pool.map(_collect_stats, argss):
            successes += result[AgentStats.SUCCESSES]
            failures += result[AgentStats.FAILURES]
            disappointment += result[AgentStats.MEAN_DISAPPOINTMENT]

    disappointment /= num_epochs
    print(f'\tMedian CTR Estimate - {beta.ppf(0.500, successes + 1, failures + 1)}\tMean Disappointment - {disappointment}\t(N = {successes + failures})')
    return (
        beta.ppf(0.500, successes + 1, failures + 1),
        beta.ppf(0.025, successes + 1, failures + 1),
        beta.ppf(0.975, successes + 1, failures + 1),
        disappointment,
    )
