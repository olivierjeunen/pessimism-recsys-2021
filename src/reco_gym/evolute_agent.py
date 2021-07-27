import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import time
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.stats import beta

import reco_gym

from reco_gym import Configuration, TrainingApproach, EvolutionCase, AgentInit, AgentStats, RoiMetrics

from tqdm import tqdm


EpsilonDelta = .02
EpsilonSteps = 6  # Including epsilon = 0.0.
EpsilonPrecision = 2
EvolutionEpsilons = (0.00, 0.01, 0.02, 0.03, 0.05, 0.08)

GraphCTRMin = 0.009
GraphCTRMax = 0.021


def evolute_agent(
        env,
        agent,
        num_initial_train_users = 100,
        num_step_users = 1000,
        num_steps = 10,
        training_approach = TrainingApproach.ALL_DATA,
        sliding_window_samples = 10000):
    initial_agent = deepcopy(agent)

    unique_user_id = 0
    for u in range(num_initial_train_users):
        env.reset(unique_user_id + u)
        agent.reset()
        new_observation, reward, done, _ = env.step(None)
        while not done:
            old_observation = new_observation
            action, new_observation, reward, done, _ = env.step_offline(new_observation, reward, False)
            agent.train(old_observation, action, reward, done)
    unique_user_id += num_initial_train_users

    rewards = {
        EvolutionCase.SUCCESS: [],
        EvolutionCase.SUCCESS_GREEDY: [],
        EvolutionCase.FAILURE: [],
        EvolutionCase.FAILURE_GREEDY: [],
        EvolutionCase.ACTIONS: dict()
    }
    training_agent = deepcopy(agent)
    samples = 0

    for action_id in range(env.config.num_products):
        rewards[EvolutionCase.ACTIONS][action_id] = [0]

    for step in range(num_steps):
        successes = 0
        successes_greedy = 0
        failures = 0
        failures_greedy = 0

        for u in range(num_step_users):
            env.reset(unique_user_id + u)
            agent.reset()
            new_observation, reward, done, _ = env.step(None)
            while not done:
                old_observation = new_observation
                action = agent.act(old_observation, reward, done)
                new_observation, reward, done, info = env.step(action['a'])
                samples += 1

                should_update_training_data = False
                if training_approach == TrainingApproach.ALL_DATA or training_approach == TrainingApproach.LAST_STEP:
                    should_update_training_data = True
                elif training_approach == TrainingApproach.SLIDING_WINDOW_ALL_DATA:
                    should_update_training_data = samples % sliding_window_samples == 0
                elif training_approach == TrainingApproach.ALL_EXPLORATION_DATA:
                    should_update_training_data = not action['greedy']
                elif training_approach == TrainingApproach.SLIDING_WINDOW_EXPLORATION_DATA:
                    should_update_training_data = (not action['greedy']) and samples % sliding_window_samples == 0
                else:
                    assert False, f"Unknown Training Approach: {training_approach}"

                if should_update_training_data:
                    training_agent.train(old_observation, action, reward, done)

                if reward:
                    successes += 1
                    if 'greedy' in action and action['greedy']:
                        successes_greedy += 1
                    rewards[EvolutionCase.ACTIONS][action['a']][-1] += 1
                else:
                    if 'greedy' in action and action['greedy']:
                        failures_greedy += 1
                    failures += 1
        unique_user_id += num_step_users

        agent = training_agent
        for action_id in range(env.config.num_products):
            rewards[EvolutionCase.ACTIONS][action_id].append(0)

        if training_approach == TrainingApproach.LAST_STEP:
            training_agent = deepcopy(initial_agent)
        else:
            training_agent = deepcopy(agent)

        rewards[EvolutionCase.SUCCESS].append(successes)
        rewards[EvolutionCase.SUCCESS_GREEDY].append(successes_greedy)
        rewards[EvolutionCase.FAILURE].append(failures)
        rewards[EvolutionCase.FAILURE_GREEDY].append(failures_greedy)

    return rewards


def build_agent_init(agent_key, ctor, def_args):
    return {
        agent_key: {
            AgentInit.CTOR: ctor,
            AgentInit.DEF_ARGS: def_args,
        }
    }


def _collect_stats(args):
    """
    Function that is executed in a separate process.

    :param args: arguments of the process to be executed.

    :return: a vector of CTR for these confidence values:
        0th: Q0.500
        1st: Q0.025
        snd: Q0.975
    """
    start = time.time()
    print(f"Start: Num of Offline Users: {args['num_offline_users']}")
    stats = reco_gym.test_agent(
        deepcopy(args['env']),
        deepcopy(args['agent']),
        args['num_offline_users'],
        args['num_online_users'],
        args['num_organic_offline_users'],
        args['num_epochs'],
        args['epoch_with_random_reset']
    )
    print(f"End: Num of Offline Users: {args['num_offline_users']} ({time.time() - start}s)")
    return stats


def gather_agent_stats(
        env,
        env_args,
        extra_env_args,
        agents_init_data,
        user_samples = (100, 1000, 2000, 3000, 5000, 8000, 10000, 13000, 14000, 15000),
        num_online_users = 15000,
        num_epochs = 1,
        epoch_with_random_reset = False,
        num_organic_offline_users = 100
):
    """
    The function that gathers Agents statistics via evaluating Agent performance
     under different Environment conditions.

    :param env: the Environment where some changes should be introduced and where Agent stats should
        be gathered.
    :param env_args: Environment arguments (default ones).
    :param extra_env_args: extra Environment conditions those alter default values.
    :param agents_init_data: Agent initialisation data.
        This is a dictionary that has the following structure:
        {
            '<Agent Name>': {
                AgentInit.CTOR: <Constructor>,
                AgentInit.DEF_ARG: <Default Arguments>,
            }
        }
    :param user_samples: Number of Offline Users i.e. Users used to train a Model.
    :param num_online_users: Number of Online Users i.e. Users used to validate a Model.
    :param num_epochs: how many different epochs should be tried to gather stats?
    :param epoch_with_random_reset: should be a Random Seed reset at each new epoch?

    :return: a dictionary with stats
        {
            AgentStats.SAMPLES: [<vector of training offline users used to train a model>]
            AgentStats.AGENTS: {
                '<Agent Name>': {
                    AgentStats.Q0_025: [],
                    AgentStats.Q0_500: [],
                    AgentStats.Q0_975: [],
                }
            }
        }
    """
    new_env_args = {
        **env_args,
        **extra_env_args,
    }

    new_env = deepcopy(env)
    new_env.init_gym(new_env_args)

    agents = build_agents(agents_init_data, new_env_args)

    agent_stats = {
        AgentStats.SAMPLES: user_samples,
        AgentStats.AGENTS: dict(),
    }

    for agent_key in agents:
        print(f"Agent: {agent_key}")
        stats = {
            AgentStats.Q0_025: [],
            AgentStats.Q0_500: [],
            AgentStats.Q0_975: [],
            AgentStats.MEAN_DISAPPOINTMENT: [],
        }

        #with Pool(processes = multiprocessing.cpu_count()) as pool:
        with ThreadPool(processes = multiprocessing.cpu_count()) as pool:
            #with NoDaemonProcessPool(processes = multiprocessing.cpu_count()) as pool:
            argss = [
                {
                    'env': new_env,
                    'agent': agents[agent_key],
                    'num_offline_users': num_offline_users,
                    'num_online_users': num_online_users,
                    'num_organic_offline_users': num_organic_offline_users,
                    'num_epochs': num_epochs,
                    'epoch_with_random_reset': epoch_with_random_reset,
                }
                for num_offline_users in user_samples
            ]

            for result in pool.map(_collect_stats, argss) if num_epochs == 1 else [
                _collect_stats(args) for args in argss
            ]:
                stats[AgentStats.Q0_025].append(result[1])
                stats[AgentStats.Q0_500].append(result[0])
                stats[AgentStats.Q0_975].append(result[2])

                # TODO - extract disappointment from result and add to stats
                stats[AgentStats.MEAN_DISAPPOINTMENT].append(result[3])

        agent_stats[AgentStats.AGENTS][agent_key] = stats

    return agent_stats


def build_agents(agents_init_data, new_env_args):
    agents = dict()
    for agent_key in agents_init_data:
        agent_init_data = agents_init_data[agent_key]
        ctor = agent_init_data[AgentInit.CTOR]
        def_args = agent_init_data[AgentInit.DEF_ARGS]
        agents[agent_key] = ctor(
            Configuration({
                **def_args,
                **new_env_args,
            })
        )
    return agents

def plot_agent_stats(agent_stats, figname = 'results.png'):
    _, ax = plt.subplots(
        1,
        1,
        figsize = (16, 8)
    )

    user_samples = agent_stats[AgentStats.SAMPLES]
    for agent_key in agent_stats[AgentStats.AGENTS]:
        stats = agent_stats[AgentStats.AGENTS][agent_key]

        ax.fill_between(
            user_samples,
            stats[AgentStats.Q0_975],
            stats[AgentStats.Q0_025],
            alpha = .05
        )

        ax.plot(user_samples, stats[AgentStats.Q0_500])

        ax.set_xlabel('Samples #')
        ax.set_ylabel('CTR')
        ax.legend([
            "$C^{CTR}_{0.5}$: " + f"{agent_key}" for agent_key in agent_stats[AgentStats.AGENTS]
        ])
    plt.savefig(figname, dpi = 400, bbox_inches = 'tight')
    plt.show()


def plot_evolution_stats(
        agent_evolution_stats,
        max_agents_per_row = 2,
        epsilons = EvolutionEpsilons,
        plot_min = GraphCTRMin,
        plot_max = GraphCTRMax
):
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) / max_agents_per_row),
        max_agents_per_row,
        figsize = (16, 10),
        squeeze = False
    )
    labels = [("$\epsilon=$" + format_epsilon(epsilon)) for epsilon in epsilons]

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        ax = axs[int(ix / max_agents_per_row), int(ix % max_agents_per_row)]
        agent_evolution_stat = agent_evolution_stats[agent_key]

        ctr_means = []
        for epsilon in epsilons:
            epsilon_key = format_epsilon(epsilon)
            evolution_stat = agent_evolution_stat[epsilon_key]

            steps = []
            ms = []
            q0_025 = []
            q0_975 = []

            assert (len(evolution_stat[EvolutionCase.SUCCESS]) == len(evolution_stat[EvolutionCase.FAILURE]))
            for step in range(len(evolution_stat[EvolutionCase.SUCCESS])):
                steps.append(step)
                successes = evolution_stat[EvolutionCase.SUCCESS][step]
                failures = evolution_stat[EvolutionCase.FAILURE][step]

                ms.append(beta.ppf(0.5, successes + 1, failures + 1))
                q0_025.append(beta.ppf(0.025, successes + 1, failures + 1))
                q0_975.append(beta.ppf(0.975, successes + 1, failures + 1))

            ctr_means.append(np.mean(ms))

            ax.fill_between(
                range(len(steps)),
                q0_975,
                q0_025,
                alpha = .05
            )
            ax.plot(steps, ms)

        ctr_means_mean = np.mean(ctr_means)
        ctr_means_div = np.sqrt(np.var(ctr_means))
        ax.set_title(
            f"Agent: {agent_key}\n"
            + "$\hat{Q}^{CTR}_{0.5}="
            + "{0:.5f}".format(round(ctr_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}^{CTR}_{0.5}="
            + "{0:.5f}".format(round(ctr_means_div, 5))
            + "$"
        )
        ax.legend(labels)
        ax.set_ylabel('CTR')
        ax.set_ylim([plot_min, plot_max])

    plt.subplots_adjust(hspace = .5)
    plt.show()


def plot_heat_actions(
        agent_evolution_stats,
        epsilons = EvolutionEpsilons
):
    max_epsilons_per_row = len(epsilons)
    the_first_agent = next(iter(agent_evolution_stats.values()))
    epsilon_steps = len(the_first_agent)
    rows = int(len(agent_evolution_stats) * epsilon_steps / max_epsilons_per_row)
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) * epsilon_steps / max_epsilons_per_row),
        max_epsilons_per_row,
        figsize = (16, 4 * rows),
        squeeze = False
    )

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        agent_evolution_stat = agent_evolution_stats[agent_key]
        for (jx, epsilon_key) in enumerate(agent_evolution_stat):
            flat_index = ix * epsilon_steps + jx
            ax = axs[int(flat_index / max_epsilons_per_row), int(flat_index % max_epsilons_per_row)]

            evolution_stat = agent_evolution_stat[epsilon_key]

            action_stats = evolution_stat[EvolutionCase.ACTIONS]
            total_actions = len(action_stats)
            heat_data = []
            for kx in range(total_actions):
                heat_data.append(action_stats[kx])

            heat_data = np.array(heat_data)
            im = ax.imshow(heat_data)

            ax.set_yticks(np.arange(total_actions))
            ax.set_yticklabels([f"{action_id}" for action_id in range(total_actions)])

            ax.set_title(f"Agent: {agent_key}\n$\epsilon=${epsilon_key}")

            _ = ax.figure.colorbar(im, ax = ax)

    plt.show()


def plot_roi(
        agent_evolution_stats,
        epsilons = EvolutionEpsilons,
        max_agents_per_row = 2
):
    """
    A helper function that calculates Return of Investment (ROI) for applying Epsilon-Greedy Selection Policy.

    :param agent_evolution_stats: statistic about Agent evolution collected in `build_exploration_data'.

    :param epsilons: a list of epsilon values.

    :param max_agents_per_row: how many graphs should be drawn per a row

    :return: a dictionary of Agent ROI after applying Epsilon-Greedy Selection Strategy in the following form:
        {
            'Agent Name': {
                'Epsilon Value': {
                    Metrics.ROI: [an array of ROIs for each ith step (starting from 1st step)]
                }
            }
        }
    """
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) / max_agents_per_row),
        max_agents_per_row,
        figsize = (16, 8),
        squeeze = False
    )
    labels = [("$\epsilon=$" + format_epsilon(epsilon)) for epsilon in epsilons if epsilon != 0.0]

    agent_roi_stats = dict()

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        ax = axs[int(ix / max_agents_per_row), int(ix % max_agents_per_row)]
        agent_stat = agent_evolution_stats[agent_key]
        zero_epsilon_key = format_epsilon(0)
        zero_epsilon = agent_stat[zero_epsilon_key]
        zero_success_evolutions = zero_epsilon[EvolutionCase.SUCCESS]
        zero_failure_evolutions = zero_epsilon[EvolutionCase.FAILURE]
        assert (len(zero_success_evolutions))

        agent_stats = dict()
        roi_mean_means = []
        for epsilon in generate_epsilons():
            if zero_epsilon_key == format_epsilon(epsilon):
                continue

            epsilon_key = format_epsilon(epsilon)
            agent_stats[epsilon_key] = {
                RoiMetrics.ROI_0_025: [],
                RoiMetrics.ROI_MEAN: [],
                RoiMetrics.ROI_0_975: [],
            }
            epsilon_evolutions = agent_stat[epsilon_key]
            success_greedy_evolutions = epsilon_evolutions[EvolutionCase.SUCCESS_GREEDY]
            failure_greedy_evolutions = epsilon_evolutions[EvolutionCase.FAILURE_GREEDY]
            assert (len(success_greedy_evolutions) == len(failure_greedy_evolutions))
            assert (len(zero_success_evolutions) == len(success_greedy_evolutions))
            steps = []
            roi_means = []
            for step in range(1, len(epsilon_evolutions[EvolutionCase.SUCCESS])):
                previous_zero_successes = zero_success_evolutions[step - 1]
                previous_zero_failures = zero_failure_evolutions[step - 1]
                current_zero_successes = zero_success_evolutions[step]
                current_zero_failures = zero_failure_evolutions[step]
                current_epsilon_greedy_successes = success_greedy_evolutions[step]
                current_epsilon_greedy_failures = failure_greedy_evolutions[step]

                def roi_with_confidence_interval(
                        epsilon,
                        previous_zero_successes,
                        previous_zero_failures,
                        current_zero_successes,
                        current_zero_failures,
                        current_epsilon_greedy_successes,
                        current_epsilon_greedy_failures
                ):
                    def roi_formulae(
                            epsilon,
                            previous_zero,
                            current_zero,
                            current_epsilon_greedy
                    ):
                        current_gain = current_epsilon_greedy / (1 - epsilon) - current_zero
                        roi = current_gain / (epsilon * previous_zero)
                        return roi

                    return {
                        RoiMetrics.ROI_SUCCESS: roi_formulae(
                            epsilon,
                            previous_zero_successes,
                            current_zero_successes,
                            current_epsilon_greedy_successes
                        ),
                        RoiMetrics.ROI_FAILURE: roi_formulae(
                            epsilon,
                            previous_zero_failures,
                            current_zero_failures,
                            current_epsilon_greedy_failures
                        )
                    }

                roi_mean = roi_with_confidence_interval(
                    epsilon,
                    previous_zero_successes,
                    previous_zero_failures,
                    current_zero_successes,
                    current_zero_failures,
                    current_epsilon_greedy_successes,
                    current_epsilon_greedy_failures
                )[RoiMetrics.ROI_SUCCESS]
                agent_stats[epsilon_key][RoiMetrics.ROI_MEAN].append(roi_mean)

                roi_means.append(roi_mean)

                steps.append(step)

            roi_mean_means.append(np.mean(roi_means))
            ax.plot(steps, roi_means)

        roi_means_mean = np.mean(roi_mean_means)
        roi_means_div = np.sqrt(np.var(roi_mean_means))
        ax.set_title(
            "$ROI_{t+1}$ of Agent: " + f"'{agent_key}'\n"
            + "$\hat{\mu}_{ROI}="
            + "{0:.5f}".format(round(roi_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}_{ROI}="
            + "{0:.5f}".format(round(roi_means_div, 5))
            + "$"
        )
        ax.legend(labels, loc = 10)
        ax.set_ylabel('ROI')

        agent_roi_stats[agent_key] = agent_stats

    plt.subplots_adjust(hspace = .5)
    plt.show()
    return agent_roi_stats
