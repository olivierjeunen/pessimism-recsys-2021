import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import OrganicUserEventCounterAgent, organic_user_count_args
from agents import SkylineAgent, skyline_args
from agents import RandomAgent, random_args
from reco_gym import (
    Configuration,
    build_agent_init,
    env_1_args,
    gather_agent_stats,
    plot_agent_stats
)

from agents import RidgeDMAgent, ridge_args
from datetime import datetime

def grid_search_parameter(
        parameter_name, parameter_values,
        agent_name, ridge_args,
        env_args, extra_env_args,
        train_samples=10000, valid_samples=10000, valid_runs=5,
):
    ''' Perform a grid search by doing an A/B test with model variants that use different parameter values, log the result and return the optimal parameter value '''
    print(f'\tGrid search for {agent_name}')
    grid_search_agent_inits = dict()
    for parameter_value in parameter_values:
        agent_name_with_specified_parameter = f'{agent_name} ({parameter_name} = {parameter_value})'
        grid_search_agent_inits.update(
            build_agent_init(
                agent_name_with_specified_parameter,
                RidgeDMAgent,
                {
                    **ridge_args,
                    parameter_name: parameter_value
                }
            )
        )

    # Run A/B test with model variants
    agent_stats = gather_agent_stats(
        env,
        env_args,
        {
            **extra_env_args
        },
        grid_search_agent_inits,
        [train_samples],
        valid_samples,
        valid_runs,
        StatEpochsNewRandomSeed
    )

    # Extract netrics and turn into DataFrame
    gg = agent_stats[list(agent_stats.keys())[1]]
    l = []
    for k in list(gg.keys()):
        f = pd.DataFrame(gg[k])
        f['Alg'] = k
        f['Samples'] = agent_stats[list(agent_stats.keys())[0]]
        l.append(f)
    agent_stats_df = pd.concat(l)
    agent_stats_df.columns = [str(c) for c in agent_stats_df.columns]
    values = agent_stats_df['AgentStats.Q0_500'].values
    optimal_parameter_value = parameter_values[np.argmax(values)]
    print(agent_stats_df)
    agent_stats_df.to_csv(f'GS_pessimism_{agent_name.replace(" ", "_")}_{num_products}_products_pop_eps{logging_epsilon}.csv', index=False)
    print(f'\t... optimal {parameter_name} = {optimal_parameter_value}!')
    return optimal_parameter_value

# General parameters for experiments
RandomSeed = 42
TrainingDataSamples = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
TestingDataSamples = 10000 
StatEpochs = 5
StatEpochsNewRandomSeed = True

for num_products in [10]: 
    for logging_epsilon in [.0, 1e-6, 1e-4, 1e-2, 1.]:
        print('#' * 48)
        print(f'{datetime.now()}\tLogging Popularity-Epsilon-Greedy with {num_products} products and epsilon {logging_epsilon}...')
        print('#' * 48)

        # Environment parameters
        std_env_args = {
            **env_1_args,
            'random_seed': RandomSeed,
            'num_products': num_products,
        }
        env = gym.make('reco-gym-v1')

        # Logging policy
        logger = OrganicUserEventCounterAgent(Configuration({
            **organic_user_count_args,
            **std_env_args,
            'select_randomly': True,
            'epsilon': logging_epsilon
        }))

        std_extra_env_args = {
            'num_products': num_products,
            'number_of_flips': num_products // 2,
            'agent': logger,
        }

        # Initialisation of different agents
        logging_agent = build_agent_init(
            'Logging',
            OrganicUserEventCounterAgent,
            {
                **organic_user_count_args,
                **std_env_args,
                'select_randomly': True,
                'reverse_pop': False,
                'epsilon': logging_epsilon
            }
        )

        skyline_agent = build_agent_init(
            'Skyline',
            SkylineAgent,
            {
                **skyline_args,
            }
        )

        MLE_agent = build_agent_init(
            'MLE',
            RidgeDMAgent,
            {
                **ridge_args,
                'subsample_negatives': 1.,
                'mode': 'mean',
                'l2': False
            }
        )

        MAP_agent = build_agent_init(
            'MAP',
            RidgeDMAgent,
            {
                **ridge_args,
                'subsample_negatives': 1.,
                'mode': 'mean',
                'l2': True
            }
        )

        #########################
        # Grid search for alpha # 
        #########################
        alphas_lcb = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45]
        ridge_lcb_args = {
            **ridge_args,
            'mode':  'LCB'
        }
        optimal_lcb_strength = grid_search_parameter(
            'alpha', alphas_lcb,
            'Ridge Regression LCB', ridge_lcb_args,
            std_env_args, std_extra_env_args,
        )

        ridge_lcb_agent = build_agent_init(
            'LCB (alpha = {0})'.format(optimal_lcb_strength),
            RidgeDMAgent,
            {
                **ridge_args,
                'mode': 'LCB',
                'alpha': optimal_lcb_strength
            }
        )

        agent_inits = {
            **ridge_lcb_agent,
            **MLE_agent,
            **MAP_agent,
            **logging_agent,
            **skyline_agent,
        }

        # Gathering performance of agents for the logging policy: uniform.
        agent_stats01 = gather_agent_stats(
            env,
            std_env_args,
            {
                'num_products': num_products,
                'number_of_flips': num_products // 2,
                'agent': logger
            },
            agent_inits,
            TrainingDataSamples,
            TestingDataSamples,
            StatEpochs,
            StatEpochsNewRandomSeed
        )

        plot_agent_stats(agent_stats01, figname='performance_pessimism_{0}_products_popepsgreedy_eps_{1}.png'.format(num_products,logging_epsilon))

        def tocsv(stats, file_name):
            import pandas as pd
            gg = stats[list(stats.keys())[1]]

            l = []
            for k in list(gg.keys()):
                f = pd.DataFrame(gg[k])
                f['Alg'] = k.replace('aa', 'a').replace('bb', 'b')
                f['Samples'] = stats[list(stats.keys())[0]]
                l.append(f)
            pd.concat(l).to_csv(file_name)

        tocsv(agent_stats01, 'performance_pessimism_{0}_products_popepsgreedy_eps_{1}.csv'.format(num_products,logging_epsilon))
