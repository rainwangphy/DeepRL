from deep_rl import *

from examples import quantile_regression_dqn_feature

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(-1)

    game = 'CartPole-v0'
    quantile_regression_dqn_feature(game=game)
