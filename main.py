from deep_rl import *

from examples import quantile_regression_dqn_feature, quantile_regression_dqn_pixel

import gym
import os

gym.logger.set_level(40)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(3)

    game = 'BreakoutNoFrameskip-v4'
    quantile_regression_dqn_pixel(game=game)
