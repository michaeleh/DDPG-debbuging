import itertools
import os

import numpy as np
import pandas as pd
from gym import Env
from keras import Model
from keras.callbacks import Callback

BINS = 5


def _gen_bins(space):
    bounds = [np.array(bound) for bound in zip(space.low.tolist(), space.high.tolist())]
    bins = []
    for bound in bounds:
        bound_min = bound[0]
        bound_max = bound[1]
        bins_i = np.array([x for x in np.linspace(start=bound_min, stop=bound_max, num=BINS)])
        bins.append(bins_i)
    return np.array(bins)


def _get_bin(o, b):
    return tuple(b[_get_bin_index(b, x)] for x, b in zip(o, b))


def _get_bin_index(b, x):
    idx = np.digitize(x, b, right=True)
    if idx >= len(b):
        idx = len(b) - 1
    return idx


class DDPGCallback(Callback):

    def __init__(self, critic: Model, env: Env, log_dir='./logs'):
        """
        a callback to be executed after each batch.
        I am to achieve 2 goals:
        1. for each state s_i plot graph where (x=action), (y=number of time action a taken in s_i).
        2. for each (state,action) pair, plot the change of Q(s,a) over time.

        For start we are gonna deal with continuous state and action space, so we'll have to divide it to bins so it
        will be discrete.
        The amount of bins is controlled by the {@BINS} constant.
        :param critic: critic model, used for Q(s,a) evaluation over time
        :param env: gym env for observation and action space
        :param log_dir: where to save logs.
        """
        super().__init__()
        self.critic = critic
        self.env = env
        self.log_dir = log_dir
        self.state_action_q_val = dict()
        self.obs_action_counter = dict()
        # generate n_d bins
        self.obs_bins = _gen_bins(env.observation_space)
        self.action_bins = _gen_bins(env.action_space)

    def on_train_begin(self, logs=None):
        """
        init dicts to defaults.
        for each possible bin combination
        """
        # combine using sets product all possible bins
        combo = lambda bins: list(itertools.product(*list(bins)))  # BIN ^ obs_dim
        # creating all possible bins for states
        states_combo_bins = combo(self.obs_bins)
        # creating all possible bins for action
        actions_combo = combo(self.action_bins)

        states_actions = combo([states_combo_bins, actions_combo])
        self.obs_action_counter = {(state, action): 0 for state, action in states_actions}
        self.state_action_q_val = {(state, action): [] for state, action in states_actions}

    def _get_obs_bin(self, obs):
        return _get_bin(obs, self.obs_bins)

    def _get_action_bin(self, action):
        return _get_bin(action, self.action_bins)

    def _calc_q(self, obs, action):
        obs_arr = np.array([np.array([np.array(obs)])])
        action_arr = np.array(action)
        return self.critic.predict([action_arr, obs_arr])[0][0]

    def on_batch_end(self, batch, logs=None):
        """
        save data after each step
        :param batch: n_step, ignored for now
        :param logs: step logs, using the observation and action
        :return: update dicts data
        """
        action = logs['action']
        obs = logs['observation']
        a_bin = self._get_action_bin(action)
        o_bin = self._get_obs_bin(obs)
        # increase state action counter
        self.obs_action_counter[(o_bin, a_bin)] += 1
        # update Q(obs,action) history
        # NOTE: i calculate the Q for the bin, NOT the actual state action pair!
        self.state_action_q_val[(o_bin, a_bin)].append(self._calc_q(o_bin, a_bin))

    def on_train_end(self, logs=None):
        """
        save to csv
        """

        df = pd.DataFrame(columns=['state', 'action', 'c', 'Q'])
        i = 0
        for state_action, q in self.state_action_q_val.items():
            df.loc[i] = [np.array(state_action[0]), np.array(state_action[1]), self.obs_action_counter[state_action],
                         q]
            i += 1
        df.to_csv(os.path.join(self.log_dir, 'ddpg_data.csv'))
        df.to_pickle(os.path.join(self.log_dir, 'ddpg_data.pkl'))
