import matplotlib.pyplot as plt
import numpy as np
from keras import Input

from train.nn import build_critic
from train.utils import make_env


class CriticDebugger:

    def __init__(self, weight_path, env, delta=0.01) -> None:
        super().__init__()
        nb_actions = env.action_space.shape[0]
        action_input = Input(shape=(nb_actions,), name='action_input')
        self.critic = build_critic(env, action_input)
        self.critic.load_weights(weight_path)
        self.action_range = np.arange(env.action_space.low, env.action_space.high, delta)

    def forward(self, ob):
        actions = np.array([np.array([action]) for action in self.action_range])
        obs = np.array([np.array([ob]) for _ in self.action_range])
        return self.critic.predict_on_batch([actions, obs]).flatten()

    def save_fig(self, results, theta):
        plt.plot(self.action_range, results)
        plt.xlabel('action')
        plt.ylabel('critic score')
        plt.title(f'theta = {theta}')
        plt.savefig(f"out\\theta_{theta}_plot.png")
        plt.close()


path = r"..\train\ddpg_weights_critic.h5f"
debug_env = make_env()
c = CriticDebugger(path, env=make_env())

# full circle, the goal is to remain at zero angle (vertical),
for theta in np.arange(-2 * np.pi, 2 * np.pi, np.pi / 2):
    debug_env.env.state = np.array([theta, 0])
    obs = debug_env.env._get_obs()
    reviews = c.forward(obs)
    c.save_fig(reviews, theta)
