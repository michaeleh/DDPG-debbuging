import numpy as np
from keras import Input
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from debug.ddpg_callback import DDPGCallback

tb_log_dir = r'train\logs'
from train.nn import build_actor, build_critic
from train.utils import make_env

env = make_env()
np.random.seed(123)
env.seed(123)

assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = build_actor(env, nb_actions)
action_input = Input(shape=(nb_actions,), name='action_input')
critic = build_critic(env, action_input)
tb_callback = DDPGCallback(log_dir=tb_log_dir, critic=critic, env=env)

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


def train():
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=100_000, visualize=False, verbose=2, nb_max_episode_steps=200, callbacks=[tb_callback])

    # After training is done, we save the final weights.
    agent.save_weights('ddpg_weights.h5f', overwrite=True)


def test():
    agent.load_weights('ddpg_weights.h5f')
    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)


if __name__ == "__main__":
    train()
    # test()
