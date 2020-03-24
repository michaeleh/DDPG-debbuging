import gym


def make_env(name='Pendulum-v0'):
    # Get the environment and extract the number of actions.
    env = gym.make(name)
    env.reset()
    return env
