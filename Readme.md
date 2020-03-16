# DDPG debbuging
This project attempt to realize how the critic behave given an observation.
I used the pendulum environment because of the continuous single dimension action space.
The code can be easy adjusted for discrete and multi-dimensional environments.

for environment <a src="https://github.com/openai/gym/wiki/Pendulum-v0">information</a>

The project contains two folders:
* <b>train</b>: train and test code for pendulum environment from gym, using ddpg and keras-rl. nothing special here.<br/>
* <b>debug</b>: critic debugging infrastructure. will produce a graph for (action, Q_a) pair given an observation and an environment.

### examples
can be found under debug/out  folder, or by running the debug/critic.py script.
I tested for theta between -2pi and 2pi in steps of pi/2, and theta-dot = 0 for all.

