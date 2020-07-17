# Quadrotor Juggling
Keeping a ball in the air by bouncing it off a quadcopter for as many times as possible. We wanted to explore reinforcement learning algorithms.

## Algorithms in action 
|      SARSA      |         VPG       |       PPO        |
| :---------------| :---------------: | ---------------: |
|  <img src="https://raw.githubusercontent.com/tanishkasingh9/quadrotorjuggling/master/demo/sarsa.gif"> | <img src="https://raw.githubusercontent.com/tanishkasingh9/quadrotorjuggling/master/demo/vpg.gif">  | <img src="https://raw.githubusercontent.com/tanishkasingh9/quadrotorjuggling/master/demo/ppo.gif">|

### Team Members for this project
- Tanishka Singh (tsingh22@asu.edu)
- Deepak Kala Vasudevan (dkalavas@asu.edu)
- Nikhil Agarwal (nagarw22@asu.edu)

### Dependencies
We recommend using [Ubuntu 16](http://releases.ubuntu.com/16.04/) to run the code.
- Install latest version of [V-Rep Pro Edu](http://www.coppeliarobotics.com/downloads.html)
- [Python 2.7](https://www.python.org/downloads/release/python-2715/) is required
- Install latest version of tensorflow using [pip install tensorflow](https://www.tensorflow.org/install)

### Running the Quadcopter environment on VREP Simulator
Navigate to where simulator is downloaded and use path of provided environment file and run:

```
./vrep.sh quad_env.ttt
```
To run in headless mode

```
./vrep.sh -h quad_env.ttt
```
### To run the code
Download and unzip the code, navigate to the unzipped folder and run:
```
$ python main.py [algorithm] [action] [number of episodes] [steps per episode]
```
| options | values |
| ------ | ------ |
| algorithm | pg or vpg or ppo |
| action | eval or train |
| number of episodes | default = 200 |
| steps per episode | default = 50 |

- zz_GraveYard.zip contains code that we worked on initially and later abandoned as we could not resolve issues. (uses ros, gazebo, sphinx)

## Policy Gradient Methods
A class of reinforcement learning techniques that rely upon optimizing parametrized policies with respect to the expected return (long-term cumulative reward) by gradient descent. The actor directly learns the policy function that map states to actions

- Simple Policy Gradient (SARSA)
- Vanilla Policy Gradient (VPG)
- Proximal Policy Optimization (PPO)





