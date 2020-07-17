# Quadrotor Juggling
Keeping a ball in the air by bouncing it off a quadcopter for as many times as possible. We wanted to explore reinforcement learning algorithms.

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

## Policy Gradient Methods Used 
- Simple Policy Gradient 
- Vanilla Policy Gradient
- Proximal Policy Optimization 



![Hector quadrotor with platform on rviz](https://raw.githubusercontent.com/tanishkasingh9/quadrotorjuggling/master/hector_platform.png)
![Hector quadrotor with platform on Gazebo](https://raw.githubusercontent.com/tanishkasingh9/quadrotorjuggling/master/hector_platform_gazebo.png)



*COMING SOON*
1. learning how to fly.
2. Simulate a ball in Gazebo
3. Perform Q-Learning and SARSA to maximize bounces on the platform.



