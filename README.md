# Deep Reinforcement Learning : Navigation

This repository contains my implementation of the [Udacity Deep Reinforcement Learning Nanodegree]((https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)) Project 1 - Navigation

## Project's Description  

For this project, we will train an agent to navigate a large, square world and collect yellow bananas. The world contains both yellow and blue banana as depicted in the animatation below. We want the agent to collect as many yellow bananas as possible while avoiding blue bananas.  

![Navigation animation](images/banana.gif)

### Rewards

1. The agent is given a reward of +1 for collecting a yellow banana
1. Reward of -1 for collecting a blue banana

### State Space  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based precpetion of objects around the agent's foward direction.

### Actions  

Four discrete actions are available, corresponding to:

- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

### The goal

The goal for the project is for the to collect as many yellow bananas as possible while avoiding blue bananas. The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

<br>
<br>

## Getting Started

### The Environment

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents).

 > Note: The project environment for this project is similar to, but **not identical** to the [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment on the Unity ML-Agents GitHub page.

#### Step 1: Clone this Repository

1. Configure your Python environment by following [instructions in the Udacity DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in the [Readme.md](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md)
1. By following the instructions you will have PyTorch, the ML-Agents toolkits, and all the Python packages required to complete the project.
1. (For Windows users) The ML-Agents toolkit supports Windows 10. It has not been test on older version but it may work.

#### Step 2: Download the Unity Environment 

- Install the Unity environment as described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) (The Unity ML-agant environment is already configured by Udacity)

  - Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

#### Step 3: Explore the Environment

Open `Navigation.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

<br>
<br>

## Train a agent

Execute the provided notebook: `Navigation.ipynb`

1. `model.py` implements the Q neural network. This currently contains fully-connected neural network with ReLU activation. You can change the structure of the neural network and play with it
2. `dqn_agent.py` implementss the Agent, and ReplayBuffer
   > Reinforcement learning algorithms use replay buffers to store trajectories of experience when executing a policy in an environment. During training, replay buffers are queried for a subset of the trajectories (either a sequential subset or a sample) to "replay" the agent's experience. [Source](https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial)
   
Playing the game as a human agent is not implemented in this repository.
