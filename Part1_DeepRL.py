#!/usr/bin/env python
# coding: utf-8

# $$
# \newcommand{\mat}[1]{\boldsymbol {#1}}
# \newcommand{\mattr}[1]{\boldsymbol {#1}^\top}
# \newcommand{\matinv}[1]{\boldsymbol {#1}^{-1}}
# \newcommand{\vec}[1]{\boldsymbol {#1}}
# \newcommand{\vectr}[1]{\boldsymbol {#1}^\top}
# \newcommand{\rvar}[1]{\mathrm {#1}}
# \newcommand{\rvec}[1]{\boldsymbol{\mathrm{#1}}}
# \newcommand{\diag}{\mathop{\mathrm {diag}}}
# \newcommand{\set}[1]{\mathbb {#1}}
# \newcommand{\cset}[1]{\mathcal{#1}}
# \newcommand{\norm}[1]{\left\lVert#1\right\rVert}
# \newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
# \newcommand{\bb}[1]{\boldsymbol{#1}}
# \newcommand{\E}[2][]{\mathbb{E}_{#1}\left[#2\right]}
# \newcommand{\ip}[3]{\left<#1,#2\right>_{#3}}
# \newcommand{\given}[]{\,\middle\vert\,}
# \newcommand{\DKL}[2]{\cset{D}_{\text{KL}}\left(#1\,\Vert\, #2\right)}
# \newcommand{\grad}[]{\nabla}
# $$
# # Part 1: Deep Reinforcement Learning
# <a id=part1></a>

# In the tutorial we have seen value-based reinforcement learning, in which we learn to approximate the action-value function $q(s,a)$.
# 
# In this exercise we'll explore a different approach, directly learning the agent's policy distribution, $\pi(a|s)$
# by using *policy gradients*, in order to safely land on the moon!

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import unittest
import os
import sys
import pathlib
import urllib
import shutil
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[2]:


test = unittest.TestCase()
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prefer CPU, GPU won't help much in this assignment
device = 'cpu'
print('Using device:', device)

# Seed for deterministic tests
SEED = 42


# Some technical notes before we begin:
# 
# - This part does not require a GPU. We won't need large models, and the computation bottleneck will be the generation of episodes to train on.
# - In order to run this notebook on the server, you must prepend the `xvfb-run` command to create a virtual screen. For example,
#     - to run the jupyter lab script with `srun` do
#         ```
#         srun -c2 --gres=gpu:1 xvfb-run -a -s "-screen 0 1440x900x24" ./jupyter-lab.sh
#         ```
#     - To run the submission script, do
#         ```
#         srun -c2 xvfb-run -a -s "-screen 0 1440x900x24" python main.py prepare-submission ...
#         ```
#     and so on.
# - The OpenAI `gym` library is not officially supported on windows. However it should be possible to install and run the necessary environment for this exercise. However, we cannot provide you with technical support for this. If you have trouble installing locally, we suggest running on the course server.

# ## Policy gradients
# <a id=part1_1></a>

# Recall from the tutorial that we define the **policy** of an agent as the conditional distribution,
# $$
# \pi(a|s) = \Pr(a_t=a\vert s_t=s),
# $$
# which defines how likely the agent is to take action $a$ at state $s$.
# 
# Furthermore we define the action-value function,
# $$
# q_{\pi}(s,a) = \E{g_t(\tau)|s_t = s,a_t=a,\pi}
# $$
# where 
# $$
# g_t(\tau) = r_{t+1}+\gamma r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+1+k},
# $$
# is the total discounted reward of a specific trajectory $\tau$ from time $t$, and the expectation in $q$ is over all possible
# trajectories,
# $
# \tau=\left\{ (s_0,a_0,r_1,s_1), \dots (s_T,a_T,r_{T+1},s_{T+1}) \right\}.
# $

# In the tutorial we saw that we can learn a value function starting with some random function and
# updating it iteratively by using the **Bellman optimality equation**.
# Given that we have some action-value function, we can immediately create a policy based on that
# by simply selecting an action which maximize the action-value at the current state, i.e.
# $$
# \pi(a|s) =
# \begin{cases}
# 1, & a = \arg\max_{a'\in\cset{A}} q(s,a') \\
# 0, & \text{else}
# \end{cases}.
# $$
# This is called $q$-learning. This approach aims to obtain a policy indirectly through the action-value function.
# Yet, in most cases we don't actually care about knowing the value of particular states,
# since all we need is a good policy for our agent. 
# 
# Here we'll take a different approach and learn a policy distribution $\pi(a|s)$ directly - by using **policy gradients**.

# ### Formalism

# We define a parametric policy, $\pi_\vec{\theta}(a|s)$, and maximize total discounted reward (or minimize the negative reward):
# $$
# \mathcal{L}(\vec{\theta})=\E[\tau]{-g(\tau)|\pi_\vec{\theta}} = -\int g(\tau)p(\tau|\vec{\theta})d\tau,
# $$
# where $p(\tau|\vec{\theta})$ is the probability of a specific trajectory $\tau$ under the policy defined by $\vec{\theta}$.
# 

# Since we want to find the parameters $\vec{\theta}$ which minimize $\mathcal{L}(\vec{\theta})$, we'll compute the gradient w.r.t. $\vec{\theta}$:
# $$
# \grad\mathcal{L}(\vec{\theta}) = -\int g(\tau)\grad p(\tau|\vec{\theta})d\tau.
# $$
# 
# Unfortunately, if we try to write $p(\tau|\vec{\theta})$ explicitly,
# we find that computing it's gradient with respect to $\vec{\theta}$ is
# quite intractable due to a huge product of terms depending on $\vec{\theta}$:
# $$
# p(\tau|\vec{\theta})=p\left(\left\{ (s_t,a_t,r_{t+1},s_{t+1})\right\}_{t\geq0}\given\vec{\theta}\right)
# =p(s_0)\prod_{t\geq0} \pi_{\vec{\theta}}(a_t|s_t)p(s_{t+1}|s_t,a_t).
# $$

# However, by using the fact that $\grad_{x}\log(f(x))=\frac{\grad_{x}f(x)}{f(x)}$, we can convert the product into a sum:
# $$
# \begin{align}
# \grad\mathcal{L}(\vec{\theta})
# &= -\int g(\tau)\grad p(\tau|\vec{\theta})d\tau
# = -\int g(\tau)\frac{\grad p(\tau|\vec{\theta})}{p(\tau|\vec{\theta})}p(\tau|\vec{\theta})d\tau \\
# &= -\int g(\tau)\grad\log\left(p(\tau|\vec{\theta})\right)p(\tau|\vec{\theta})d\tau \\
# &= -\int g(\tau)\grad\log\left( p(s_0)\prod_{t\geq0} \pi_{\vec{\theta}}(a_t|s_t)p(s_{t+1}|s_t,a_t) \right)
# p(\tau|\vec{\theta})d\tau \\
# &= -\int g(\tau)\grad\left( \log p(s_0) + \sum_{t\geq0} \log \pi_{\vec{\theta}}(a_t|s_t) + 
# \sum_{t\geq0}\log p(s_{t+1}|s_t,a_t) \right) p(\tau|\vec{\theta})d\tau \\
# &= -\int g(\tau)\sum_{t\geq0} \grad\log \pi_{\vec{\theta}}(a_t|s_t) p(\tau|\vec{\theta})d\tau \\
# &= \E[\tau]{-g(\tau)\sum_{t\geq0} \grad\log \pi_{\vec{\theta}}(a_t|s_t)}.
# \end{align}
# $$

# This is the "vanilla" version of the policy gradient. We can interpret is as a weighted log-likelihood function.
# The log-policy is the log-likelihood term we wish to maximize and the total discounted reward acts as a weight: high-return positive
# trajectories will cause the probability of actions taken during them to increase, and negative-return trajectories will cause the
# probabilities of actions taken to decrease.
# 
# In the following figures we see three trajectories: high-return positive-reward (green), low-return positive-reward (yellow) and negative-return (red) and the action probabilities along the trajectories after the update. Credit: Sergey Levine.
# 
# |<strong></strong>||
# |-----| ----|
# |<img src="imgs/pg1.png" height="200">|<img src="imgs/pg2.png" height="200">|
# 

# The major drawback of the policy-gradient is it's high variance, which causes erratic optimization behavior and therefore slow convergence.
# One reason for this is that the log-policy weight term, $g(\tau)$ can vary wildly between different trajectories, even if they're similar in
# actions. Later on we'll implement the loss and explore some methods of variance reduction.

# ### Landing on the moon with policy gradients

# In the spirit of the recent achievements of the Israeli space industry,
# we'll apply our reinforcement learning skills to solve a simple game called **LunarLander**.
# 
# This game is available as an `environment` in OpenAI `gym`.
# 
# <video loop autoplay src="http://gym.openai.com/videos/2019-04-06--My9IiAbqha/LunarLander-v2/original.mp4" />

# In this environment, you need to control the lander and get it to land safely on the moon.
# To do so, you must apply bottom, right or left thrusters (each are either fully on or fully off)
# and get it to land within the designated zone as quickly as possible and with minimal wasted fuel.

# In[3]:


import gym

# Just for fun :) ... but also to re-define the default max number of steps
ENV_NAME = 'Beresheet-v2'
MAX_EPISODE_STEPS = 300
if ENV_NAME not in gym.envs.registry.env_specs:
    gym.register(
        id=ENV_NAME,
        entry_point='gym.envs.box2d:LunarLander',
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=200,
    )


# In[4]:


import gym

env = gym.make(ENV_NAME)

print(env)
print(f'observations space: {env.observation_space}')
print(f'action space: {env.action_space}')

ENV_N_ACTIONS = env.action_space.n
ENV_N_OBSERVATIONS = env.observation_space.shape[0]


# The observations at each step is the Lander's position, velocity, angle, angular velocity and ground contact state.
# The actions are no-op, fire left truster, bottom thruster and right thruster.
# 
# You are **highly encouraged** to read the [documentation](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py) in the source code of the `LunarLander` environment to understand the reward system,
# and see how the actions and observations are created.

# ### Policy network and Agent

# Let's start with our policy-model. This will be a simple neural net, which should take an observation and return a score for each possible action.

# **TODO**:
# 1. Implement all methods in the `PolicyNet` class in the `hw4/rl_pg.py` module.
#    Start small. A simple MLP with a few hidden layers is a good starting point. You can come back and change it later based on the the experiments. The we'll use the `build_for_env` method to instantiate a `PolicyNet` based on the configuration of a given environment.
# 2. If you need hyperparameters to configure your model (e.g. number of hidden layers, sizes, etc.), add them in `part1_pg_hyperparams()` in `hw4/answers.py`.

# In[5]:


import hw4.rl_pg as hw4pg
import hw4.answers

hp = hw4.answers.part1_pg_hyperparams()

# You can add keyword-args to this function which will be populated from the hyperparameters dict.
p_net = hw4pg.PolicyNet.build_for_env(env, device, **hp)
p_net


# Now we need an **agent**. The purpose of our agent will be to act according to the current policy and generate experiences.
# Our `PolicyAgent` will use a `PolicyNet` as the current policy function.
# 
# 
# We'll also define some extra datatypes to help us represent the data generated by our agent.
# You can find the `Experience`, `Episode` and `TrainBatch` datatypes in the `hw4/rl_pg.py` module.

# **TODO**: Implement the `current_action_distribution()` method of the `PolicyAgent` class in the `hw4/rl_pg.py` module.

# In[6]:


for i in range (10):
    agent = hw4pg.PolicyAgent(env, p_net, device)
    d = agent.current_action_distribution()
    test.assertSequenceEqual(d.shape, (env.action_space.n,))
    test.assertAlmostEqual(d.sum(), 1.0, delta=1e-5)
    
print(d)


# **TODO**: Implement the `step()` method of the `PolicyAgent`.

# In[7]:


agent = hw4pg.PolicyAgent(env, p_net, device)
exp = agent.step()

test.assertIsInstance(exp, hw4pg.Experience)
print(exp)


# To test our agent, we'll write some code that allows it to play an environment. We'll use the `Monitor`
# wrapper in `gym` to generate a video of the episode for visual debugging.

# **TODO**: Complete the implementation of the `monitor_episode()` method of the `PolicyAgent`.

# In[8]:


env, n_steps, reward = agent.monitor_episode(ENV_NAME, p_net, device=device)


# To display the Monitor video in this notebook, we'll use a helper function from our `jupyter_utils` and a small wrapper that extracts the path of the last video file. 

# In[9]:


import cs236781.jupyter_utils as jupyter_utils

def show_monitor_video(monitor_env, idx=0, **kw):
    # Extract video path
    video_path = monitor_env.videos[idx][0]
    video_path = os.path.relpath(video_path, start=os.path.curdir)
    
    # Use helper function to embed the video
    return jupyter_utils.show_video_in_notebook(video_path, **kw)


# In[10]:


print(f'Episode ran for {n_steps} steps. Total reward: {reward:.2f}')

show_monitor_video(env)


# ### Training data

# The next step is to create data to train on.
# We need to train on batches of state-action pairs, so that our network can learn to predict the actions.
# 
# We'll split this task into three parts:
# 1. Generate a batch of `Episode`s, by using an `Agent` that's playing according to our current policy network.
#    Each `Episode` object contains the `Experience` objects created by the agent.
# 2. Calculate the total discounted reward for each state we encountered and action we took. This is our action-value estimate.
# 3. Convert the `Episode`s into a batch of tensors to train on.
#    Each batch will contain states, action taken per state, reward accrued, and the calculated estimated state-values.
#    These will be stored in a `TrainBatch` object.
# 

# **TODO**: Complete the implementation of the `episode_batch_generator()` method in the `TrainBatchDataset` class within the `hw4.rl_data` module. This will address part 1 in the list above.

# In[11]:


import hw4.rl_data as hw4data

def agent_fn():
    env = gym.make(ENV_NAME)
    hp = hw4.answers.part1_pg_hyperparams()
    p_net = hw4pg.PolicyNet.build_for_env(env, device, **hp)
    return hw4pg.PolicyAgent(env, p_net, device)
    
ds = hw4data.TrainBatchDataset(agent_fn, episode_batch_size=8, gamma=0.9)
batch_gen = ds.episode_batch_generator()
b = next(batch_gen)
print('First episode:', b[0])

test.assertEqual(len(b), 8)
for ep in b:
    test.assertIsInstance(ep, hw4data.Episode)
    
    # Check that it's a full episode
    is_done = [exp.is_done for exp in ep.experiences]
    test.assertFalse(any(is_done[0:-1]))
    test.assertTrue(is_done[-1])


# **TODO**: Complete the implementation of the `calc_qvals()` method in the `Episode` class.
# This will address part 2.
# These q-values are an estimate of the actual action value function: $\hat{q}_{t} = \sum_{t'\geq t} \gamma^{t'}r_{t'+1}$.

# In[12]:


np.random.seed(SEED)
test_rewards = np.random.randint(-10, 10, 100)
test_experiences = [hw4pg.Experience(None,None,r,False) for r in test_rewards] 
test_episode = hw4data.Episode(np.sum(test_rewards), test_experiences)

qvals = test_episode.calc_qvals(0.9)
qvals = list(qvals)

expected_qvals = np.load(os.path.join('tests', 'assets', 'part1_expected_qvals.npy'))
for i in range(len(test_rewards)):
    test.assertAlmostEqual(expected_qvals[i], qvals[i], delta=1e-3)


# **TODO**: Complete the implementation of the `from_episodes()` method in the `TrainBatch` class.
# This will address part 3.
# 
# Notes:
# - The `TrainBatchDataset` class provides a generator function that will use the above function to lazily generate batches of training samples and labels on demand.
# - This allows us to use a standard `PyTorch` dataloader to wrap our Dataset and provide us with parallel data loading for free!
#   This means we can run multiple environments with multiple agents in separate background processes to generate data for training and thus prevent the data loading bottleneck which is caused by the fact that we must generate full Episodes to train on in order to calculate the q-values.
# - We'll set the `DataLoader`'s `batch_size` to `None` because we have already implemented custom batching in our dataset.

# In[13]:


from torch.utils.data import DataLoader

ds = hw4data.TrainBatchDataset(agent_fn, episode_batch_size=8, gamma=0.9)
dl = DataLoader(ds, batch_size=None, num_workers=2) # Run multiple agents/env in separate worker process


for i, train_batch in enumerate(dl):
    states, actions, qvals, reward_mean = train_batch
    print(f'#{i}: {train_batch}')
    test.assertEqual(states.shape[0], actions.shape[0])
    test.assertEqual(qvals.shape[0], actions.shape[0])
    test.assertEqual(states.shape[1], env.observation_space.shape[0])
    if i > 5:
        break


# ### Loss functions

# As usual, we need a loss function to optimize over.
# We'll calculate three types of losses:
# 1. The causal vanilla policy gradient loss.
# 1. The policy gradient loss, with a baseline to reduce variance.
# 2. An entropy-based loss whos purpose is to diversify the agent's action selection,
#    and prevent it from being "too sure" about its actions.
#    This loss will be used together with one of the above losses.

# #### Causal vanilla policy-gradient

# We have derived the policy-gradient as
# $$
# \grad\mathcal{L}(\vec{\theta}) = \E[\tau]{-g(\tau)\sum_{t\geq0} \grad\log \pi_{\vec{\theta}}(a_t|s_t)}.
# $$
# 
# By writing the discounted reward explicitly and enforcing causality, i.e. the action taken at time $t$ can't affect
# the reward at time $t'<t$, we can get a slightly lower-variance version of the policy gradient:
# 
# $$
# \grad\mathcal{L}_{\text{PG}}(\vec{\theta}) = 
# \E[\tau]{-\sum_{t\geq0} \left(\sum_{t'\geq t} \gamma^{t'}r_{t'+1} \right)\grad\log \pi_{\vec{\theta}}(a_t|s_t)}.
# $$

# In practice, the expectation over trajectories is calculated using a Monte-Carlo approach, i.e. simply sampling $N$
# trajectories and average the term inside the expectation. Therefore, we will use the following estimated version of the policy gradient:
# 
# $$
# \begin{align}
# \hat\grad\mathcal{L}_{\text{PG}}(\vec{\theta})
# &=-\frac{1}{N}\sum_{i=1}^{N}\sum_{t\geq0} \left(\sum_{t'\geq t} \gamma^{t'}r_{i,t'+1} \right)\grad\log \pi_{\vec{\theta}}(a_{i,t}|s_{i,t}) \\
# &=-\frac{1}{N}\sum_{i=1}^{N}\sum_{t\geq0} \hat{q}_{i,t} \grad\log \pi_{\vec{\theta}}(a_{i,t}|s_{i,t}).
# \end{align}
# $$
# 
# Note the use of the notation $\hat{q}_{i,t}$ to represent the estimated action-value at time $t$ in the sampled trajectory $i$.
# Here $\hat{q}_{i,t}$ is acting as the weight-term for the policy gradient.

# **TODO**: Complete the implementation of the `VanillaPolicyGradientLoss` class in the `hw4/rl_pg.py` module.

# In[14]:


# Ensure deterministic run
env = gym.make(ENV_NAME)
env.seed(SEED)
torch.manual_seed(SEED)

def agent_fn():
    # Use a simple "network" here, so that this test doesn't depend on your specific PolicyNet implementation
    p_net_test = nn.Linear(8, 4)
    agent = hw4pg.PolicyAgent(env, p_net_test)
    return agent

dataloader = hw4data.TrainBatchDataset(agent_fn, gamma=0.9, episode_batch_size=4)

test_batch = next(iter(dataloader))
test_action_scores = torch.randn(len(test_batch), env.action_space.n)

loss_fn_p = hw4pg.VanillaPolicyGradientLoss()
loss_p, _ = loss_fn_p(test_batch, test_action_scores)

print('loss =', loss_p)
test.assertAlmostEqual(loss_p.item(), -35.535522, delta=1e-3)


# #### Policy-gradient with baseline

# Another way to reduce the variance of our gradient is to use relative weighting of the log-policy instead of absolute reward values.
# $$
# \hat\grad\mathcal{L}_{\text{BPG}}(\vec{\theta})
# =-\frac{1}{N}\sum_{i=1}^{N}\sum_{t\geq0} \left(\hat{q}_{i,t}-b\right) \grad\log \pi_{\vec{\theta}}(a_{i,t}|s_{i,t}).
# $$
# In other words, we don't measure a trajectory's worth by it's total reward, but by how much better that total reward is relative to some
# expected ("baseline") reward value, denoted above by $b$.
# Note that subtracting a baseline has no effect on the expected value of the policy gradient. It's easy to prove this directly by definition.
# 
# Here we'll implement a very simple baseline (not optimal in terms of variance reduction): the average of the estimated state-values $\hat{q}_{i,t}$.

# **TODO**: Complete the implementation of the `BaselinePolicyGradientLoss` class in the `hw4/rl_pg.py` module.

# In[15]:


# Using the same batch and action_scores from above cell
loss_fn_p = hw4pg.BaselinePolicyGradientLoss()
loss_p, _ = loss_fn_p(test_batch, test_action_scores)

print('loss =', loss_p)
test.assertAlmostEqual(loss_p.item(), -2.4665009, delta=1e-3)


# #### Entropy loss

# The entropy of a probability distribution (in our case the policy), is
# $$
# H(\pi) = -\sum_{a} \pi(a|s)\log\pi(a|s).
# $$
# The entropy is always positive and obtains it's maximum for a uniform distribution.
# We'll use the entropy of the policy as a bonus, i.e. we'll try to maximize it.
# The idea is the prevent the policy distribution from becoming too narrow and thus promote the agent's exploration.

# First, we'll calculate the maximal possible entropy value of the action distribution for a set number of possible actions.
# This will be used as a normalization term.
# 
# **TODO**: Complete the implementation of the `calc_max_entropy()` method in the `ActionEntropyLoss` class.

# In[16]:


loss_fn_e = hw4pg.ActionEntropyLoss(env.action_space.n)
print('max_entropy = ', loss_fn_e.max_entropy)

test.assertAlmostEqual(loss_fn_e.max_entropy, 1.38629436, delta=1e-3)


# **TODO**: Complete the implementation of the `forward()` method in the `ActionEntropyLoss` class.

# In[17]:


loss_e, _ = loss_fn_e(test_batch, test_action_scores)
print('loss = ', loss_e)

test.assertAlmostEqual(loss_e.item(), -0.7927002, delta=1e-3)


# ### Training

# We'll implement our training procedure as follows:
# 
# 1. Initialize the current policy to be a random policy.
# 1. Sample $N$ trajectories from the environment using the current policy.
# 2. Calculate the estimated $q$-values, $\hat{q}_{i,t} = \sum_{t'\geq t} \gamma^{t'}r_{i,t'+1}$ for each trajectory $i$.
# 3. Calculate policy gradient estimate $\hat\grad\mathcal{L}(\vec{\theta})$ as defined above.
# 4. Perform SGD update $\vec{\theta}\leftarrow\vec{\theta}-\eta\hat\grad\mathcal{L}(\vec{\theta})$.
# 5. Repeat from step 2.
# 
# This is known as the **REINFORCE** algorithm.

# Fortunately, we've already implemented everything we need for steps 1-4 so we need only a bit more code to put it all together.
# 
# The following block implements a wrapper, `train_pg` to create all the objects we need in order to train our policy gradient model.
# 

# In[18]:


import hw4.answers
from functools import partial

ENV_NAME = "Beresheet-v2"

def agent_fn_train(agent_type, p_net, seed, envs_dict):
    winfo = torch.utils.data.get_worker_info()
    wid = winfo.id if winfo else 0
    seed = seed + wid if seed else wid

    env = gym.make(ENV_NAME)
    envs_dict[wid] = env
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return agent_type(env, p_net)

def train_rl(agent_type, net_type, loss_fns, hp, seed=None, checkpoints_file=None, **train_kw):
    print(f'hyperparams: {hp}')
    
    envs = {}
    p_net = net_type(ENV_N_OBSERVATIONS, ENV_N_ACTIONS, **hp)
    p_net.share_memory()
    agent_fn = partial(agent_fn_train, agent_type, p_net, seed, envs)
    
    dataset = hw4data.TrainBatchDataset(agent_fn, hp['batch_size'], hp['gamma'])
    dataloader = DataLoader(dataset, batch_size=None, num_workers=2)
    optimizer = optim.Adam(p_net.parameters(), lr=hp['learn_rate'], eps=hp['eps'])
    
    trainer = hw4pg.PolicyTrainer(p_net, optimizer, loss_fns, dataloader, checkpoints_file)
    try:
        trainer.train(**train_kw)
    except KeyboardInterrupt as e:
        print('Training interrupted by user.')
    finally:
        for env in envs.values():
            env.close()

    # Include final model state
    training_data = trainer.training_data
    training_data['model_state'] = p_net.state_dict()
    return training_data
    
def train_pg(baseline=False, entropy=False, **train_kwargs):
    hp = hw4.answers.part1_pg_hyperparams()
    
    loss_fns = []
    if baseline:
        loss_fns.append(hw4pg.BaselinePolicyGradientLoss())
    else:
        loss_fns.append(hw4pg.VanillaPolicyGradientLoss())
    if entropy:
        loss_fns.append(hw4pg.ActionEntropyLoss(ENV_N_ACTIONS, hp['beta']))

    return train_rl(hw4pg.PolicyAgent, hw4pg.PolicyNet, loss_fns, hp, **train_kwargs)


# The `PolicyTrainer` class implements the training loop, collects the losses and rewards and provides some useful checkpointing functionality.
# The training loop will generate batches of episodes and train on them until either:
# - The average total reward from the last `running_mean_len` episodes is greater than the `target_reward`, OR
# - The number of generated episodes reached `max_episodes`.
# 
# Most of this class is already implemented for you. 

# **TODO**:
# 1. Complete the training loop by implementing the `train_batch()` method of the `PolicyTrainer`.
# 2. Tweak the hyperparameters in the `part1_pg_hyperparams()` function within the `hw4/answers.py` module as needed. You get some sane defaults.

# Let's check whether our model is actually training.
# We'll try to reach a very low (bad) target reward, just as a sanity check to see that training works.
# Your model should be able to reach this target reward within a few batches.
# 
# You can increase the target reward and use this block to manually tweak your model and hyperparameters a few times.

# In[ ]:


target_reward = 0 # VERY LOW target
train_data = train_pg(target_reward=target_reward, seed=SEED, max_episodes=4000, running_mean_len=10)

test.assertGreater(train_data['mean_reward'][-1], target_reward)


# ### Experimenting with different losses

# We'll now run a few experiments to see the effect of diferent loss functions on the training dynamics. Namely, we'll try:
# 1. Vanilla PG (`vpg`): No baseline, no entropy
# 2. Baseline PG (`bpg`): Baseline, no entropy loss
# 3. Entropy PG (`epg`): No baseline, with entropy loss
# 3. Combined PG (`cpg`): Baseline, with entropy loss

# In[ ]:
