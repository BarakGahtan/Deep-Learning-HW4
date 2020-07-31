r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======

    # hp['hidden_layers'] = [256, 1024]
    # hp['gamma'] = 0.9

    # hp['hidden_layers'] = [256, 512]
    # hp['gamma'] = 0.85
    # hp['beta'] = 0.3

    hp['hidden_layers'] = [512, 256]
    # hp['batch_size'] = 64
    hp['gamma'] = 0.995
    hp['beta'] = 0.7
    hp['learn_rate'] = 0.001
    # hp['eps'] = 0.05

    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp['hidden_layers'] = [32, 32]
    hp['batch_size'] = 64
    hp['gamma'] = 0.95
    hp['beta'] = 0.8
    hp['delta'] = 0.7
    hp['learn_rate'] = 0.008
    # hp['eps'] = 0.05
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

The advantage is the amount of how much the current action is better than what we would usually do in that state.
In policy gradient methods, we update the policy in the direction of received reward.
However, in tcomplicated task, the policy may recive different rewards. Therefor, the policy
may need to collect many experiances and average them over the different rewards.
With that said, the advantage has lower variance since the baseline compensates for the variance introduced by
being in different states.
"""


part1_q2 = r"""
**Your answer:**
The estimated q-values is calculated in order to estimate the value of the next move. 
If we agree that the estimate q-value is actually giving a good estimation, therefore,
it can be used to estimate a state value. 

"""


part1_q3 = r"""
**Your answer:**
1. In the first experiament we tried to compare between 4 different losses. 
The loss_p graph represent the loss of each function. while the loss_e represent
only the entropy loss (only for cpg and epg). They both influence over the mean_reward graph which
represent the rewards in each iteration for each loss function.
It is abvious that the bpg got the best resault while the cpg falls after around 2500 episodes.
The vpg is failing and doesn't improve after the 1000 episodes and ath epg remaines relatively constant.


2. Around 3000 episodes the aac is overcoming the cpg in terms of mean rewards. The fall of the
cpg is aligned to the base line graph which falls around that amount of episodes. Moreover,
the loss_p graph shows that the cpg loss is increasing from (around) 3000 episodes while in that 
time the aac loss decrease toward the zero.

"""
