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
    hp['hidden_layers'] = [512, 256]
    # hp['batch_size'] = 64
    hp['gamma'] = 0.995
    hp['beta'] = 0.3
    hp['learn_rate'] = 0.01
    # hp['eps'] = 0.05
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
