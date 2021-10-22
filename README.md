# Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks

This is the implementation of the paper [Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks]().

This repository provide an environment of distributed/decentralised active voltage control on power distribution networks and a batch of state-of-the-art multi-agent actor-critic algorithms that can be used for training.

The framework of environment implementation follows the multi-agent environment framework provided in [PyMARL](https://github.com/oxwhirl/pymarl/). Therefore, all baselines that are compatible with that framework can be easily applied to this environment.

<br />

## Brief Introduction of the Task

In this section, we give a brief introduction of this task so that the users can easily understand the objective of this environment. 

**Objective:** Each agent controls a PV inverter that generates the reactive power so that the voltage of each bus is varied and within the safety range defined as $0.95 \ p.u. \leq v_{k} \leq 1.05 \ p.u., \ \forall k \in V$, where $V$ is the set of buses of the whole system and $p.u.$ is a unit to measure voltage. Since each agent's decision could influence each other due to property of power networks and not all buses is installed a PV, agents should cooperate to control the voltage of all buses in a power network. Also, each agent can only observe the partial information as the observation. This problem is natually a Dec-POMDP.

**Action:** The reactive power is constrained by the capacity of the equipment, and the capacity is related to the active power of PV. As a result, the range of reactive power is dynamically varied. Mathematically, the reactive power of each PV inverter is represented as $$q_{k}^{\scriptscriptstyle PV} = a_{k} \ \sqrt{(s_{k}^{\scriptscriptstyle \max})^{2} - (p_{k}^{\scriptscriptstyle PV})^{2}},$$ where $s_{k}^{\scriptscriptstyle \max}$ is the maximum apparent power of the $k\text{th}$ node that is dependent on the physical capacity of the PV inverter; $p_{k}^{\scriptscriptstyle PV}$ is the instantaneous PV active power. The action we control is the variable $0 \leq a_{k} \leq 1$, indicating the percentage of the intantaneous capacity of reactive power. For this reason, the action is continuous in this task.

**Observation:** Each agent can observe the information of the zone where it belongs. For example, in Figure 1 the agent on bus 25 can observe the information in zone 3. Each agent's observation consists of the following variables within the zone:

* Load Active Power,
* Load Reactive Power,
* PV Active Power,
* PV Reactive Power,
* Voltage.


<figure>
  <br />
  <img src="img/bus33.png" height="240" weight="720">
  <figcaption>
    Figure 1: Illustration on 33-bus system. Each bus is indexed by a circle with a number. Four control regions are partitioned by the smallest path from the terminal to the main branch (bus 1-6). We control the voltage on bus 2-33 whereas bus 0-1 represent the substation with constant voltage and infinite active and reactive power capacity. G represents an external generator; small Ls represent loads; and emoji of sun represents the location where a PV is installed.
  </figcaption>
  <br /> <br />
</figure>

**Reward:** The reward function is shown as follows:
$$\mathit{r} = - \frac{1}{|V|} \sum_{i \in V} l_{v}(v_{i}) - \alpha \cdot l_{q}(\mathbf{q}^{\scriptscriptstyle PV}),$$
where $l_{v}(\cdot)$ is voltage loss that measure whether the voltage of a bus is within the safety range; $l_{q}(\mathbf{q}^{\scriptscriptstyle PV})=\frac{1}{|\mathcal{I}|}||\mathbf{q}^{\scriptscriptstyle PV}||\_{1}$ that can be seen as a simple approximation of power loss, where $\mathbf{q}^{\scriptscriptstyle PV}$ is a vector of agents' reactive power, $\mathcal{I}$ is a set of agents and $\alpha$ is a multiplier to adjust the balance between voltage control and the generation of reactive power. In this work, we investigate different forms of $l_{v}(\cdot)$. Literally, the aim of this reward function is controlling the voltage, meanwhile minimising the power loss that is correlated with the economic loss.

<br />

## Installation of the Dependencies

1. Install [Anaconda](https://www.anaconda.com/products/individual#Downloads).
2. After cloning or downloading this repository, assure that the current directory is `[your own parent path]/distributed-active-voltage-conrtol`.
3. Execute the following command. 
   ```{bash}
   conda env export > environment.yml
   ```
4. Activate the installed virtual environment using the following command.
    ```{bash}
    conda activate distributed_powernet
    ```

<br />

## Downloading the Dataset

1. Download the data from the link: https://drive.google.com/file/d/1Z23Z5mZhDcK6bO5tJ-D_WGxM6WpTJuYr/view?usp=sharing.
2. Unzip the zip file and you can see the following 3 folders:

    * `bus33_3min_final`
    * `bus141_3min_final`
    * `bus322_3min_final`
3. Go to the directory `[Your own parent path]/distributed-active-voltage-control/environments/var_voltage_control/` and create a folder called `data`.
4. Move the 3 extracted folders by step 2 to the directory `[Your own parent path]/distributed-active-voltage-control/environments/var_voltage_control/data/`.

<br />

## Two modes of Tasks

### Background

There are 2 modes of tasks included in this environment, i.e. distributed active voltage control and decentralised active voltage control. Distributed active voltage control is the task introduced in the paper [Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks](), whereas Decentralised active voltage control is the task that most of the prior works considered. The primary difference between these 2 modes of tasks are that in decentralised active voltage control the equipments in each zone are controlled by an agent, while in distributed active voltage control each equipment is controlled by an agent (see Figure 1).

### How to use?

If you would attempt distributed active voltage control, you can set the argument for `train.py` and `test.py` as follows.

```bash
python train.py --mode distributed
```

```bash
python test.py --mode distributed
```

If you would attempt decentralised active voltage control, you can set the argument for `train.py` and `test.py` as follows.

```bash
python train.py --mode decentralised
```

```bash
python test.py --mode decentralised
```

<br />

## Quick Start

### Training Your Model

You can execute the following command to train a model on a power system using the following command.

```
python train.py --alg matd3 --alias 0 --mode distributed --scenario bus33bw_gu_3min --voltage-loss-type l1 --save-path trial
```

The the meanings of the arguments are illustrated as follows:
* `--alg` indicates the MARL algorithm you would like to use.
* `--alias` is the alias to distinguish different experiments.
* `--mode` is the mode of the envrionment. It contains 2 modes, e.g. distributed and decentralised. Distributed mode is the one introduced in this work, whereas decentralised mode is the traditional environment used by the prior works.
* `--scenario` indicates the power system on which you would like to train.
* `--voltage-loss-type` indicates the voltage loss you would like to use for training.
* `--save-path` is the path you would like to save the model, tensorboard and configures.

### Testing Your Model

After training, you can exclusively test your model to do the further analysis using the following command.

```
python test.py --save-path trial/model_save/bus33bw --alg matd3 --alias 0 --scenario bus33bw_gu_3min --voltage-loss-type l1 --test-mode single --test-day 730 --render
```

The the meanings of the arguments are illustrated as follows:
* `--alg` indicates the MARL algorithm you used.
* `--alias` is the alias you used to distinguish different experiments.
* `--mode` is the mode of the envrionment you used to train your model.
* `--scenario` indicates the power system on which you trained your model.
* `--voltage-loss-type` indicates the voltage loss you used for training.
* `--save-path` is the path you saved your model. You just need to give the parent path including the directory `model_save`.
* `--test-mode` is the test mode you would like to use. There are 2 modes you can use, i.e. `single` and `batch`. 
* `--test-day` is the day that you would like to do the test. Note that it is only activated if the `--test-mode` is `batch`.
* `--render` indicates activating the rendering of the environment.

<br />

## Interaction with Environment

The simple use of the environment is shown as the following codes.

```python
state, global_state = env.reset()

for t in range(240):
    actions = agents.get_actions() # a vector involving all agents' actions
    reward, done, info = env.step(actions, add_noise=False)
```

<br />

## API Usage

For more details for this environment, users can check the upcoming [API Docs]().

<br />

## Citation

