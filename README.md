# Crafter RL Project – PPO & DQN Training

This project demonstrates **training and evaluating reinforcement learning (RL) agents** on the [Crafter](https://github.com/danijar/crafter) environment using [Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/).  

The main goal is learn how to:
- Train an RL agent on a **partially observable environment (POMDP)**  
- Incorporate **improvements** such as reward shaping and image preprocessing  
- Compare performance between **PPO** (Proximal Policy Optimization) and **DQN** agents  
- Test generalization on **unseen environment seeds**  

---

## Learning Objectives

1. Configure and run multiple RL algorithms (**PPO** and **DQN**) on the same environment  
2. Apply **reward shaping** to sparse reward tasks  
3. Preprocess visual inputs using **image processing techniques**  
4. Evaluate and compare agent performance on **seen vs. unseen seeds**  
5. Analyze training results and make iterative improvements to agent design  

---

##  Setup

### 1. Clone the repository
```bash
git clone https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git
cd Reinforcement-Learning-Project-2026-Crafter
```
### Create Environment
```bash
conda env create -f environment.yml
conda activate crafter_env

```
## Project Folder
```graphql
Crafter_Project/

├── Model_Building/  
│   ├── __init__.py                # Training scripts
│   ├── dqn_models.py
│   ├── ppo_models.py
│
├── Model_Testing/  
│   ├── __init__.py                # Evaluation and testing scripts
│   ├── dqn_testing.py
│   └── ppo_testing.py
│
├── Model_Helpers/                 # Environment wrappers and preprocessing
│   ├── __init__.py
│   ├── environments.py
│   ├── belief_reward_shaping.py 
│   └── image_processing.py        # Image normalization, resizing, and filtering
│
├── environment.yml                # Project dependencies
└── README.md

```
