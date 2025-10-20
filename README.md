# Crafter RL Project – SAC & DQN Training

This project demonstrates **training and evaluating reinforcement learning (RL) agents** on the [Crafter](https://github.com/danijar/crafter) environment using [Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/).  

The main goal is to help students and researchers learn how to:
- Train an RL agent on a **partially observable environment (POMDP)**  
- Incorporate **improvements** such as reward shaping and temporal context (frame stacking)  
- Compare performance between **SAC** (Soft Actor-Critic) and **DQN** agents  
- Test generalization on **unseen environment seeds**  

---

## 🧠 Learning Objectives

By completing this project, students will be able to:
1. Configure and run multiple RL algorithms (SAC and DQN) on the same environment  
2. Apply **reward shaping** to sparse reward tasks  
3. Implement **short-term memory** using **frame stacking**  
4. Evaluate and compare agent performance on **seen vs. unseen seeds**  
5. Analyze training results and make iterative improvements to agent design  

---

## 🛠 Setup

### 1. Clone the repository
```bash
git clone https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git
cd Reinforcement-Learning-Project-2026-Crafter
```
### 2.Create Python environment
```bash
conda env create -f environment.yml
conda activate crafter_env
```
### 3.Project Folder Structure
```graphql
Crafter_Project/
│
├── Model_Building/  
    |──__init__.py               # Training scripts
│   ├── dqn_models.py
│   ├── sac_models.py
│
├── Model_Testing/  
    ├──__init__.py           # Evaluation and testing scripts
│   ├── dqn_testing.py
│   └── sac_testing.py
│
├── Model_Helpers/              # Environment wrappers for Crafter
│   ├── __init__.py
│   ├── environment.py         # Base + reward-shaped environments
│   └── belief_reward_shaping.py 
│     

├── environment.yml          # Project dependencies
└── README.md     
```
