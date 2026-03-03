#  Deep-Q-Learning-Atari

A PyTorch implementation of **Deep Q-Network (DQN)** for Atari **MsPacman** using Gymnasium and the Arcade Learning Environment (ALE).

This project implements experience replay, target network synchronization, frame stacking, and epsilon-greedy exploration.

---

## 🚀 Features

- ✅ Deep Q-Network (DQN)
- ✅ Experience Replay Buffer
- ✅ Target Network Updates
- ✅ Frame Preprocessing & Stacking
- ✅ Epsilon-Greedy Exploration
- ✅ Gymnasium + ALE integration
--

## 🧠 Architecture

The agent consists of:

- **Local Network** (online Q-network)
- **Target Network** (periodically synchronized)
- **Replay Memory** (deque-based buffer)
- **Adam Optimizer**
- **MSE Loss**

## ⚙️ Installation

### 1️⃣ Clone Repository

bash
- git clone https://github.com/N1S55M-X/Deep-Q-Learning-Atari.git

- cd Deep-Q-Learning-Atari
  
### 2️⃣ Install Dependencies
- pip install -r requirements.txt
  
### 3️⃣ Install Atari ROMs
- AutoROM --accept-rom-license

### ▶️ Run Training
- python atari-dqn/Train.py
---

## 📚 Inspiration & Learning Resources

This implementation was inspired by reinforcement learning concepts presented in:

**Artificial Intelligence A-Z 2026: Agentic AI, Gen AI, and RL**  
by Hadelin de Ponteves, Kirill Eremenko, and the SuperDataScience Team.

All code in this repository has been written for educational and research purposes.



