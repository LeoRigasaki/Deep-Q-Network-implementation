# Deep Q-Network (DQN) Implementation for Atari Breakout

This repository implements a Deep Q-Network (DQN), a reinforcement learning algorithm, to train an agent to play Atari's Breakout game. The implementation includes advanced features like experience replay, target networks, and game monitoring with video exports.

## Features

- **Preprocessing**: Converts raw game frames to grayscale, resizes to 84x84, and applies cropping for efficient input.
- **Frame Buffer**: Maintains the last four frames to help the agent observe motion.
- **Neural Network**: Utilizes a Convolutional Neural Network (CNN) to approximate Q-values for actions.
- **Experience Replay**: Stores past experiences for training stability and efficiency.
- **Target Networks**: Stabilizes Q-learning by periodically updating reference weights.
- **Game Monitoring**: Records gameplay videos to evaluate the agent's performance.

## Installation

### Prerequisites
- Python 3.6+
- TensorFlow/Keras
- OpenAI Gym
- Additional libraries: `opencv-python`, `unrar`, `numpy`, `matplotlib`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dqn-atari-breakout.git
   cd dqn-atari-breakout
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and import Atari ROMs:
   ```bash
   python -m atari_py.import_roms ./rars
   ```

## Usage

1. Launch the training notebook:
   ```bash
   jupyter notebook DQN_Atari_Breakout.ipynb
   ```

2. Train the agent:
   - Configure hyperparameters (e.g., learning rate, batch size, epsilon decay).
   - Run the training loop.

3. Evaluate and monitor the agent:
   - Save the trained weights.
   - Record gameplay videos for evaluation.

## Project Structure

- `DQN_Atari_Breakout.ipynb`: Main notebook implementing the DQN training and evaluation.
- `videos/`: Directory for storing gameplay videos.
- `requirements.txt`: List of Python dependencies.

## Demo Video

Watch the trained agent play Breakout:

<video controls>
  <source src="videos/trained-agent-game.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## How It Works

### Algorithm Overview
- **DQN**: Combines Q-learning with deep neural networks to handle high-dimensional inputs like game frames.
- **Experience Replay**: Improves learning efficiency by breaking correlations between consecutive observations.
- **Target Networks**: Addresses instability in Q-value updates by using a separate network for reference.

### Training Details
- The agent learns to maximize rewards by exploring the game environment and updating Q-values.
- Hyperparameters like epsilon control the balance between exploration and exploitation.

### Evaluation
- Performance is monitored via reward plots and TD loss.
- Videos showcase the agent's progression over time.

## Results
- The agent progressively learns to play Breakout, achieving higher rewards as training proceeds.
- Training may take several hours depending on hardware and hyperparameter configurations.

## Acknowledgments
- OpenAI Gym for providing the Atari game environment.
- DQN algorithm inspired by the original DeepMind paper and OpenAI Baselines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
