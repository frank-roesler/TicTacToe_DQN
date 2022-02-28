# TicTacToe_DQN
This repository contains a PyTorch implementation of a reinforcement learning algorithm to learn to play the classical TicTacToe game. The algorithm is based on the "Deep Q-Network" (DQN) architecture and is based on code from the textbook [Zai, A., & Brown, B. (2020). Deep reinforcement learning in action. Manning Publications].  
The repo includes the environment (the TicTacToe game) as well as the DQN training loop and a test script for evaluation. To make the test script run out-of-the-box, a trained state dict is also included. For simplicity, the agent is trained against an opponent that takes random moves only.
To stabilize training, the bare DQN training loop is supplemented with Experience Replay and a Target Network (see [Lin, L. J. (1992)]).

To train the model, download the contents of the repo and run the `train.py` script. On a modern laptop (without GPU) training should take a couple of minutes. The `test.py` script allows the user to play against the trained network. Note that as it is only trained against a random opponent, it cannot be expected to reach human level proficiency.

Any comments or queries are welcome at https://frank-roesler.github.io/contact/
