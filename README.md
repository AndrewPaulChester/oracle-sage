# Oracle-SAGE
This is the code for the ECML submission of "Oracle-SAGE: planning ahead in graph-based deep reinforcement learning" (ID 137).

## Installation
* Create a new conda virtual environment from oracle-sage.yml (conda env create -f sage.yml)
* Activate the new environment and then install this package in dev mode (conda activate sage; pip install -e .)
* For the minihack domains, Minihack must be installed separately according to the instructions [here](https://github.com/facebookresearch/minihack)

## Credits
This code builds heavily on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
A forked versions of this repository used in this work is contained in the forks/ folder. Installation instructions are left in that folders for completeness, but are not required for this project. Please simply follow the steps outlined in the installation section above.
GNN architecture and PyTorch geometric code based on [SR-DRL](https://github.com/jaromiru/sr-drl).

## Training
The `train` script contains commands for reproducing the results of the paper. For commands with lists of hyperparameters, make multiple separate calls with one item per list.