# Implementation of `iterated Deep Q-Network (i-DQN)` in PyTorch

Paper 👉[📄](https://arxiv.org/pdf/2403.02107)

Official implementation in JAX 👉[💻](https://github.com/theovincent/i-DQN)

## User installation
We recommend using Python 3.11.5 In the folder where the code is, create a Python virtual environment, activate it, update pip and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```