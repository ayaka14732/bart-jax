# JAX implementation of BART

This project is a JAX implementation of [BART](https://arxiv.org/abs/1910.13461). The aim of this project is to demonstrate how Transformer-based models can be implemented using JAX and trained on Google Cloud TPUs.

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

This project is inspired by [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer), while the code for this project is entirely written by myself.

## Environment Setup

This project requires at least Python 3.11 and JAX 0.3.24.

```sh
python3.11 -m venv ./venv
. ./venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.24" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
```

## Test

Execute the following script to run the tests:

```sh
python scripts/run_tests.py
```

If TPU is available, TPU-related tests can be included as well:

```sh
python scripts/run_tests.py --tpu
```
