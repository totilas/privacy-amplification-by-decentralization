# Figure 1

## Theoretical bounds

to reproduce the figure (a), run `main_a.py`

Required packages: numpy and matplotlib

## Gaussian experimental loss

We reuse the implementation of shuffling mechanisms in the [paper](https://arxiv.org/abs/1903.02837):

> B. Balle, J. Bell, A. Gascon, and K. Nissim. *The Privacy Blanket of the Shuffle Model*, International Cryptology Conference (CRYPTO), 2019


The whole simulation is in the `expe.py` and it links the different parts:

- `amplification.py` comes from the previous implementation with the addition of subsampling mechanisms
- `mechanisms.py` comes also from the previous implementation, and it is not fully used, but convenient for possible extensions
- `randomWalk.py` generates a random walks on the complete graph
- `composition.py` implements advanced composition for identical and different epsilon

To reproduce the figure (b), run `main_b.py`

Required packages: numpy, scipy, matplotlib