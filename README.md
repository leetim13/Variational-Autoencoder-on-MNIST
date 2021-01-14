This project implements and investigates the Variational Autoencoder on binarized MNIST digits by building a generative model to infer the bottom half of the given binarized MNIST digits conditioned on the top half of these images.

![](/sample_images/image_conditioned.png)
![](/sample_images/image_binarized.png)

## Implementations include:
* `Project.toml` packages for the Julia environment.
* `variational_autoencoder.py` Python version.
* `loadMNIST.py` loading MNIST data in Python.
* `example_flux_model.jl` example flux model in Julia.
* `vae.jl` source code in Julia.
* `encoder_params.bson` final params/weights of trained model.
* `decoder_params.bson` final params/weights of trained model.
* **`Julia-Variational-Autoencoder-Final.ipynb`** the final jupyter notebook project.

Note: this project is part of of the assignment from Statistical Methods for Machine Learning II at the Univeristy of Toronto.
