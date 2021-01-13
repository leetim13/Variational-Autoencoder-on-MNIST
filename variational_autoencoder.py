# Implements auto-encoding variational Bayes.

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import expit as sigmoid

from autograd import grad
from autograd.misc.optimizers import adam
from data import load_mnist, save_images

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(b, unnormalized_logprob):
    # returns log Ber(b | mu)
    # unnormalized_logprob is log(mu / (1 - mu)
    # b must be 0 or 1
    s = b * 2 - 1
    return -np.logaddexp(0., -s * unnormalized_logprob)

def relu(x):    return np.maximum(0, x)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)           # nonlinear transformation
    return outputs

def nn_predict_gaussian(params, inputs):
    # Returns means and diagonal variances
    return unpack_gaussian_params(neural_net_predict(params, inputs))

def generate_from_prior(gen_params, num_samples, noise_dim, rs):
    latents = rs.randn(num_samples, noise_dim)
    return sigmoid(neural_net_predict(gen_params, latents))

def p_images_given_latents(gen_params, images, latents):
    bernoulli_means = #TODO
    likelihoods = #TODO
    return np.sum(likelihoods, axis=-1)  # Sum across pixels.

def vae_lower_bound(gen_params, rec_params, data, rs):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means, q_log_stds = #TODO
    latents = #TODO
    log_q_z = #TODO
    log_prior_z = #TODO
    log_p_x_given_z = #TODO
    elbo_estimate = #TODO
    return elbo_estimate


if __name__ == '__main__':
    # Model hyper-parameters
    latent_dim = 10
    data_dim = 784  # How many pixels in each image (28x28).
    gen_layer_sizes = [latent_dim, 300, 200, data_dim] #TODO: change to assignment model
    rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2] #TODO: change to assignment model

    # Training parameters
    param_scale = 0.01
    batch_size = 200
    num_epochs = 15
    step_size = 0.001

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()
    def binarise(images):
        on = images > 0.5
        images = images * 0.0
        images[on] = 1.0
        return images

    print("Binarising training data...")
    train_images = binarise(train_images)
    test_images = binarise(test_images)

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes)
    combined_init_params = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    # this is called loss in writeup
    def objective(combined_params, iter):
        data_idx = batch_indices(iter)
        gen_params, rec_params = combined_params
        return #TODO variational objective

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Objective       |    Test ELBO  ")
    def print_perf(combined_params, iter, grad):
        if iter % 10 == 0:
            gen_params, rec_params = combined_params
            bound = np.mean(objective(combined_params, iter))
            message = "{:15}|{:20}|".format(iter//num_batches, bound)
            if iter % 100 == 0:
                test_bound = -vae_lower_bound(gen_params, rec_params, test_images, seed) / data_dim
                message += "{:20}".format(test_bound)
            print(message)

            fake_data = generate_from_prior(gen_params, 20, latent_dim, seed)
            save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)
