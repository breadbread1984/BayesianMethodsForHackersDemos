#!/usr/bin/python3

import wget;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # the data is assumed to have been sampled from a 2-model gaussian mixtured model (GMM) distribution
    # inference the parameters of the gaussian mixtured model (GMM).
    # download observation data and convert to tensor
    url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter3_MCMC/data/mixture_data.csv';
    filename = wget.download(url);
    data = np.loadtxt(filename, delimiter = ',');
    data = tf.constant(data);
    
    step_size = tf.Variable(0.5, dtype = tf.float32, trainable = False);
    # inference the posteriori according to the observation with MCMC
    [model1_probs, mus, sigmas], kernel_results = tfp.mcmc.sample_chain(
        num_results = 25000,
        num_burnin_steps = 1000,
        current_state = [tf.constant(0.5), tf.constant(120.,190.), tf.constant(10., 10.)],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(data),
                num_leapfrog_steps = 2,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(),
                state_gradients_are_stopped = True
            ),
            bijector = [tfp.bijectors.Identity(), tfp.bijectors.Identity(), tfp.bijectors.Identity()]
        )
    );
            
    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));
    print('final step size: %f' % tf.math.reduce_mean(kernel_results.inner_results.extra.step_size_assign[-100:]));

def log_prob_generator(data):
    def func(model1_prob, mus, sigmas):
        # mixture probs of the two models
        mixture_prob_dist = tfp.distributions.Uniform(low = 0., high = 1.);
        model2_prob = 1 - model1_prob;
        # distributions of mu and sigma of the two gaussians
        mus_dist = tfp.distributions.Normal(loc = [120., 190.], scale = [10., 10.]);
        sigmas_dist = tfp.distributions.Uniform(low = [0., 0.], high = [100., 100.]);
        # distribution of the GMM
        observation_dist = tfp.distributions.MixtureSameFamily(
            mixture_distribution = tfp.distributions.Categorical(probs = tf.stack([model1_prob, model2_prob])),
            components_distribution = tfp.distributions.Normal(loc = mus, scale = sigmas)
        );
        return mixture_prob_dist.log_prob(model1_prob) + \
               mixture_prob_dist.log_prob(model2_prob) + \
               tf.math.reduce_sum(mus_dist.log_prob(mus)) + \
               tf.math.reduce_sum(sigmas_dist.log_prob(sigmas)) + \
               tf.math.reduce_sum(observation_dist.log_prob(data));
    return func;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
