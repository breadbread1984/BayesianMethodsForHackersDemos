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
    # inference the posteriori according to the observation with MCMC
    

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
