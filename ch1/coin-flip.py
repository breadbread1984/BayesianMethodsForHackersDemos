#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # underlying distribution
    rv_coin_flip_prior = tfp.distributions.Bernoulli(probs = 0.5, dtype = tf.int32);
    # experiment times
    num_trials = tf.constant([0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000]);
    # sample 2000 times from the underlying distribution
    coin_flip_data = rv_coin_flip_prior.sample(num_trials[-1]);
    # tf.math.cumsum(coin_flip_data) = (0, c1, c1 + c2, c1 + c2 + c3, ..., c1 + ... + c2000)
    # cumulative_headcounts = (0,c1,c1+c2,...,c1+...+c2000)
    # cumulative_headcounts.shape = (2001)
    coin_flip_data = tf.pad(coin_flip_data, tf.convert_to_tensor([[1,0]]), mode = "CONSTANT");
    cumulative_headcounts = tf.gather(tf.math.cumsum(coin_flip_data), num_trials);
    # create 12 normalized likelihood functions with 12 alpha, beta pairs
    # alpha and beta plus one to avoid zeros
    likelihoods = tfp.distributions.Beta(
        concentration1 = tf.cast(1 + cumulative_headcounts, dtype = tf.float32),
        concentration0 = tf.cast(1 + num_trials - cumulative_headcounts, dtype = tf.float32)
    );
    # when ploting the likelihood function, these are ploted x
    x = tf.linspace(start = 0., stop = 1., num = 100, name = "linspace");
    # value of 12 likelihood functions at x points
    # y.shape = (12,100)
    y = tf.transpose(likelihoods.prob(x[:, tf.newaxis]));
    
    # plot the experiment result
    plt.figure(figsize(16,9));
    for i in range(tf.shape(num_trials)[0]):
        # for 12 accumulative counts over different experiment times
        sx = plt.subplot(num_trials.shape[0] / 2, 2, i+1);
        plt.xlabel("$p$, probability of heads") if i in [0, tf.shape(num_trials)[0] - 1] else None;
        plt.setp(sx.get_yticklabels(), visible = False);
        plt.plot(x, y[i,...], label = "observe %d tosses,\n %d heads" % (num_trials[i], cumulative_headcounts[i]));
        plt.fill_between(x, 0, y[i], color = '#5DA5DA', alpha = 0.4);
        plt.vlines(0.5, 0, 4, color = "k", linestyles = "--", lw = 1);
        leg = plt.legend();
        leg.get_frame().set_alpha(0.4);
        plt.autoscale(tight = True);
    plt.suptitle("Bayesian updating posterior probabilities", y = 1.02, fontsize = 14);
    plt.tight_layout();
    plt.show();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
