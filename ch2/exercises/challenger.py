#!/usr/bin/python3

import wget;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # get challenger data
    url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv';
    filename = wget.download(url);
    # parse file and convert the data to tensor
    challenger_data_ = np.genfromtxt(filename, skip_header=1, usecols=[1, 2], missing_values="NA", delimiter=",");
    challenger_data_ = challenger_data_[~np.isnan(challenger_data_[:, 1])];
    # challenger_data.shape = (item num, 2)
    # every item is (temperature, failure or not)
    challenger_data = tf.constant(challenger_data_, dtype = tf.float32);
    
    temperatures = challenger_data[:,0];
    failures = challenger_data[:,1];

    # regression with beyasian method
    step_size = tf.Variable(0.01, dtype = tf.float32, trainable = False);
    [alpha, beta], kernel_results = tfp.mcmc.sample_chain(
        num_results = 10000,
        num_burnin_steps = 2000,
        current_state = [tf.constant(0.), tf.constant(0.)],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(temperatures, failures),
                num_leapfrog_steps = 40,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 1600),
                state_gradients_are_stopped = True
            ),
            bijector = [tfp.bijectors.AffineScalar(100.), tfp.bijectors.Identity()]
        )
    );

    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));

    #type your code here.
    plt.figure(figsize(12.5, 4));

    plt.scatter(alpha, beta, alpha=0.1);
    plt.title("Why does the plot look like this?");
    plt.xlabel(r"$\alpha$");
    plt.ylabel(r"$\beta$");
    plt.show();
    
def log_prob_generator(temperatures,failures):
    
    def func(alpha, beta):
        beta_dist = tfp.distributions.Normal(loc = 0., scale = 1000.);
        alpha_dist = tfp.distributions.Normal(loc = 0., scale = 1000.);
        probs = 1. / (1. + tf.math.exp(beta * temperatures + alpha));
        failure_dists = tfp.distributions.Bernoulli(probs = probs);
        return beta_dist.log_prob(beta) + alpha_dist.log_prob(alpha) \
            + tf.math.reduce_sum(failure_dists.log_prob(failures));
    return func;

def separation_plot(p, y):
    """
    This function creates a separation plot for logistic and probit classification. 
    See http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf
    
    p: The proportions/probabilities, can be a nxM matrix which represents M models.
    y: the 0-1 response variables.
    
    """    
    assert p.shape[0] == y.shape[0], "p.shape[0] != y.shape[0]";

    colors_bmh = ["#eeeeee", "#348ABD"];

    fig = plt.figure();
    ax = fig.add_subplot(1, 1, 1);
    ix = tf.argsort(p);
    #plot the different bars
    bars = ax.bar(tf.range(p.shape[0]), tf.ones_like(p), width = 1., color = colors_bmh[tf.gather(y,ix)], edgecolor = 'none');
    ax.plot(tf.range(p.shape[0] + 1), tf.gather(p,tf.concat([ix,[ix[-1]]], axis = -1)), "k", linewidth = 1., drawstyle = "steps-post");
    #create expected value bar.
    ax.vlines([tf.math.reduce_sum(1 - tf.gather(p,ix))], [0], [1]);
    plt.xlim(0, p.shape[0]);

    plt.tight_layout();
    plt.show();
    
    return

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
