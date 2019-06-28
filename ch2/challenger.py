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

    plt.figure(figsize(12.5, 6))

    #histogram of the distributions of parameter alpha and beta:
    plt.subplot(211);
    plt.title(r"Posterior distributions of the variables $\alpha, \beta$");
    plt.hist(beta, histtype = 'stepfilled', bins = 35, alpha = 0.85, label = r"posterior of $\beta$", color = '#B276B2', density = True)
    plt.legend();

    plt.subplot(212);
    plt.hist(alpha, histtype = 'stepfilled', bins = 35, alpha = 0.85, label = r"posterior of $\alpha$", color = '#F15854', density = True)
    plt.legend();
    
    plt.show();
    
    # plot regressed function with the origin al data
    alpha_mean = tf.math.reduce_mean(alpha);
    beta_mean = tf.math.reduce_mean(beta);
    
    def logistic(x, alpha, beta):
        return 1.0 / (1.0 + tf.math.exp(beta * x + alpha));
    
    temps = tf.linspace(tf.math.reduce_min(temperatures) - 5, tf.math.reduce_max(temperatures) + 5 , 2500);
    probs = logistic(temps, alpha_mean, beta_mean);
    
    plt.figure(figsize(12.5, 4))

    plt.plot(temps, probs, lw = 3, label = "average posterior \nprobability of defect");
    plt.plot(temps, probs, ls = "--", label = "realization from posterior");
    plt.plot(temps, probs[-8], ls = "--", label = "realization from posterior");
    plt.scatter(temperatures, failures, color = "k", s = 50, alpha = 0.5);
    plt.title("Posterior expected value of probability of defect; plus realizations");
    plt.legend(loc = "lower left");
    plt.ylim(-0.1, 1.1);
    plt.xlim(temps[0], temps[-1]);
    plt.ylabel("probability");
    plt.xlabel("temperature");
    
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

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
