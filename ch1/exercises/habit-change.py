#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # check out whether there is an abrupt habit change of message numbers
    # give the day when the change occurs.
    count_data = tf.constant([
        13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,  
        11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,  
        19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,  
        15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,  
        12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37, 
        5,  14,  13,  22,], dtype = tf.float32);
    
    step_size = tf.Variable(0.05, dtype = tf.float32, trainable = False);
    # sample parameters according to log joint prob function
    [lambda1_samples, lambda2_samples, tau_samples], kernel_results = tfp.mcmc.sample_chain(
        num_results = 100000,
        num_burnin_steps = 10000,
        current_state = [tf.math.reduce_mean(count_data), tf.math.reduce_mean(count_data), 0.5],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(count_data),
                num_leapfrog_steps = 2,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(100),
                state_gradients_are_stopped = True
            ),
            bijector = [tfp.bijectors.Exp(), tfp.bijectors.Exp(), tfp.bijectors.Sigmoid()]
        )
    );
    # convert from float to day count
    tau_samples = tf.math.floor(tau_samples * count_data.shape[0]);

    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));
    print('final step size: %f' % tf.math.reduce_mean(kernel_results.inner_results.extra.step_size_assign[-100:]));

    # print mean of lambda_1 and lambda_2
    print('mean of lambda_1 is %f' % tf.math.reduce_mean(lambda1_samples));
    print('mean of lambda_2 is %f' % tf.math.reduce_mean(lambda2_samples));
    print('expected percentage increase is %f' % tf.math.reduce_mean(lambda1_samples / lambda2_samples));
    count = tf.math.reduce_sum(tf.cast(tf.less(tau_samples, 45.), dtype = tf.float32));
    lambda1_sum = tf.math.reduce_sum(tf.where(tf.less(tau_samples, 45.), lambda1_samples, tf.zeros_like(lambda1_samples)));
    print('mean of lambda_1 when tau is less than 45 is %f' % (lambda1_sum / count));

def log_prob_generator(count_data):
    # return log join prob: log P(data, lambda1, lambda2, tau)
    def func(lambda1, lambda2, tau):
        # parameter of P(lambda1) and P(lambda2)
        alpha = 1. / tf.math.reduce_mean(count_data, axis = 0);
        # P(lambda1)
        lambda1_dist = tfp.distributions.Exponential(rate = alpha);
        # p(lambda2)
        lambda2_dist = tfp.distributions.Exponential(rate = alpha);
        # p(tau)
        tau_dist = tfp.distributions.Uniform();
        # parameters of p(data | lambda1, lambda2, tau)
        lambdas = tf.gather([lambda1, lambda2], indices = tf.cast(tau * count_data.shape[0] <= tf.range(count_data.shape[0], dtype = tf.float32), dtype = tf.int32));
        # p(data | lambda1, lambda2, tau)
        obs_dists = tfp.distributions.Poisson(rate = lambdas);
        # log p(data, lambda1, lambda2, tau) = log [p(data | lambda1, lambda2, tau) * p(lambda1) * p(lambda2) * p(tau)]
        #                                    = log p(data | lambda1, lambda2, tau) + log p(lambda1) + log p(lambda2) + log p(tau)
        return lambda1_dist.log_prob(lambda1) + lambda2_dist.log_prob(lambda2) + tau_dist.log_prob(tau) + tf.math.reduce_sum(obs_dists.log_prob(count_data), axis = 0);
    return func;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
