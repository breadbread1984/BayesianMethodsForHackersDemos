#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # generate occurrences from a known bernoulli distribution
    occurrences = tfp.distributions.Bernoulli(probs = .05).sample(sample_shape = 1500, seed = 10);
    occurrences = tf.cast(occurrences, dtype = tf.float32);
    
    # now make an educated guess of the true prob from the occurrences
    step_size = tf.Variable(0.5, dtype = tf.float32, trainable = False);
    # sample parameters according to log joint prob function
    [probs], kernel_results = tfp.mcmc.sample_chain(
        num_results = 48000,
        num_burnin_steps = 25000,
        current_state = [tf.math.reduce_mean(occurrences)],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(occurrences),
                num_leapfrog_steps = 2,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 20000),
                state_gradients_are_stopped = True
            ),
            bijector = [tfp.bijectors.Identity()]
        )
    );
    
    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));
    
    probs = probs[25000:];
    
    plt.figure(figsize(12.5, 4));
    plt.title('Posterior distribution of $p_A$, the true effectiveness of site A');
    plt.vline(.05, 0, 90, linestyle = '--', label = 'true $p_A$ (unknown)');
    plt.hist(probs, bins = 25, histtype = 'stepfilled', normed = True);
    plt.legend();
    plt.show();

def log_prob_generator(occurrences):
    # return log joint prob: log P(occurrences, prob)
    def func(prob):
        # p(prob)
        prob_dist = tfp.distributions.Uniform(low = 0., high=1.);
        # p(occurrence | prob)
        occurrence_dist = tfp.distributions.Bernoulli(probs = prob);
        # log p(occurrences, prob) = sum csub {occurrence_i} log p(occurrence_i | prob) + log p(prob)
        return prob_dist.log_prob(prob) + tf.math.reduce_sum(occurrence_dist.log_prob(occurrences));
    return func;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
