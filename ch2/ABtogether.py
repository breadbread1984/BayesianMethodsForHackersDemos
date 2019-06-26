#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # generate fake observations from two different known bernoulli distributions
    occurrencesA = tfp.distributions.Bernoulli(probs = .05).sample(sample_shape = 1500, seed = 6.45);
    occurrencesA = tf.cast(occurrencesA, dtype = tf.float32);
    occurrencesB = tfp.distributions.Bernoulli(probs = .04).sample(sample_shape = 750, seed = 6.45);
    occurrencesB = tf.cast(occurrencesB, dtype = tf.float32);
    
    # simulate drawing from the joint (probA, probB) distribution with MCMC
    step_size = tf.Variable(0.5, dtype = tf.float32, trainable = False);
    # sample parameters according to log joint prob function
    [probsA, probsB], kernel_results = tfp.mcmc.sample_chain(
        num_results = 37200,
        num_burnin_steps = 1000,
        current_state = [tf.math.reduce_mean(occurrencesA), tf.math.reduce_mean(occurrencesB)],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(occurrencesA, occurrencesB),
                num_leapfrog_steps = 3,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 800),
                state_gradients_are_stopped = True
            ),
            bijector = [tfp.bijectors.Identity(), tfp.bijectors.Identity()]
        )
    );

    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));

    # calculate diff between probs A and probs B
    deltas = (probsA - probsB)[1000:];
    
    plt.figure(figsize(12.5, 12.5))

    #histogram of posteriors

    ax = plt.subplot(311);

    plt.xlim(0, .1);
    plt.hist(probsA, histtype = 'stepfilled', bins = 25, alpha = 0.85, label = "posterior of $p_A$", color = '#F15854', normed = True);
    plt.vlines(.05, 0, 80, linestyle = "--", label = "true $p_A$ (unknown)");
    plt.legend(loc = "upper right");
    plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns");

    ax = plt.subplot(312);

    plt.xlim(0, .1);
    plt.hist(probsB, histtype = 'stepfilled', bins = 25, alpha = 0.85, label = "posterior of $p_B$", color = '#60BD68', normed = True);
    plt.vlines(.04, 0, 80, linestyle = "--", label = "true $p_B$ (unknown)");
    plt.legend(loc = "upper right");

    ax = plt.subplot(313);
    plt.hist(deltas, histtype = 'stepfilled', bins = 30, alpha = 0.85, label = "posterior of delta", color = '#B276B2', normed = True);
    plt.vlines(.05 - .04, 0, 60, linestyle = "--", label = "true delta (unknown)");
    plt.vlines(0, 0, 60, color = "black", alpha = 0.2);
    plt.legend(loc = "upper right");
    
    better = tf.math.reduce_mean(tf.where(tf.greater(deltas,0), tf.ones_like(deltas, dtype = tf.float32), tf.zeros_like(deltas, dtype = tf.float32)));
    worse = tf.math.reduce_mean(tf.where(tf.less(deltas,0), tf.ones_like(deltas, dtype = tf.float32), tf.zeros_like(deltas, dtype = tf.float32)));
    print('Probability site A is WORSE than site B: %.3f' % worse);
    print('Probability site A is BETTER than site B: %.3f' % better);
    
    plt.show();

def log_prob_generator(occurrencesA, occurrencesB):
    # return log joint prob: log P(occurrencesA, occurrencesB, probA, probB)
    def func(probA, probB):
        # p(probA)
        probA_dist = tfp.distributions.Uniform(low = 0., high = 1.);
        # p(probB)
        probB_dist = tfp.distributions.Uniform(low = 0., high = 1.);
        # p(occurrenceA | probA)
        occurrenceA_dist = tfp.distributions.Bernoulli(probs = probA);
        # p(occurrenceB | probB)
        occurrenceB_dist = tfp.distributions.Bernoulli(probs = probB);
        # log p(occurrencesA, occurrencesB, probA, probB)
        #  = sum csub {occurrenceA_i} log p(occurrenceA_i | probA) + log p(probA)
        #  + sum csub {occurrenceB_i} log p(occurrenceB_i | probB) + log p(probB)
        return probA_dist.log_prob(probA) + probB_dist.log_prob(probB) \
               + tf.math.reduce_sum(occurrenceA_dist.log_prob(occurrencesA)) \
               + tf.math.reduce_sum(occurrenceB_dist.log_prob(occurrencesB));
    return func;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
