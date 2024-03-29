#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():
    
    # generate ground truth data (whether 100 students cheated or not)
    cheat_prob = tfp.distributions.Uniform(low = 0., high = 1.).sample();
    print('actually cheated probability is %f' % cheat_prob);
    cheated = tfp.distributions.Bernoulli(probs = cheat_prob).sample(sample_shape = 100, seed = 5);
    first_toss = tfp.distributions.Bernoulli(probs = 0.5).sample(100);
    second_toss = tfp.distributions.Bernoulli(probs = 0.5).sample(100);
    alleged_cheat_num = tf.cast(tf.math.reduce_sum(first_toss * cheated + (1 - first_toss) * second_toss), dtype = tf.float32);
    
    step_size = tf.Variable(0.5, dtype = tf.float32, trainable = False);
    # inference the cheat prob from the data
    [actual_cheat_prob], kernel_results = tfp.mcmc.sample_chain(
        num_results = 25000,
        num_burnin_steps = 2500,
        current_state = [tf.constant(0.2)],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(alleged_cheat_num),
                num_leapfrog_steps = 2,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 2000),
                state_gradients_are_stopped = True
            ),
            bijector = [tfp.bijectors.Sigmoid()]
        )
    );

    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));

    # get burned samples
    actual_cheat_prob = actual_cheat_prob[15000:];

    plt.figure(figsize(12.5, 6));
    plt.hist(actual_cheat_prob, histtype = "stepfilled", density = True, alpha = 0.85, bins = 30, label = "posterior distribution", color = '#5DA5DA');
    plt.xlim(0, 1);
    plt.legend();
    plt.show();

def log_prob_generator(alleged_cheat_num):
    # return log joint prob: log P(answers, cheated, first_toss, second_toss, cheat prob, cheat_num)
    def func(cheat_prob):
        # p(cheat prob)
        cheat_prob_dist = tfp.distributions.Uniform(low = 0., high = 1.);
        # alleged cheat prob = 1/2(head) * cheat_prob + 1/2(tail) * 1/2(head)
        alleged_cheat_prob = cheat_prob / 2 + 0.25;
        alleged_cheat_num_dist = tfp.distributions.Binomial(total_count = 100. ,probs = alleged_cheat_prob);
        return cheat_prob_dist.log_prob(cheat_prob) + alleged_cheat_num_dist.log_prob(alleged_cheat_num);
    return func;
    
if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
