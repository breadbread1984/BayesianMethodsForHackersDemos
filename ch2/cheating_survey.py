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
    
    # inference the cheat prob from the data
    [actual_cheat_prob], kernel_results = tfp.mcmc.sample_chain(
        num_results = 40000,
        num_burnin_steps = 15000,
        current_state = [tf.constant(0.4)],
        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn = log_prob_generator(alleged_cheat_num),
            seed = 54
        ),
        parallel_iterations = 1
    );

    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.is_accepted, dtype = tf.float32)));

    # get burned samples
    actual_cheat_prob = actual_cheat_prob[15000:];
    
    plt.figure(figsize(12.5, 6));
    plt.hist(actual_cheat_prob, histtype = "stepfilled", density = True, alpha = 0.85, bins = 30, label = "posterior distribution", color = '#5DA5DA');
    plt.vlines([.1, .40], [0, 0], [5, 5], alpha = 0.3);
    plt.xlim(0, 1);
    plt.legend();
    plt.show();
    
def log_prob_generator(alleged_cheat_num):
    # return log joint prob: log P(answers, cheated, first_toss, second_toss, cheat prob, cheat_num)
    def func(cheat_prob):
        # p(cheat prob)
        cheat_prob_dist = tfp.distributions.Uniform(low = 0., high = 1.);
        # first toss ~ p(first toss)
        first_toss = tfp.distributions.Bernoulli(probs = 0.5).sample(100);
        # second toss ~ p(second toss)
        second_toss = tfp.distributions.Bernoulli(probs = 0.5).sample(100);
        # cheated ~ p(cheated)
        actually_cheated = tfp.distributions.Bernoulli(probs = cheat_prob).sample(100);
        # cheat_prob = f(first toss, second toss, cheated)
        alleged_cheat_prob = tf.cast(tf.math.reduce_sum(first_toss * actually_cheated + (1 - first_toss) * second_toss), dtype = tf.float32) / 100.;
        # use monte carlo integral to get
        # p(alleged cheated num) =
        # int csub {first toss, second toss, actually cheated} { 
        #    p(alleged cheated num | first toss, second toss, actually cheated) *
        #    p(first toss) p(second toss) p(actually cheated)
        # } d first toss, second toss, actually cheated
        alleged_cheat_num_dist = tfp.distributions.Binomial(total_count = 100. ,probs = alleged_cheat_prob);
        return cheat_prob_dist.log_prob(cheat_prob) + alleged_cheat_num_dist.log_prob(alleged_cheat_num);
    return func;
    
if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
