#!/usr/bin/python3

import wget;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # the data is assumed to have been sampled from a 2-model gaussian mixtured model (GMM) distribution
    # inference the parameters of the gaussian mixtured model (GMM).
    # download observation data and convert to tensor
    url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter3_MCMC/data/mixture_data.csv';
    filename = wget.download(url);
    data = np.loadtxt(filename, delimiter = ',');
    data = tf.constant(data, dtype = tf.float32);
    
    step_size = tf.Variable(0.5, dtype = tf.float32, trainable = False);
    bijectors = [tfp.bijectors.Identity(), tfp.bijectors.Identity(), tfp.bijectors.Identity()];
    # inference the posteriori according to the observation with MCMC
    [model1_probs, mus, sigmas], kernel_results = tfp.mcmc.sample_chain(
        num_results = 25000,
        num_burnin_steps = 1000,
        current_state = [tf.constant(0.5), tf.constant([120.,190.]), tf.constant([10., 10.])],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(data),
                num_leapfrog_steps = 2,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 1000),
                state_gradients_are_stopped = True
            ),
            bijector = bijectors
        )
    );

    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));
    print('final step size: %f' % tf.math.reduce_mean(kernel_results.inner_results.extra.step_size_assign[-100:]));

    # plot the samples
    # for pretty colors later in the book.
    colors = ['#5DA5DA', '#F15854'] if mus[-1, 0] > mus[-1, 1] else ['#F15854', '#5DA5DA'];
    plt.figure(figsize(12.5, 9));
    
    # plot means of two models.
    plt.subplot(311);
    plt.plot(mus[:, 0], label = "trace of center 0", c = colors[0], lw = 1);
    plt.plot(mus[:, 1], label = "trace of center 1", c = colors[1], lw = 1);
    plt.title("Traces of unknown parameters");
    leg = plt.legend(loc="upper right");
    leg.get_frame().set_alpha(0.7);

    # plot sigmas of two models.
    plt.subplot(312);
    plt.plot(sigmas[:, 0], label = "trace of standard deviation of cluster 0", c = colors[0], lw = 1);
    plt.plot(sigmas[:, 1], label = "trace of standard deviation of cluster 1", c = colors[1], lw = 1);
    plt.legend(loc = "upper left");

    # plot mixture probability of GMM.
    plt.subplot(313);
    plt.plot(model1_probs, label = "$p$: frequency of assignment to cluster 0", c = '#60BD68', lw = 1);
    plt.xlabel("Steps");
    plt.ylim(0, 1);
    plt.legend();
    
    plt.show();
    
    # sample 50000 extra samples.
    # inference the posteriori according to the observation with MCMC
    [model1_probs_extended, mus_extended, sigmas_extended], kernel_results = tfp.mcmc.sample_chain(
        num_results = 50000,
        num_burnin_steps = 0,
        current_state = [model1_probs[-1], mus[-1], sigmas[-1]],
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn = log_prob_generator(data),
                num_leapfrog_steps = 2,
                step_size = step_size,
                step_size_update_fn = tfp.mcmc.make_simple_step_size_update_policy(num_adaptation_steps = 1000),
                state_gradients_are_stopped = True
            ),
            bijector = bijectors
        )
    );
            
    print('acceptance rate: %f' % tf.math.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype = tf.float32)));
    print('final step size: %f' % tf.math.reduce_mean(kernel_results.inner_results.extra.step_size_assign[-100:]));
    
    plt.figure(figsize(12.5, 4));
    
    # draw the first 25000 samples
    x = tf.range(25000);
    plt.plot(x, mus[:, 0], label = 'previous trace of center 0', lw = 1, alpha = 0.4, c = colors[1]);
    plt.plot(x, mus[:, 1], label = 'previous trace of center 1', lw = 1, alpha = 0.4, c = colors[0]);
    # draw the following 50000 samples
    x = tf.range(25000,75000);
    plt.plot(x, mus_extended[:,0], label = 'new trace of center 0', lw = 1, c = '#5DA5DA');
    plt.plot(x, mus_extended[:,1], label = 'new trace of center 1', lw = 1, c = '#F15854');
    
    plt.title('Traces of unknown center parameters');
    leg = plt.legend(loc = 'upper right');
    leg.get_frame().set_alpha(0.8);
    plt.xlabel('Steps');
    
    plt.show();
    
    plt.figure(figsize(12.5, 8));
    for i in range(2):
        plt.subplot(2, 2, 2 * i + 1);
        plt.title('Posterior of center of cluster %d' % i);
        plt.hist(mus_extended[:, i], color = colors[i], bins = 30, histtype = 'stepfilled');
        
        plt.subplot(2, 2 , 2 * i + 2);
        plt.title('Posterior of standard deviation of cluster %d' % i);
        plt.hist(sigmas_extended[:, 1], color = colors[i], bins = 30, histtype = 'stepfilled');
    plt.tight_layout();
    
    plt.show();
    
    dist_0 = tfp.distributions.Normal(loc = tf.math.reduce_mean(mus_extended[:,0]), scale = tf.math.reduce_mean(sigmas_extended[:,0]));
    dist_1 = tfp.distributions.Normal(loc = tf.math.reduce_mean(mus_extended[:,1]), scale = tf.math.reduce_mean(sigmas_extended[:,1]));
    prob_assignment_1 = dist_0.prob(data);
    prob_assignment_2 = dist_1.prob(data);
    probs_assignments = 1. - (prob_assignment_1 / (prob_assignment_1 + prob_assignment_2));
    probs_assignments_inv = 1. - probs_assignments;
    # cluster_probs.shape = (data.shape[0], 2)
    cluster_probs = tf.transpose(tf.stack([probs_assignments, probs_assignments_inv]));
    # burned_assignment_trace.shape = (300, data.shape[0])
    burned_assignment_trace = tfp.distributions.Categorical(probs = cluster_probs).sample(sample_shape = 300);

    plt.figure(figsize(12.5, 5));
    plt.cmap = mpl.colors.ListedColormap(colors);
    output = tf.stack([tf.gather(burned_assignment_trace[i,...], tf.argsort(data)) for i in tf.range(burned_assignment_trace.shape[0])]);
    plt.imshow(output.numpy(), cmap = plt.cmap, aspect = .4, alpha = .9);
    plt.xticks(tf.range(0,data.shape[0],40).numpy(), ["%.2f" % s for s in tf.sort(data)[::40]]);
    plt.ylabel('posterior sample');
    plt.xlabel('value of $i$th data point');
    plt.title('Posterior labels of data points');
    
    plt.show();
    
    plt.figure(figsize(12.5, 5));
    plt.scatter(tf.gather(data, tf.argsort(data)).numpy(), tf.gather(probs_assignments, tf.argsort(data)), cmap = mpl.colors.LinearSegmentedColormap.from_list('BMH', colors), c = (1 - output).numpy(), s = 50);
    plt.ylim(-.05, 1.05);
    plt.title('Probability of data point belonging to cluster 0');
    plt.ylabel('probability');
    plt.xlabel('value of data point');
    
    plt.show();

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
            mixture_distribution = tfp.distributions.Categorical(probs = [model1_prob, model2_prob]),
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
