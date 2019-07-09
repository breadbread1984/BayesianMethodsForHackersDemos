#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
import matplotlib.pyplot as plt;
from IPython.core.pylabtools import figsize;

def main():

    # unknown ground truth
    lambda1 = tf.constant(1.);
    lambda2 = tf.constant(3.);

    # sample an fake observation data ~ poisson(data[0]; lambda1) * poisson(data[1]; lambda2);
    data = tf.stack([
        tfp.distributions.Poisson(rate = lambda1).sample(sample_shape = (1), seed = 4),
        tfp.distributions.Poisson(rate = lambda2).sample(sample_shape = (1), seed = 8)
    ]);

    # (posteriori L) likelihood(lambda1,lambda2 | data)  = p(data | lambda1, lambda2) = poisson(data[0] ; lambda1) * poisson(data[1] ; lambda2)
    # x represents lambda1, y represents lambda2
    x = y = tf.linspace(.01, 5., 100);
    prob_x = tfp.distributions.Poisson(rate = x).prob(data[0,...]);
    prob_y = tfp.distributions.Poisson(rate = y).prob(data[1,...]);
    L = tf.expand_dims(prob_x, 1) * tf.expand_dims(prob_y, 0);
    # (prior M) p(lambda1, lambda2) = P(lambda1) * P(lambda2), where lambda1 ~ U(0,5), lambda2 ~ U(0,5)
    uniform_x = tfp.distributions.Uniform(low = 0., high = 5.).prob(x);
    m = median(tf.gather_nd(uniform_x,tf.where(tf.greater(uniform_x,0))));
    uniform_x = tf.where(tf.equal(uniform_x, 0), uniform_x, m);
    uniform_y = tfp.distributions.Uniform(low = 0., high = 5.).prob(y);
    m = median(tf.gather_nd(uniform_y,tf.where(tf.greater(uniform_y,0))));
    uniform_y = tf.where(tf.equal(uniform_y, 0), uniform_y, m);
    M = tf.expand_dims(uniform_x, 1) * tf.expand_dims(uniform_y, 0);

    plt.figure(figsize(12.5, 15.0));

    # 1) plot P(lambda1, lambda2) = P(lambda1) * P(lambda2)
    # lambda ~ Uniform(0, 5)
    plt.subplot(221);
    im = plt.imshow(M.numpy(), interpolation = 'none', origin = 'lower', cmap = plt.cm.jet, vmax = 1, vmin = -.15, extent = (0, 5, 0, 5));
    plt.scatter(lambda2.numpy(), lambda1.numpy(), c = 'k', s = 50, edgecolor = 'none');
    plt.xlim(0, 5);
    plt.ylim(0, 5);
    plt.title(r'Landscape formed by Uniform priors on $p_1, p2$');
    # 2) plot P(lambda1, lambda2, data) = p(lambda1, lambda2) * p(data | lambda1, lambda2)
    plt.subplot(223);
    plt.contour(x.numpy(), y.numpy(), (M * L).numpy());
    im = plt.imshow(M * L, interpolation = 'none', origin = 'lower', cmap = plt.cm.jet, extent = (0, 5, 0, 5));
    plt.title('Landscape warped by %d data observation;\n Uniform priors on $p_1, p_2$.' % 1);
    plt.scatter(lambda2.numpy(), lambda1.numpy(), c = 'k', s = 50, edgecolor = 'none');
    plt.xlim(0, 5);
    plt.ylim(0, 5);
    # 3) plot P(lambda1, lambda2) = P(lambda1) * P(lambda2)
    # lambda1 ~ Exponential(0.3)
    # lambda2 ~ Exponential(0.1)
    plt.subplot(222);
    expx = tfp.distributions.Exponential(rate = .3).prob(x);
    expx = tf.where(tf.math.is_nan(expx),tf.ones_like(expx) * expx[1], expx);
    expy = tfp.distributions.Exponential(rate = .10).prob(y);
    expy = tf.where(tf.math.is_nan(expy),tf.ones_like(expy) * expy[1], expy);
    M = tf.expand_dims(expx,1) * tf.expand_dims(expy,0);
    plt.contour(x,y,M);
    im = plt.imshow(M, interpolation = 'none', origin = 'lower', cmap = plt.cm.jet, extent = (0, 5, 0, 5));
    plt.scatter(lambda2.numpy(), lambda1.numpy(), c = 'k', s = 50, edgecolor = 'none');
    plt.xlim(0,5);
    plt.ylim(0,5);
    plt.title('Landscape formed by Exponential priors on $p_1, p_2$.');
    # 4) plot P(lambda1, lambda2, data) = P(lambda1, lambda2) * p(data | lambda1, lambda2)
    plt.subplot(224);
    plt.contour(x, y, M * L);
    im = plt.imshow(M * L, interpolation = 'none', origin = 'lower', cmap = plt.cm.jet, extent = (0, 5, 0, 5));
    plt.title('Landscape warped by %d data objservation; \n Exponential priors on $p_1, p_2$.' % 1);
    plt.scatter(lambda2.numpy(), lambda1.numpy(), c = 'k', s = 50, edgecolor = 'none');
    plt.xlim(0, 5);
    plt.ylim(0, 5);
    
    plt.show();

def median(x):

    mid = x.shape[0] // 2 + 1;
    return tf.math.top_k(x,mid).values[-1];

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();

