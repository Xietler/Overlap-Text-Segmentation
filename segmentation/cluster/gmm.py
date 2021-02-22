# -*- coding: utf-8 -*-
# @Time    : 2019/3/6 20:46
# @Author  : Ruichen Shao
# @File    : gmm.py

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch

from matplotlib.patches import Ellipse
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from torch.distributions import constraints
import time


@config_enumerate(default='parallel')
@poutine.broadcast
def model(data):
    # Global variables.
    weights = pyro.param('weights', torch.tensor([0.5]), constraint=constraints.unit_interval)
    scales = pyro.param('scales', torch.tensor([[[1., 0.], [0., 2.]], [[3., 0.], [0., 4.]]]), constraint=constraints.positive)
    locs = pyro.param('locs', torch.tensor([[1., 2.], [3., 4.]]))

    with pyro.plate('data', data.size(0)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Bernoulli((torch.ones(len(data)) * weights).cuda())).to(torch.int64).cuda()
        pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scales[assignment]), obs=data)


@config_enumerate(default="parallel")
@poutine.broadcast
def full_guide(data):
    with pyro.plate('data', data.size(0)):
        # Local variables.
        assignment_probs = pyro.param('assignment_probs', torch.ones(len(data)) / K,
                                      constraint=constraints.unit_interval)
        pyro.sample('assignment', dist.Bernoulli(assignment_probs), infer={"enumerate": "sequential"})


def initialize(data):
    pyro.clear_param_store()
    import pyro.optim as optim
    optim = optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    svi = SVI(model, full_guide, optim, loss=elbo)

    # Initialize weights to uniform.
    pyro.param('auto_weights', 0.5 * torch.ones(K), constraint=constraints.simplex)

    # Assume half of the data variance is due to intra-component noise.
    var = (data.var() / 2).sqrt()
    pyro.param('auto_scale', torch.tensor([var]*4), constraint=constraints.positive)

    # Initialize means from a subsample of data.
    pyro.param('auto_locs', data[torch.multinomial(torch.ones(len(data)) / len(data), K)])

    loss = svi.loss(model, full_guide, data)

    return loss, svi


def get_samples():
    num_samples = 5000

    # 2 clusters
    # note that both covariance matrices are diagonal
    mu1 = torch.tensor([0., 5.])
    sig1 = torch.tensor([[2., 0.], [0., 3.]])

    mu2 = torch.tensor([5., 0.])
    sig2 = torch.tensor([[4., 0.], [0., 1.]])

    # generate samples
    dist1 = dist.MultivariateNormal(mu1, sig1)
    samples1 = [pyro.sample('samples1', dist1) for _ in range(num_samples)]

    dist2 = dist.MultivariateNormal(mu2, sig2)
    samples2 = [pyro.sample('samples2', dist2) for _ in range(num_samples)]

    data = torch.cat((torch.stack(samples1), torch.stack(samples2)))
    return data


def plot(data, mus=None, sigmas=None, colors='black', figname='fig.png'):
    # Create figure
    fig = plt.figure()

    # Plot data
    x = data[:, 0].cpu()
    y = data[:, 1].cpu()
    plt.scatter(x, y, 24, c=colors)

    # Plot cluster centers
    if mus is not None:
        x = [float(m[0]) for m in mus]
        y = [float(m[1]) for m in mus]
        plt.scatter(x, y, 99, c='red')

    # Plot ellipses for each cluster
    if sigmas is not None:
        for sig_ix in range(K):
            ax = fig.gca()
            cov = np.array(sigmas[sig_ix].cpu())
            lam, v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            ell = Ellipse(xy=(x[sig_ix], y[sig_ix]),
                          width=lam[0]*4, height=lam[1]*4,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color='blue')
            ell.set_facecolor('none')
            ax.add_artist(ell)

    # Save figure
    fig.savefig(figname)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    pyro.enable_validation(True)
    pyro.set_rng_seed(42)

    # Create our model with a fixed number of components
    K = 2

    data = get_samples()

    start = time.time()
    global_guide = AutoDelta(poutine.block(model, expose=['weights', 'locs', 'scales']))
    global_guide = config_enumerate(global_guide, 'parallel')
    _, svi = initialize(data)

    true_colors = [0] * 5000 + [1] * 5000
    plot(data, colors=true_colors, figname='pyro_init.png')

    for i in range(151):
        svi.step(data)

        if i % 50 == 0:
            print(1)
            locs = pyro.param('locs')
            scales = pyro.param('scales')
            weights = pyro.param('weights')
            assignment_probs = pyro.param('assignment_probs')

            # print("locs: {}".format(locs))
            # print("scales: {}".format(scales))
            # print('weights = {}'.format(weights))
            # print('assignments: {}'.format(assignment_probs))

            # todo plot data and estimates
            assignments = assignment_probs.cpu().data.numpy()
            threshold = 0.2
            assignments = np.where(np.logical_and(assignments<=threshold, assignments>=0), 0, assignments)
            assignments = np.where(np.logical_and(assignments>=1-threshold, assignments<=1), 1, assignments)
            assignments = np.where(np.logical_and(assignments>threshold, assignments<1-threshold), 2, assignments)
            # print(assignments)
            plot(data, locs.data, scales.data, assignments, figname='pyro_iteration{}.png'.format(i))

    print('Total time: {}'.format(time.time()-start))