import shutil
import sys
from pathlib import Path
import torch.autograd as autograd
import argparse
from functools import partial

import torch
from torch.nn import functional as F
import torch.nn as nn

import numpy as np

import egg.core as core

#  def gumbel_softmax(logits):
#      # gave up for pytorch's native functions
#      uniform = torch.zeros_like(logits).uniform_()
#      # gumbel's density exp(-g -exp(-g))
#      g = -torch.log(-torch.log(uniform))
#      log_p_g = (-g - torch.exp(-g)).sum()
#      noisy_logits = logits + g
#      return noisy_logits, log_p_g

def conditional_relax(log_theta, theta, b):
    """ sample from p(z|b, theta)
    """
    v = torch.zeros_like(theta).uniform_()
    v_prime = ((b==0) * (v * (1 - theta)) + 
               (b==1) * ((v * theta) + (1 - theta)))
    return log_theta + torch.logit(v_prime)


def f(b, t):
    return (b-t)**2
    

def log_pmf_bernoulli(theta, b):
    """ No mini-batch here!
    """
    if b == 0:
        return torch.log1p(-theta)
    else:
        return torch.log(theta)

def reinforce_(loss, theta, b):
    log_p_b = log_pmf_bernoulli(theta, b)
    return (loss(b)).detach() * log_p_b 


def reinforce(loss, log_theta, log_temp=None):
    theta = torch.sigmoid(log_theta)
    b = torch.distributions.bernoulli.Bernoulli(probs=theta).sample()
    return reinforce_(loss, theta, b)

def dlax(loss, log_theta, log_temp):
    print("temp ignored")
    concrete = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(
        temperature=log_temp.exp(), logits=log_theta,
    )
    #  z, log_p_z = gumbel_softmax(log_theta)
    z = concrete.rsample()
    log_p_z = concrete.log_prob(z)
    b = (z > 0).float()
    theta = torch.sigmoid(log_theta)
    c = loss(torch.sigmoid(z / T))
    return reinforce_(loss, theta, b) - (c.detach() * log_p_z) + c

def one_run(loss, eps, max_iter, estimator, lr):
    # minimize E_p(b)[(b-t)^2]
    log_theta = nn.Parameter(torch.tensor([0.]))
    log_temp = nn.Parameter(torch.tensor([0.5]))
    optimizer = torch.optim.Adam([log_theta], lr)#, momentum=0)
    i = 0
    for i in range(1, max_iter+1):
        l = estimator(loss, log_theta, log_temp)
        l.backward()
        #  print("∇={}".format(log_theta.grad.item()))
        optimizer.step()
        #  print("θ={:.3f}, l={:.3f}".format(S(log_theta).item(), l.item()))
        theta = torch.sigmoid(log_theta)
        if theta.abs().item() < eps or theta.abs().item() > 1.-eps:
            break
        optimizer.zero_grad()
    return i, theta.item(), log_temp.exp()


def rebar(loss, log_theta, log_temp, eta, params_theta, net=None):
    concrete = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(
        temperature=log_temp.exp(), logits=log_theta,
    )
    z = concrete.rsample()
    b = (z > 0).int()
    theta = torch.sigmoid(log_theta)
    z_tilda = conditional_relax(log_theta, theta, b)
    T = log_temp.exp()
    c_z = loss(torch.sigmoid(z / T))
    c_z_tilda = loss(torch.sigmoid(z_tilda / T))
    if net:
        c_z += net(z)
        c_z_tilda += net(z_tilda)
        #  print(net(z), net(z_tilda))
    log_p_b = log_pmf_bernoulli(theta, b)
    # we have to retrain_graph, b/c it is then used for computing the second
    # derivative
    # TODO how does it work when theta is produced by many parameters?
    #      is autograd.grad on each efficient, in that case? I don't think so,
    #      it should be a simple backward()?
    log_p_b_grad, = autograd.grad([log_p_b], params_theta, create_graph=True,
            retain_graph=True)
    c_z_grad, = autograd.grad([c_z], params_theta, create_graph=True, retain_graph=True)
    c_z_tilda_grad, = autograd.grad([c_z_tilda], params_theta, create_graph=True, retain_graph=True)
    g = [(((loss(b) - eta * c_z_tilda) * log_p_b_g) +
         eta * (c_z_g - c_z_tilda_g)) for c_z_g, c_z_tilda_g, log_p_b_g in
             zip(c_z_grad, c_z_tilda_grad, log_p_b_grad)]
    return g


def one_run_relax(loss, eps, max_iter, estimator, lr, relax):
    # minimize E_p(b)[(b-t)^2]
    log_theta = nn.Parameter(torch.tensor([0.]))
    log_temp = nn.Parameter(torch.tensor([0.]))
    eta = nn.Parameter(torch.tensor([1.]))
    if relax:
        net = torch.nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1, bias=False),
        )
    else:
        net = None
    params_phi = [eta, log_temp]
    if relax:
        params_phi += list(net.parameters())
    optimizer_phi = torch.optim.Adam(params_phi, lr)
    params_theta = [log_theta]
    optimizer_theta = torch.optim.Adam(params_theta, lr)#, momentum=0)
    i = 0
    for i in range(1, max_iter+1):
        grads = estimator(loss, log_theta, log_temp, eta, params_theta, net=net)
        for p, grad in zip(params_theta, grads):
            p.backward(grad)
            var_loss = (grad**2).mean()
            var_loss.backward()
        optimizer_theta.step()
        optimizer_phi.step()
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()

        theta = torch.sigmoid(log_theta)
        #  print("θ={:.3f}, ∇θ^2={:.5f}".format(theta.item(), var_loss))
        if theta.abs().item() < eps or theta.abs().item() > 1.-eps:
            break
    return i, theta.item(), log_temp.exp()


def eval_estimator(estimator, exp_runner, true_thetas, max_iter, lr):
    successes = 0
    tol = 1e-2
    n_iterations = []
    for true_theta in true_thetas: 
        loss = lambda b: f(b, true_theta)
        r = exp_runner(loss, tol, max_iter, estimator, lr)
        i, pred_theta, temp = r 
        print(i, true_theta, pred_theta, temp)
        if i < max_iter:
            successes += int((true_theta > 0.5) == (pred_theta > 0.5)) 

            n_iterations.append(i)
    print("n_successes={}, avg_n_success_iter={}".format(
        successes, np.mean(n_iterations),
    ))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max-iter', default=1000, type=int)
    parser.add_argument('-t', default=0.495, type=float)
    parser.add_argument('estimator')
    args = parser.parse_args()
    torch.manual_seed(0)
    n = 5
    u = 0.05
    max_iter = args.max_iter
    #  true_theta = 0.50 + (torch.rand(n) * 2 * u) - u
    true_theta = torch.tensor([(args.t,)*n]).squeeze()
    for estimator in args.estimator.split(','):
        if estimator == 'rf':
            eval_estimator(reinforce, one_run, true_theta, max_iter, args.lr)
        elif estimator == 'rebar':
            raise NotImplementedError()
            runner = partial(one_run_relax, relax=False)
            eval_estimator(rebar, runner, true_theta, max_iter, args.lr)
        elif estimator == 'relax':
            runner = partial(one_run_relax, relax=True)
            eval_estimator(rebar, runner, true_theta, max_iter, args.lr)
