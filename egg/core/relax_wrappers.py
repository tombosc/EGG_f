import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.autograd as autograd
from typing import Callable, Optional
from .interaction import LoggingStrategy


def compute_grad(term, params):
    """ Compute grad of term wrt. list params; keep graph for 2nd derivative.
    """
    # create_graph=True is necessary for higher order derivatives
    # autograd.grad "computes and returns the sum of gradients of outputs", so
    # we have to pass each output one after the other. (I don't know why they
    # don't simply require a scalar there, forcing the user to sum/average?)
    return autograd.grad([term], params, create_graph=True, retain_graph=True)


def RELAX_builder(loss):
    class RELAX(autograd.Function):
        @staticmethod
        def forward(ctx, b, z, distribution, log_temp, eta, net, params_theta):
            # rebar_categorical(loss, logits, log_temp, eta, params_theta, net=None):
            logits = distribution.logits
            theta = distribution.probs
            log_p_b = distribution.log_prob(b)
            z_tilda = conditional_relax_categorical(logits, theta, b)
            T = log_temp.exp()
            # compute control variate
            loss_b = loss(b)
            c_z = loss(torch.softmax(z / T, dim=1)).mean()
            #  print("Here, c_z={}, net={}".format(c_z.size(), net(z).size()))
            c_z += net(z).mean()
            c_z_tilda = loss(torch.softmax(z_tilda / T, dim=1)).mean()
            c_z_tilda += net(z_tilda).mean()
            ctx.save_for_backward(log_p_b, c_z, c_z_tilda, eta, loss_b, *params_theta)
            return loss_b


        @staticmethod
        def backward(ctx, grad_output):
            print("BACKW")
            print(grad_output)
            log_p_b, c_z, c_z_tilda, eta, loss_b, *params_theta = ctx.saved_tensors
            print("params:", [p.size() for p in params_theta])
            print("params:", [p.grad_fn for p in params_theta])
            print("params:", [p.requires_grad for p in params_theta])
            # TODO how to pass params_theta
            import pdb; pdb.set_trace()
            c_z_grad = compute_grad(c_z, params_theta)
            c_z_tilda_grad = compute_grad(c_z_tilda, params_theta)
            log_p_b_grad = compute_grad(log_p_b, params_theta)

            grads = estimator(loss, log_theta, log_temp, eta, params_theta, net=net)
            for p, grad in zip(params_theta, grads):
                print("GRAD", p)
                p.backward(grad)
                var_loss = (grad**2).mean()
                var_loss.backward()

            # compute gradient estimator
            g = [(((loss_b - eta * c_z_tilda) * log_p_b_g) +
                 eta * (c_z_g - c_z_tilda_g)) for c_z_g, c_z_tilda_g, log_p_b_g in
                     zip(c_z_grad, c_z_tilda_grad, log_p_b_grad)]
            return grad_output
    return RELAX

def conditional_relax_bernoulli(log_theta, theta, b):
    """ sample from p(z|b, theta)
    """
    v = torch.zeros_like(theta).uniform_()
    v_prime = ((b==0) * (v * (1 - theta)) + 
               (b==1) * ((v * theta) + (1 - theta)))
    return log_theta + torch.logit(v_prime)


def log_pmf_bernoulli(theta, b):
    """ No mini-batch here!
    """
    if b == 0:
        return torch.log1p(-theta)
    else:
        return torch.log(theta)


def conditional_relax_categorical(log_theta, theta, b):
    z = torch.zeros_like(theta)
    v = torch.rand(z.size())
    bs = b.size(0)
    arange = torch.arange(bs)
    sample_v = v[arange, b]
    # TODO check
    z = -torch.log(-torch.log(v) / theta - torch.log(sample_v.unsqueeze(1)))
    z[arange, b] = -torch.log(-torch.log(sample_v))
    return z


def rebar_categorical(loss, logits, log_temp, eta, params_theta, net=None):
    z = logits - torch.log( - torch.log( torch.rand(logits.size())))
    b = z.argmax(1)
    d = torch.distributions.categorical.Categorical(logits=logits)
    theta = d.probs
    z_tilda = conditional_relax_categorical(logits, theta, b)
    T = log_temp.exp()
    log_p_b = d.log_prob(b)
    # compute control variate
    c_z = loss(torch.softmax(z / T, dim=1))
    c_z_tilda = loss(torch.softmax(z_tilda / T, dim=1))
    if net:
        c_z += net(z)
        c_z_tilda += net(z_tilda)
    # gather gradients
    log_p_b_grad = compute_grad(log_p_b, params_theta)
    c_z_grad = compute_grad(c_z, params_theta)
    c_z_tilda_grad = compute_grad(c_z_tilda, params_theta)
    # compute gradient estimator
    g = [(((loss(b) - eta * c_z_tilda) * log_p_b_g) +
         eta * (c_z_g - c_z_tilda_g)) for c_z_g, c_z_tilda_g, log_p_b_g in
             zip(c_z_grad, c_z_tilda_grad, log_p_b_grad)]
    return g


def rebar_bernoulli(loss, log_theta, log_temp, eta, params_theta, net=None):
    concrete = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(
        temperature=log_temp.exp(), logits=log_theta,
    )
    z = concrete.rsample()
    b = (z > 0).int()
    theta = torch.sigmoid(log_theta)
    z_tilda = conditional_relax_bernoulli(log_theta, theta, b)
    T = log_temp.exp()
    c_z = loss(torch.sigmoid(z / T))
    c_z_tilda = loss(torch.sigmoid(z_tilda / T))
    if net:
        c_z += net(z)
        c_z_tilda += net(z_tilda)
    log_p_b = log_pmf_bernoulli(theta, b)

    log_p_b_grad = compute_grad(log_p_b, params_theta)
    c_z_grad = compute_grad(c_z, params_theta)
    c_z_tilda_grad = compute_grad(c_z_tilda, params_theta)
    g = [(((loss(b) - eta * c_z_tilda) * log_p_b_g) +
         eta * (c_z_g - c_z_tilda_g)) for c_z_g, c_z_tilda_g, log_p_b_g in
             zip(c_z_grad, c_z_tilda_grad, log_p_b_grad)]
    return g




class RelaxSenderWrapper(nn.Module):
    """ Relax TODO
    """

    def __init__(
        self,
        agent,
    ):
        """
        :param agent: The agent to be wrapped. agent.forward() has to output log-probabilities over the vocabulary
        """
        super(RelaxSenderWrapper, self).__init__()
        self.agent = agent
        #  self.log_theta = nn.Parameter(torch.tensor([0.]))
        #  self.log_temp = nn.Parameter(torch.tensor([0.]))
        #  self.eta = nn.Parameter(torch.tensor([1.]))
        #  self.control_variate = torch.nn.Sequential(
        #      nn.Linear(1, 10),
        #      nn.Tanh(),
        #      nn.Linear(10, 1)
        #  )
        #  params_theta = [self.log_theta]
        #  params_phi = [self.eta, self.log_temp] + list(self.control_variate)

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        z = logits - torch.log( - torch.log( torch.rand(logits.size())))
        sample = z.argmax(1)
        bs = z.size(0)
        one_hot_sample = torch.zeros_like(logits)
        one_hot_sample[torch.arange(bs), sample] = 1.
        distrib = torch.distributions.categorical.Categorical(logits=logits)
        # log_p_b = d.log_prob(b)
        return sample, one_hot_sample, distrib


class RelaxGame(nn.Module):
    """ Sender/receiver game with RELAX gradient estimator.
    """
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        """
        """
        super(RelaxGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )
        self.log_temp = nn.Parameter(torch.tensor([0.]))
        self.eta = nn.Parameter(torch.tensor([1.]))
        vocab_size = sender.agent.vocab_size
        self.net = torch.nn.Sequential(
            nn.Linear(vocab_size, 10),
            nn.Tanh(),
            nn.Linear(10, 1, bias=False),
        )


    def forward(self, sender_input, labels, receiver_input=None):
        message, z, distribution = self.sender(sender_input)
        receiver_output = self.receiver(
            message, receiver_input
        )

        print("Furst call to loss:", message.size(), receiver_output.size())
        loss, aux_info = self.loss(
            sender_input, message, distribution, receiver_input, receiver_output, labels
        )
        
        #  if self.training:
        #      _verify_batch_sizes(loss, sender_log_prob, receiver_log_prob)

        if self.training:
            # TODO loss!
            pass

        #  full_loss = policy_loss + entropy_loss + loss.mean()

        receiver_entropy = distribution.entropy()
        aux_info["sender_entropy"] = distribution.entropy().detach()
        aux_info["receiver_entropy"] = receiver_entropy.detach()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        def closure_loss(closure_message):
            closure_receiver_output = self.receiver(message, receiver_input)
            print("Call closure msg={}, rcv_out={}".format(
                closure_message.size(), closure_receiver_output.size()))
            loss, _ = self.loss(
                _sender_input=sender_input,
                _message=closure_message,
                distrib_message=distribution,
                _receiver_input=receiver_input,
                receiver_output=closure_receiver_output,
                labels=labels,
            )
            return loss.mean()

        relax = RELAX_builder(closure_loss)
        sender_params = list(self.sender.parameters())
        full_loss = relax.apply(message, z, distribution,
                                self.log_temp, self.eta, self.net,
                                sender_params)
        return full_loss, interaction
