#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:59:32 2022

@author: dhulls
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt

## Effective sample size

target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])
# Get 1000 states from one chain.
states = tfp.mcmc.sample_chain(
    num_burnin_steps=200,
    num_results=1000,
    current_state=tf.constant([0., 0.]),
    trace_fn=None,
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target.log_prob,
      step_size=0.05,
      num_leapfrog_steps=20))
states.shape
ess = tfp.mcmc.effective_sample_size(states, filter_beyond_positive_pairs=True)

# R_hat

target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])

# Get 10 (2x) overdispersed initial states.
initial_state = target.sample(10) * 2.

# Get 1000 samples from the 10 independent chains.
chains_states = tfp.mcmc.sample_chain(
    num_burnin_steps=200,
    num_results=1000,
    current_state=initial_state,
    trace_fn=None,
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=0.05,
        num_leapfrog_steps=20))
chains_states.shape

rhat = tfp.mcmc.diagnostic.potential_scale_reduction(
    chains_states, independent_chain_ndims=1)

# The second dimension needed a longer burn-in.
rhat.eval()