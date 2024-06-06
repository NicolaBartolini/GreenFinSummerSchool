#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:41:50 2024

@author: mac
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
 
 
############# computes mean and covariance of a sample based on a vector of probabilities
def mean_and_cov(y, p=None):   ####
    if p is None:
        p = np.ones(len(y)) / len(y)
    e_y = np.dot(p,y)
    cov_y = ((y-e_y).T*p) @ (y-e_y)
    return e_y, cov_y
 
######## transforms covariance matrix into correlation matrix #########
def covariance_to_correlation(cov):
    std = np.sqrt(np.diag(cov))     #find standard deviations
    inv_diag_std = np.diag(1/std) #diagonal matrix with inverse std as entries
    #find correlation matrix
    corr = np.dot(np.dot(inv_diag_std,cov),inv_diag_std)
    return corr
 
 
def dcc_parameters(xi, p=None, rho2=None):  
    # xi = standardized residuals from univariate regressions
    t_ = len(xi)
    n_ = xi.shape[1]
    if p is None:
        p = np.full(shape=t_, fill_value=1/t_) #equal probabilities unless otherwise provided
    if rho2 is None:  #method of moments estimator if not provided
        _, rho2 = mean_and_cov(xi, p)
        rho2 = covariance_to_correlation(rho2)
    param_initial = [0.01, 0.99]  #initial guess for parameters
    #find (negative) log-likelihood of GARCH (to then minimize)
    def neg_llh_gauss(parameters):
        mu = np.zeros(n_) #they are standardized
        a, b = parameters
        q2_t = rho2.copy() #initialize value as = q bar
        r2_t = covariance_to_correlation(q2_t)
        n_llh = 0.0 #initialize negative loglikelihood
        for t in range(t_):
            n_llh = n_llh - p[t] * multivariate_normal.logpdf(xi[t, :], mu, r2_t)
            q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
            r2_t = covariance_to_correlation(q2_t)
        return n_llh
    #minimize neg. log-likelihood
    #set boundaries
    bounds = ((1e-21, 1.), (1e-21, 1.))
    #we impose a stationary constraint
    cons = {'type': 'ineq', 'fun': lambda param: 0.99 - param[0] - param[1]}
    #we find minimizer parameters 
    a, b = minimize(neg_llh_gauss, param_initial, bounds=bounds, constraints=cons)['x']
    #compute realized correlations and residuals
    q2_t = rho2.copy() #initialize value as = q bar
    r2_t = np.zeros((t_, n_, n_)) #there are t_   n_xn_  matrices of dcc
    r2_t[0, :, :] = covariance_to_correlation(q2_t) #the first one is the initial value
    for t in range(1, t_):
        q2_t = rho2 * (1 - a - b) + a * np.outer(xi[t, :], xi[t, :]) + b * q2_t
        r2_t[t, :, :] = covariance_to_correlation(q2_t)
    l_t = np.linalg.cholesky(r2_t)
    epsi = np.linalg.solve(l_t, xi)
    return [1. - a - b, a, b], r2_t, epsi, q2_t




























