#!/usr/bin/env python3
# coding: utf-8
# Licensed under the terms of the MIT license, see LICENSE.md

import argparse
import jax 
import math
import os
import sys

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from collections import defaultdict
from scipy.optimize import minimize
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from jax.config import config
config.update("jax_enable_x64", True)

from simplex_projection import euclidean_proj_l1ball
from datasets import *

@jax.jit    
def logistic(r):
    return jax.scipy.special.expit(r)

@jax.jit    
def softabs(r, root_eps=1e-8):
    return r**2/jnp.sqrt(r**2+root_eps)

@jax.jit
def crossent(x, y): # numerically stable following https://github.com/ddehueck/jax-loss/blob/master/jax_loss/classification.py 
    clipped_val = jnp.clip(x, 0, None)
    loss = x - x*y + clipped_val + jnp.log(jnp.exp(-clipped_val) + jnp.exp((-x-clipped_val)))
    return loss.mean()
             
@jax.jit
def theta_to_cw(theta):
    return theta[0],theta[1:]

@jax.jit
def cw_to_theta(c,w):
    return jnp.hstack([c, w])

@jax.jit
def predict_decision_function(theta, X, clip=20):
    c,w = theta_to_cw(theta)
    logits = jnp.dot(X,w) + c
    if clip:
      logits = jnp.clip(logits, -clip, clip)
    return logits
    
@jax.jit
def predict_proba(theta, X):
    dec = predict_decision_function(theta, X)
    return logistic(dec)

@jax.jit
def proba_to_pred(prob):
    return 1*(prob>=0.5)

@jax.jit
def proba_to_dec(prob):
    return jnp.log(1e-10+prob) - jnp.log(1-prob+1e-10)

@jax.jit
def dec_to_pred(dec):
    return 1*(dec>=0)

@jax.jit
def predict(theta, X):
    dec = predict_decision_function(theta, X)
    return dec_to_pred(dec)

@jax.jit
def predict_all(theta, X):
    dec = predict_decision_function(theta, X)
    return dec, dec_to_pred(dec), logistic(dec)

def predict_all_xgboost(clf, X): # clf: XGBClassifier
    prob = clf.predict_proba(X)[:,1]
    return proba_to_dec(prob), proba_to_pred(prob), prob

def train_fairreg(ds, data, lam=0., eta=0.):
    """Train logistic regression classifier with demographic parity regularizer"""
    X,y = ds.features(data)
    
    mask01 = jnp.asarray(data['target']==1, dtype=bool) # mask if target=1 or 0
    X0,X1 = X[~mask01],X[mask01] # predict 0 on a and 1 on not-a
    
    maskab = jnp.asarray(data['protected']==1, dtype=bool) # mask if protected=1 or 0
    Xa,Xb = X[maskab],X[~maskab]
    
    theta0 = train_sklearn_Xy(X, y, lam=1)
    res = minimize(
        cost_fairreg,
        theta0,
        args=(X0, X1, Xa, Xb, eta),
        method="BFGS",
        jac=jax.grad(cost_fairreg),
        options={"gtol": 1e-4, "disp": False, "maxiter": 500}
    )
    return res.x

@jax.jit
def disparity(theta, X0, X1, Xa, Xb): # demographic parity violation
    p0 = predict(theta, X0) # label 0
    p1 = predict(theta, X1) # label 1
    pa = predict(theta, Xa) # group a
    pb = predict(theta, Xb) # group b

    loss = jnp.abs(jnp.mean(p1) - jnp.mean(p0))
    fairness = jnp.abs(jnp.mean(pa) - jnp.mean(pb))
    return jnp.abs(loss - fairness)

@jax.jit
def cost_fairreg_ext(theta, X0, X1, Xa, Xb):
    p0 = predict_decision_function(theta, X0)
    p1 = predict_decision_function(theta, X1)
    pa = predict_proba(theta, Xa)
    pb = predict_proba(theta, Xb)

    loss = (crossent(p0, y=0)+crossent(p1, y=1))/2.
    fairness = softabs(jnp.mean(pa)-jnp.mean(pb))
    return loss, fairness

@jax.jit
def cost_fairreg(theta, X0, X1, Xa, Xb, eta=0.5):
    loss, fairness = cost_fairreg_ext(theta, X0, X1, Xa, Xb)
    return loss + eta*fairness


@jax.jit
def cost_fairadv_ext(theta, X, y, theta_adv, y_adv):
    logits = predict_decision_function(theta, X)
    loss = crossent(logits, y)
    
    c, w = theta_to_cw(theta)
    reg = 0.5*jnp.dot(w, w)

    c_adv, w_adv = theta_to_cw(theta_adv)
    logits_adv = w_adv*logits-c_adv # linear adversary
    loss_adv = crossent(logits_adv, y_adv)
    return loss, reg, loss_adv

@jax.jit
def cost_fairadv(theta, X, y, theta_adv, y_adv, lam, eta):
    loss, reg, loss_adv = cost_fairadv_ext(theta, X, y, theta_adv, y_adv)
    return loss + lam*reg - eta*loss_adv

def train_fairadv(ds, data, lam=0., eta=0., learning_rate=1e-3, learning_rate_adv=1e-3, nsteps=1000):
    X, y = ds.features(data)
    
    theta = train_sklearn(ds, data, lam=1) # inialize without fairness, but avoid extreme values in w
    opt = optax.adam(learning_rate)
    opt_state = opt.init(theta)
    
    dec = predict_decision_function(theta, X).reshape(-1,1)
    y_adv = jnp.asarray(data['protected'], dtype=float) # adversary tries to predict protected attribute
    
    theta_adv = cw_to_theta(0.,0.)
    opt_adv = optax.adam(learning_rate_adv)
    opt_adv_state = opt_adv.init(theta_adv)
    
    for i in range(nsteps):
      grads = jax.grad(cost_fairadv, argnums=0)(theta, X, y, theta_adv, y_adv, lam=lam, eta=eta)
      updates, opt_state = opt.update(grads, opt_state)
      theta = optax.apply_updates(theta, updates)

      grads_adv = -jax.grad(cost_fairadv, argnums=3)(theta, X, y, theta_adv, y_adv, lam=lam, eta=eta)
      updates_adv, opt_adv_state = opt_adv.update(grads_adv, opt_adv_state)
      theta_adv = optax.apply_updates(theta_adv, updates_adv)

    return theta

@jax.jit
def cost_fair_const_ext(theta, X0, X1, Xa, Xb, wa, wb, xi): # like fairreg but with per-sample weights in fairness term
    # X0,X1: data for class 0,1
    # Xa,Xb: data for protected group A,B
    # wa,wb: per-sample weights for groups A,B
    # xi: slack variable for fairness constraints
    n0,n1 = len(X0),len(X1)
    f0 = predict_decision_function(theta, X0)
    f1 = predict_decision_function(theta, X1)
    wa = wa/wa.sum() # L1-normalize weights
    wb = wb/wb.sum() 

    loss = (crossent(f0, y=0)+crossent(f1, y=1))/2.
    
    pa = predict_proba(theta, Xa)
    pb = predict_proba(theta, Xb)
    constrainta = (jnp.abs(n1/(n0+n1) - jnp.dot(wa,pa)) - xi) # violation of equality constraint
    constraintb = (jnp.abs(n1/(n0+n1) - jnp.dot(wb,pb)) - xi)
    
    constraints = jnp.asarray([constrainta,constraintb])
    constraints = jnp.maximum(constraints, 0)
        
    return loss, constraints

def cost_fair_const(theta, X0, X1, Xa, Xb, wa, wb, lag, xi):
    # lag: lagrange multiplier \lambda_j for fairness constraints a0,a1,b0,b1
    loss, constraints = cost_fair_const_ext(theta, X0, X1, Xa, Xb, wa, wb, xi)
    return loss + jnp.dot(lag, constraints)

def sample_in_l1ball(center, radius, k=1):
    '''
    generate a random vector in L1-ball around v with radius r
    see http://www.nrbook.com/devroye/ (page 207)
    '''
    n = len(center)
    tmp = np.random.uniform(size=(k,n+1))
    tmp[:,0]=0.
    tmp[:,-1]=1.
    tmp.sort(axis=1)
    diffs = tmp[:,1:]-tmp[:,:-1]
    signs = (2*np.random.randint(0,2,size=diffs.shape)-1)
    result = center + signs*diffs*radius
    result = np.add(diffs, center)
    result = np.maximum(result, np.zeros(n))
    return result
    
def extended_fairness_constraints(theta, X0, X1, Xa, Xb, wa0, wb0, tv, xi):
    n0,n1 = len(X0),len(X1)
    pa = predict_proba(theta, Xa)
    pb = predict_proba(theta, Xb)
    many_wa = sample_in_l1ball(wa0, 2*tv, k=10)
    many_wa /= many_wa.sum(axis=1, keepdims=True)
    many_wb = sample_in_l1ball(wb0, 2*tv, k=10)
    many_wb /= many_wb.sum(axis=1, keepdims=True)
    constrainta = jnp.max(jnp.abs(n1/(n0+n1) - jnp.dot(many_wa, pa)) - xi) # max violation of equality constraint
    constraintb = jnp.max(jnp.abs(n1/(n0+n1) - jnp.dot(many_wb, pb)) - xi)
    return max(constrainta,constraintb)


def train_dro_fairconst(ds, data, eta=0., tv=0., learning_rate=1e-2, learning_rate_w=1e-2, learning_rate_lag=1e-1, nsteps=1000):
    """Train logistic regression classifier with demographic parity constraints
       and distributional robustness following https://arxiv.org/pdf/2002.09343.pdf"""
    X,y = ds.features(data)
    n, dim = X.shape
    
    mask01 = jnp.asarray(data['target']==1, dtype=bool) # mask if target=1 or 0
    X0,X1 = X[~mask01],X[mask01] # predict 0 on a and 1 on not-a
    
    maskab = jnp.asarray(data['protected']==1, dtype=bool) # mask if protected=1 or 0
    Xa,Xb = X[maskab],X[~maskab]

    na = maskab.sum()
    wa0 = jnp.ones(na)/na
    wb0 = jnp.ones(n-na)/(n-na)
     
    final_theta = None
    xi = 0.01 # start small
    while True:
      theta = cw_to_theta(jnp.zeros(dim),jnp.zeros(1)) # trivial classifier fulfills the constraints
      wa,wb = wa0,wb0
      lag = jnp.zeros(2)
      
      best_theta = cw_to_theta(jnp.zeros(dim),jnp.zeros(1))
      best_loss = np.inf
      for i in range(nsteps):
        theta_grads,wa_grads,wb_grads,lag_grads = jax.grad(cost_fair_const, argnums=(0,5,6,7))(theta, X0, X1, Xa, Xb, wa, wb, lag, xi)
        theta = theta - learning_rate*theta_grads
        
        # ascent step on Lagrange multipliers
        lag = lag + learning_rate_lag*lag_grads
        lag = jnp.maximum(lag, 0) # keep Lagrange multipliers positive
        
        # ascent step on weights wa,wb
        wa = wa + learning_rate_w*wa_grads
        wa = euclidean_proj_l1ball(wa-wa0, s=2*tv)+wa0
        wa = jnp.maximum(wa, 0) # keep in positive orthant
        
        wb = wb + learning_rate_w*wb_grads
        wb = euclidean_proj_l1ball(wb-wb0, s=2*tv)+wb0 # used in DRO code on github
        wb = jnp.maximum(wb, 0) # keep in positive orthant
        
        loss, constraints = cost_fair_const_ext(theta, X0, X1, Xa, Xb, wa, wb, xi)
        if (constraints<=1e-8).all() and (loss <= best_loss):
          ext_constraints = extended_fairness_constraints(theta, X0, X1, Xa, Xb, wa0, wb0, tv=tv, xi=xi)
          if ext_constraints <= 0:
            best_theta = theta.copy()
            best_loss = loss
      
      pred = predict(best_theta, X)
      if (pred==0).all() or (pred==1).all():
        xi *= 2 # if solution is degenerate, try again with more slack
      else:
        final_theta = best_theta
        break
    return final_theta


def estimate_disparity(ds, df_loss, df_fair, column='protected', value=0):
    X01,_ = ds.features(df_loss)
    Xab,_ = ds.features(df_fair)
    scaler = StandardScaler().fit(jnp.vstack([X01,Xab]))
    
    X01 = scaler.transform(X01)
    mask01 = jnp.asarray(df_loss[column]==value, dtype=bool) # mask if value=a or not
    X0,X1 = X01[~mask01],X01[mask01] # predict 0 on a and 1 on not-a
    
    Xab = scaler.transform(Xab)
    maskab = jnp.asarray(df_fair[column]==value, dtype=bool)
    Xa,Xb = Xab[maskab],Xab[~maskab]
    
    theta0 = train_sklearn_Xy(X01, jnp.asarray(mask01, dtype=float), lam=1) # initialize without fairness
    res = minimize(
        cost_fairreg,
        theta0,
        args=(X0, X1, Xa, Xb, 1.), # eta=1 for disparity estimate
        method="BFGS",
        jac=jax.grad(cost_fairreg),
        options={"gtol": 1e-4, "disp": False, "maxiter": 500}
    )
    disp = disparity(res.x, X0, X1, Xa, Xb)
    cost_disp = cost_fairreg_ext(res.x, X0, X1, Xa, Xb)
    return disp


@jax.jit
def cost_reg(theta):
    c, w = theta_to_cw(theta)
    reg = 0.5*jnp.dot(w, w)
    return reg

@jax.jit
def cost_TERM(theta,Xa,ya,Xb,yb,tau):
    na,nb = len(Xa),len(Xb)
    n = na+nb
    pa = predict_proba(theta, Xa)
    pb = predict_proba(theta, Xb)
    lossa = crossent(pa, ya)
    lossb = crossent(pb, yb)
    expoa = tau*lossa+jnp.log(na/n) # weighted by group size
    expob = tau*lossb+jnp.log(nb/n)
    
    loss_per_attribute = jnp.asarray([expoa,expob])
    tilted_loss = jax.scipy.special.logsumexp(loss_per_attribute)
    tilted_loss /= tau
    return tilted_loss

def cost_TERM_hierarchical(theta, data, lam, t, tau):
    loss_per_group = []
    n_total = sum(len(Xa)+len(Xb) for Xa,_,Xb,_  in data)

    for (Xa,ya,Xb,yb) in data:
        n = len(Xa)+len(Xb)
        cost = cost_TERM(theta,Xa,ya,Xb,yb,tau)
        loss_per_group.append( t*cost+jnp.log(n/n_total) )
    loss_per_group = jnp.asarray(loss_per_group)
    cost = jax.scipy.special.logsumexp(loss_per_group)
    cost /= t # value not needed for optimization, but its sign matters
    
    reg = cost_reg(theta)
    cost += lam*reg
    return cost

def train_TERM(ds, sources, lam=0, t=-2, tau=2, classifier="linear"):
    """Train linear classifier with hierarchical TERM loss (as suggested by TMLR Reviewer)"""
    assert classifier == 'linear', "TERM currently only works with a linear classifier"

    data = []
    for df in sources:
        Xa,ya = ds.features(df[df['protected']==0])
        Xb,yb = ds.features(df[df['protected']==1])
        data.append([Xa,ya,Xb,yb])

    dummyX,dummyY = ds.features(pd.concat(sources))
    theta0 = train_sklearn_Xy(dummyX, dummyY, lam=1)
    res = minimize(
        cost_TERM_hierarchical,
        theta0,
        args=(data, lam, t, tau),
        method="BFGS",
        jac=jax.grad(cost_TERM_hierarchical),
        options={"gtol": 1e-4, "disp": False, "maxiter": 500}
    )
    return jnp.asarray(res.x)


def resample_source(df):
    counts_a = df.groupby(['protected']).size()
    counts_y = df.groupby(['target']).size()
    counts_ay = df.groupby(['protected','target']).size()
    df_resampled = []
    for (a,y),df_ay in df.groupby(['protected','target']):
      n_ay = int(counts_a[a]*counts_y[y]/len(df))
      df_resampled.append(df_ay.sample(n_ay, replace=True))

    df_resampled = pd.concat(df_resampled)
    return df_resampled

def train_fairresample(ds, df, lam=0, classifier="linear"):
    df_resampled = resample_source(df)
    return train_classifier(ds, df_resampled, lam=lam, classifier=classifier)

def train_classifier(ds, df, lam=0, classifier="linear"):
    if classifier == "linear":
      return train_sklearn(ds, df, lam)
    else:
      return train_xgboost(ds, df)

def train_sklearn(ds, df, lam=0, classifier="linear"):
    X, y = ds.features(df)
    return train_sklearn_Xy(X, y, lam)

def train_sklearn_Xy(X, y, lam=0):
    if lam > 0:
      clf = LogisticRegression(C=1/lam, max_iter=500).fit(X, y)
    else:
      clf = LogisticRegression(penalty='none', max_iter=700).fit(X, y)
    theta = cw_to_theta(clf.intercept_[0],clf.coef_[0])
    return theta

def train_xgboost(ds, df, params=None):
    X, y = ds.features(df)
    return train_xgboost_Xy(X, y, params)

def train_xgboost_Xy(X, y, params=None, use_CV=True):
    if params is None:
      params = {'objective':'binary:logistic', 'eval_metric':'error', 'n_estimators':100, 'max_depth':5, 'use_label_encoder':False, 'n_jobs':4} 
    
      if use_CV:
        clf = XGBClassifier(**params)
            
        n_estimators = range(100, 501, 100)
        max_depth = [3,5,7,9]
        param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
        grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, verbose=0)
        grid_result = grid_search.fit(X,y)
        params = grid_result.best_params_    
    
    clf = XGBClassifier(**params)
    clf.fit(X,y)
    return clf


def evaluate(ds, df, theta):
    X, y = ds.features(df)
    _, y_pred, y_prob = predict_all(theta, X)
    auc = roc_auc_score(y, y_prob)
    acc = jnp.mean(y_pred == y)
    dpr = jnp.mean(y) # data positive rate 
    cpr = jnp.mean(y_pred) # classifier positive rate
    tnr,fpr,fnr,tpr = confusion_matrix(y, y_pred, normalize='true').ravel() # true/false negatives/positives
    return jnp.asarray([acc,auc,dpr,cpr,tpr,tnr])

def evaluate_xgboost(ds, df, clf):
    X, y = ds.features(df)
    _, y_pred, y_prob = predict_all_xgboost(clf, X)
    auc = roc_auc_score(y, y_prob)
    acc = jnp.mean(y_pred == y)
    dpr = jnp.mean(y) # data positive rate 
    cpr = jnp.mean(y_pred) # classifier positive rate
    tnr,fpr,fnr,tpr = confusion_matrix(y, y_pred, normalize='true').ravel() # true/false negatives/positives
    return jnp.asarray([acc,auc,dpr,cpr,tpr,tnr])

def evaluate_ensemble(ds, df, all_thetas, adv_mask=None, classifier="linear"):
    X, y = ds.features(df)
    
    if classifier == 'linear':
        all_y_prob = [predict_proba(theta, X) for theta in all_thetas]
    elif classifier == 'xgboost':
        all_y_prob = []
        for clf in all_thetas:
            _, _, y_prob = predict_all_xgboost(clf, X)
            all_y_prob.append(y_prob)
    else:
        print("Unknown classifier", classifier, file=sys.stderr)
        raise SystemExit
    
    N = len(all_y_prob) # assume odd for simplicity
    
    idx_selected = np.argsort(all_y_prob,axis=0)[N//2,:] # which classifier yields middle (median) score
    if adv_mask is None:
      adv_fraction = 0
    else:
      adv_fraction = adv_mask[idx_selected].mean()  # fraction of decisions based on adversary data
    
    y_prob = np.median(all_y_prob, axis=0)   # median scores
    y_pred = proba_to_pred(y_prob)
    
    auc = roc_auc_score(y, y_prob)
    acc = jnp.mean(y_pred == y)
    auc = roc_auc_score(y, y_prob)
    acc = jnp.mean(y_pred == y)
    dpr = jnp.mean(y) # data positive rate 
    cpr = jnp.mean(y_pred) # classifier positive rate
    tnr,fpr,fnr,tpr = confusion_matrix(y, y_pred, normalize='true').ravel() # true/false negatives/positives
    return jnp.asarray([acc,auc,dpr,cpr,tpr,tnr,adv_fraction])

def estimate_disbalance(ds, df1, df2, column, value):
    c1 = (df1[column]==value).mean() # fraction of A=a
    c2 = (df2[column]==value).mean()
    return jnp.abs(c1-c2)

def estimate_disb_matrix(ds, sources, column='protected',value=0):
    n_sources = len(sources)
    disb_matrix = np.zeros((n_sources,n_sources))
    for i in range(1, n_sources):
        for j in range(i):
          disb_matrix[i,j] = estimate_disbalance(ds, sources[i], sources[j], column, value)

    disb_matrix += disb_matrix.T
    return disb_matrix

def estimate_disp_matrix(ds, sources, column='protected',value=0):
    n_sources = len(sources)
    disp_matrix = np.zeros((n_sources,n_sources))
    for i in range(n_sources):
        for j in range(n_sources): # we do both directions because of the asymmetric relaxation
          if i==j:
            continue
          disp_matrix[i,j] = estimate_disparity(ds, sources[i], sources[j], column, value)

    return jnp.maximum(disp_matrix, disp_matrix.T)

def discrepancy(theta, X1, y1, X2, y2):
    p1 = predict(theta, X1)
    p2 = predict(theta, X2)
    disc = jnp.abs(jnp.mean(y1 != p1) - jnp.mean(y2 != p2))
    return disc

def estimate_discrepancy(ds, df1, df2, column='target'):
    X1,y1 = ds.features(df1)
    w1 = np.ones_like(y1)/len(y1)
    
    X2,y2 = ds.features(df2)
    w2 = np.ones_like(y2)/len(y2)

    X = jnp.vstack([X1,X2])
    y = jnp.asarray(jnp.concatenate([y1,1-y2]),dtype=float) # fit y on df1 and 1-y and df2
    w = jnp.concatenate([w1,w2])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf = LogisticRegression(penalty='none', max_iter=700).fit(X, y, sample_weight=w)
    
    X1 = scaler.transform(X1)
    X2 = scaler.transform(X2)
    
    theta = cw_to_theta(clf.intercept_[0],clf.coef_[0])
    disc = discrepancy(theta, X1, y1, X2, y2)
    return jnp.clip(disc, 0, 1)

def estimate_disc_matrix(ds, sources):
    n_sources = len(sources)
    disc_matrix = np.zeros((n_sources,n_sources))
    for i in range(1,n_sources):
        for j in range(i):
            disc_matrix[i,j] = estimate_discrepancy(ds, sources[i], sources[j])

    disc_matrix += disc_matrix.T
    return disc_matrix

def manipulate(ds, sources, adv_idx, strategy='flip#protected', value=None):
  all_strategies = ['flip#protected', 'flip#target', 'flip#protected#target', 'shuffle#protected', 'copy#protected#target', 'copy#target#protected', 'resample#1', 'randomanchor#0', 'randomanchor#1']
  strategy_list = [strategy]*len(sources) # per-source strategy. By default all the same strategy
  
  adv_sources = []
  for i,df in enumerate(sources):
    strategy = strategy_list[i].split('#')
    
    if strategy[0] == 'random':
      strategy = np.random.choice(all_strategies).split('#')  # pick random strategy

    if (strategy[0] == 'none') or (i not in adv_idx):
      adv_df = df
    elif strategy[0] == 'flip':
      cols = strategy[1:]
      adv_df = manipulate_dataset_flip_columns(df, cols)
    elif strategy[0] == 'copy':
      tgt_col, src_col = strategy[1],strategy[2]
      adv_df = manipulate_dataset_copy_column(df, tgt_col, src_col)
    elif strategy[0] == 'shuffle':
      col = strategy[1]
      adv_df = manipulate_dataset_shuffle_column(df, col)
    elif strategy[0] == 'resample':
      value = float(strategy[1])
      adv_df = create_dataset_downsample_column(df, sources, value)
      adv_df = create_dataset_upsample_column(adv_df, sources, 1-value)
    elif strategy[0] == 'randomanchor':
      try:
        value = float(strategy[1])
      except IndexError:
        value = 1
      adv_df = create_dataset_randomanchor(ds, df, sources, value)
    else:
      print("unsupported strategy", strategy)
      return adv_sources

    adv_sources.append(adv_df)
  
  return adv_sources

def create_dataset_randomanchor(ds, df, sources, value):
  """Random anchor attack following Alg2 in (Mehrabi et al, "Exacerbating 
     Algorithmic Bias through Fairness Attacks", AAAI 2021)"""

  df = df.copy(deep=True) # preserve original, as this here will be overwritten
  all_data = pd.concat(sources)
  data_a_minus = all_data[(all_data['protected']!=value) & (all_data['target']==0)] # 'a' for advantaged group
  data_a_plus = all_data[(all_data['protected']!=value) & (all_data['target']==1)]
  data_d_minus = all_data[(all_data['protected']==value) & (all_data['target']==0)] # 'd' for disadvantaged (=protected) group
  data_d_plus = all_data[(all_data['protected']==value) & (all_data['target']==1)]
  
  n_minus = (df['target']==0).sum()
  df_a_minus = df[(df['protected']!=value) & (df['target']==0)].sample(1) # pick one example
  feat_ex, _ = ds.features(df_a_minus)
  feat_all, _ = ds.features(data_a_plus)
  dist = np.sum((feat_all-feat_ex)**2,axis=1)
  dist_idx = np.argsort(dist)
  
  idx = []
  while len(idx)<n_minus:
    idx.extend(dist_idx[:n_minus]) # repeat points if not enough otherwise
    
  df_G_plus = data_a_plus.iloc[idx]
  
  n_plus = (df['target']==1).sum()
  df_d_plus = df[(df['protected']==value) & (df['target']==1)].sample(1) # pick one example
  feat_ex, _ = ds.features(df_d_plus)
  feat_all, _ = ds.features(data_d_minus)
  dist = np.sum((feat_all-feat_ex)**2,axis=1)
  dist_idx = np.argsort(dist)
  
  idx = []
  while len(idx)<n_plus:
    idx.extend(dist_idx[:n_plus]) # repeat points if not enough otherwise
  
  df_G_minus = data_d_minus.iloc[idx]

  df = pd.concat([df_G_plus,df_G_minus])
  return df

def create_dataset_upsample_column(df, sources, value):
  """Replace samples with protected==value and y=1 by samples with y=0"""

  df = df.copy(deep=True) # preserve original, as this here will be overwritten
  all_data = pd.concat(sources)
  all_data = all_data[(all_data['protected']==value) & (all_data['target']==0)]
  m = len(all_data)
  
  df_idx = np.flatnonzero( (df['protected']==value) & (df['target']==1) )

  k = len(df_idx)
  while k > m: # if too many rows to fill, use complete copies
    df.iloc[df_idx[:m]] = all_data.values
    df_idx = df_idx[m:]
    k -= m
  
  selected = all_data.sample(k) 
  df.iloc[df_idx] = selected.values # overwrite rows, need "values" to avoid index trouble
  return df

def create_dataset_downsample_column(df, sources, value):
  """Replace samples with protected==value and y=0 by samples with y=1"""

  df = df.copy(deep=True) # preserve original, as this here will be overwritten
  all_data = pd.concat(sources)
  all_data = all_data[(all_data['protected']==value) & (all_data['target']==1)]
  m = len(all_data)
  
  df_idx = np.flatnonzero( (df['protected']==value) & (df['target']==0) )

  k = len(df_idx)
  while k > m: # if too many rows to fill, use complete copies
    df.iloc[df_idx[:m]] = all_data.values
    df_idx = df_idx[m:]
    k -= m

  selected = all_data.sample(k)
  df.iloc[df_idx] = selected.values # overwritte rows, need "values" to avoid index trouble
  return df

def manipulate_dataset_flip_columns(df, columns):
    df = df.copy(deep=True)
    n = len(df)
    for col in columns: # can be as many as wanted
      values = df[col].unique()
      assert len(values) == 2, f"Error, more than 2 values in column {col}!"

      df.iloc[:n,df.columns.get_loc(col)].replace(values, values[::-1], inplace=True)
    return df

def manipulate_dataset_copy_column(df, column, src_col):
    n = len(df[column])
    df = df.copy(deep=True)
    df.iloc[:n,df.columns.get_loc(column)] = df.iloc[:n,df.columns.get_loc(src_col)].values
    return df

def manipulate_dataset_shuffle_column(df, column):
    n = len(df[column])
    idx = np.random.permutation(np.arange(n))
    df = df.copy(deep=True)
    df.iloc[:n,df.columns.get_loc(column)] = df.iloc[idx,df.columns.get_loc(column)].sample(frac=1).values
    return df

def postprocess_for_demographic_parity(ds, df, theta, group_by):
    linear = np.linspace(0,1,101)

    acc = jnp.zeros_like(linear)
    nr_curve = {} 
    for value,df_subset in df.groupby(group_by):
        X, y = ds.features(df_subset)
        n, _ = X.shape
        
        negative_rate = np.arange(1.,n+1.)/n 
        logits = predict_decision_function(theta, X)

        idx = jnp.argsort(logits)
        nr_curve[value] = np.interp(linear, negative_rate, logits[idx]) # make curve fixed length
        num_errors = jnp.cumsum(1-2*y[idx], dtype=float)+y.sum() # number of errors 
        acc += np.interp(linear, negative_rate, num_errors)

    idx_opt = np.argmax(acc)
    
    if args.onehotdrop == 'none':
      for value,df_subset in df.groupby(group_by):
        col_idx = ds.feature_names.index(f"{group_by}#{value}")
        theta = theta.at[col_idx].add(-nr_curve[value][idx_opt])
    else:
      col_idx = ds.feature_names.index("const")  # correct class 0 (where protected#0==0)
      theta = theta.at[col_idx].add(-nr_curve[0][idx_opt])
      col_idx = ds.feature_names.index(f"{group_by}#1")  # correct entry 1 by difference
      theta = theta.at[col_idx].add(nr_curve[0][idx_opt] - nr_curve[1][idx_opt])
    return theta


def eval_setting_ablation(ds, df, prefix, lam, eta_reg, eta_adv, extra=defaultdict(float), classifier="linear", outputformat="csv"):
  """Train and evaluate classifers for given setting and output result in tabular form:
     lam: L^2 regularization strength,  eta_reg/eta_adv: fairness trade-offs, extra: additional values to print"""
  #remark: for efficiency, we do ablation study only for regulation-based and preprocessing-based fairness
  #eval_classifier(ds, df, fairness='none', prefix=prefix, lam=lam, extra=extra['none'], classifier=classifier, outputformat=outputformat)
  if classifier == "linear":
    eval_classifier(ds, df, fairness='reg', prefix=prefix, lam=lam, eta=eta_reg, extra=extra['reg'], outputformat=outputformat)
    #  eval_classifier(ds, df, fairness='pp', prefix=prefix, lam=lam, extra=extra['pp'], outputformat=outputformat)
    #  eval_classifier(ds, df, fairness='adv', prefix=prefix, lam=lam, eta=eta_adv, extra=extra['adv'], outputformat=outputformat)
  else:
    eval_classifier(ds, df, fairness='resample', prefix=prefix, lam=lam, eta=eta_adv, extra=0, classifier=classifier, outputformat=outputformat)

def eval_setting(ds, df, prefix, lam, eta_reg, eta_adv, extra=defaultdict(float), classifier="linear", outputformat="csv"):
  """Train and evaluate classifers for given setting and output result in tabular form:
     lam: L^2 regularization strength,  eta_reg/eta_adv: fairness trade-offs, extra: additional values to print"""
  eval_classifier(ds, df, fairness='none', prefix=prefix, lam=lam, extra=extra['none'], classifier=classifier, outputformat=outputformat)
  if classifier == "linear":
    eval_classifier(ds, df, fairness='reg', prefix=prefix, lam=lam, eta=eta_reg, extra=extra['reg'], outputformat=outputformat)
    eval_classifier(ds, df, fairness='pp', prefix=prefix, lam=lam, extra=extra['pp'], outputformat=outputformat)
    eval_classifier(ds, df, fairness='adv', prefix=prefix, lam=lam, eta=eta_adv, extra=extra['adv'], outputformat=outputformat)
  eval_classifier(ds, df, fairness='resample', prefix=prefix, lam=lam, eta=eta_adv, extra=0, classifier=classifier, outputformat=outputformat)

def print_output(stage, prefix, e, extra, outputformat):
    assert outputformat == 'csv' or outputformat == 'json', "Unknown output format"
    if outputformat == 'csv':
        print_output_csv(stage=stage, prefix=prefix, e=e, extra=extra)
    else:
        print_output_json(stage=stage, prefix=prefix, e=e, extra=extra)

def print_output_csv(stage, prefix, e, extra=0):
    # CSV columns are: method,acc,auc,data_pr,cls_pr,tpr,tnr,extra
    print(f"{prefix}-{stage},{e[0]:5.3f},{e[1]:5.3f},{e[2]:5.3f},{e[3]:5.3f},{e[4]:5.3f},{e[5]:5.3f},{extra:5.3f}")

def print_output_json(stage, prefix, e, extra=0):
    if stage == 'all':
        # extract relevant components from prefix string. TODO: smarter format
        if 'fair' not in prefix:
            prefix += '-fair_none'

        method = prefix.split('-')[-2]
        fairness = prefix.split('_')[-1]
        print(f'{{ "method":"{method}", "fairness":"{fairness}", "acc":{e[0]:5.3f}, "auc":{e[1]:5.3f}, "data_pr":{e[2]:5.3f}, ', end="")
        print(f'"cls_pr":{e[3]:5.3f}, "tpr":{e[4]:5.3f}, "tnr":{e[5]:5.3f}, "extra":{extra:5.3f},', end="")
    else:
        print(f'  "acc-{stage}":{e[0]:5.3f}, "auc-{stage}":{e[1]:5.3f}, "data_pr-{stage}":{e[2]:5.3f}, ', end="")
        print(f'"cls_pr-{stage}":{e[3]:5.3f}, "tpr-{stage}":{e[4]:5.3f}, "tnr-{stage}":{e[5]:5.3f}, "extra-{stage}":{extra:5.3f}', end="")
        if stage == 'delta':
            print("}")
        else:
            print(", ", end="")

def eval_classifier(ds, df, df_test=None, lam=0, eta=None, fairness='none', group_by='protected', prefix='', extra=0, classifier="linear", outputformat="csv"): 
  if df_test == None:
    df_test = ds.df_test
  
  class_ratio = df_test[group_by].value_counts(normalize=True)
  class0_name, class0_ratio = class_ratio[[0]].index[0], class_ratio[[0]].values[0]

  if 'DRO' in prefix: # only compatible with fairness regularization
    prefix += '-fair_reg'
    theta = train_dro_fairconst(ds, df, eta=eta, tv=args.nadv/args.nsources)
  elif fairness == 'regularizer' or fairness == 'reg':
    prefix += '-fair_reg'
    theta = train_fairreg(ds, df, lam=lam, eta=eta)
  elif fairness == 'adversarial' or fairness == 'adv':
    prefix += '-fair_adv'
    theta = train_fairadv(ds, df, lam=lam, eta=eta)
  elif fairness == 'resample':
    prefix += '-fair_resample'
    theta = train_fairresample(ds, df, lam=lam, classifier=classifier) # linear or xgboost
    # theta can be clf or weight vector. future refactoring should make this consistent
  elif fairness == 'postprocessing' or fairness == 'pp':
    prefix += '-fair_pp'
    theta = train_sklearn(ds, df, lam=lam, classifier=classifier)
    theta = postprocess_for_demographic_parity(ds, df, theta, group_by)
  else: # fairness_type == "none"
    theta = train_classifier(ds, df, lam=lam, classifier=classifier) # linear or xgboost
  
  if classifier == "linear":
    e = evaluate(ds, df=df_test, theta=theta)
  elif classifier == "xgboost":
    e = evaluate_xgboost(ds, df=df_test, clf=theta)
    
  print_output(stage='all', prefix=prefix, e=e, extra=extra, outputformat=outputformat)
      
  evals = []
  for label,df_subset in df_test.groupby(group_by):
      if classifier == "linear":
        e = evaluate(ds, df=df_subset, theta=theta)
      elif classifier == "xgboost":
        e = evaluate_xgboost(ds, df=df_subset, clf=theta)

      print_output(stage=f'group-{label}', prefix=prefix, e=e, extra=class_ratio[label], outputformat=outputformat)
      evals.append(e)
  
  diff = evals[0]-evals[1] # we'll do the abs later if we want
  print_output(stage=f'delta', prefix=prefix, e=diff, extra=np.abs(diff[4:6]).max(), outputformat=outputformat)

  # UNUSED: evaluate also on trainset
  if False:
    e = evaluate(ds, df=df, theta=theta)
    print(f"trn-{prefix},{e[0]:5.3f},{e[1]:5.3f},{e[2]:5.3f},{e[3]:5.3f},{e[4]:5.3f},{e[5]:5.3f},{extra:5.3f}")

    evals = []
    for label,df_subset in df.groupby(group_by):
        e = evaluate(ds, df_subset, theta)
        print(f"trn-{prefix}-group-{label},{e[0]:5.3f},{e[1]:5.3f},{e[2]:5.3f},{e[3]:5.3f},{e[4]:5.3f},{e[5]:5.3f},{class_ratio[label]:5.3f}")
        evals.append(e)
    
    diff = evals[0]-evals[1] # we'll do the abs later if we want
    print(f"trn-{prefix}-delta,{diff[0]:5.3f},{diff[1]:5.3f},{diff[2]:5.3f},{diff[3]:5.3f},{diff[4]:5.3f},{diff[5]:5.3f},{np.abs(diff[4:6]).max():5.3f}")


def eval_setting_hierarchical(ds, sources, adv_mask, prefix, lam, t, tau, classifier="linear", outputformat="csv"): # extra values will be computed inside
  if classifier != "linear":
      return
  # hTERM only compatible with no fairness, postprocessing and preprocessing
  eval_hierarchical(ds, sources, adv_mask=adv_mask, fairness='none', prefix=prefix, lam=lam, t=t, tau=tau, classifier=classifier, outputformat=outputformat)
  eval_hierarchical(ds, sources, adv_mask=adv_mask, fairness='pp', prefix=prefix, lam=lam, classifier=classifier, outputformat=outputformat)
  eval_hierarchical(ds, sources, adv_mask=adv_mask, fairness='resample', prefix=prefix, lam=lam, classifier=classifier, outputformat=outputformat)


def eval_hierarchical(ds, sources, adv_mask=None, lam=0, t=-2, tau=2, fairness='none', group_by='protected', prefix='', classifier="linear", extra=0, outputformat="csv"):
  """Train hierarchical model with TERM loss across all sources following Eq.(4) in https://arxiv.org/abs/2007.01162
     lam: L^2 regularization strength, t/tau: tilts for outer/inner loss, extra: additional values to print"""

  if fairness == 'preprocessing' or fairness == 'resample':
    prefix += '-fair_resample'
    sources = [resample_source(df) for df in sources]

  theta = train_TERM(ds, sources, lam=lam, t=t, tau=tau, classifier=classifier)

  if fairness == 'postprocessing' or fairness == 'pp':
    prefix += '-fair_pp'
    df = pd.concat(sources)
    theta = postprocess_for_demographic_parity(ds, df, theta, group_by)
  
  # evaluate the resulting classifier
  df_test = ds.df_test
  e = evaluate(ds, df=df_test, theta=theta)
  print_output(stage='all', prefix=prefix, e=e, extra=extra, outputformat=outputformat)

  class_ratio = df_test[group_by].value_counts(normalize=True)
  evals = []
  for label,df_subset in df_test.groupby(group_by):
      e = evaluate(ds, df=df_subset, theta=theta)
      evals.append(e)
      print_output(stage=f'group-{label}', prefix=prefix, e=e, extra=class_ratio[label], outputformat=outputformat)
  
  diff = evals[0]-evals[1] # we'll do the abs later if we want
  print_output(stage='delta', prefix=prefix, e=diff, extra=np.abs(diff[4:6]).max(), outputformat=outputformat)



def eval_setting_ensemble(ds, sources, adv_mask, prefix, lam, eta_reg, eta_adv, classifier="linear", outputformat="csv"): # extra values will be computed inside
  eval_ensemble(ds, sources, adv_mask=adv_mask, fairness='none', prefix=prefix, lam=lam, classifier=classifier, outputformat=outputformat)
  if classifier == "linear":
    eval_ensemble(ds, sources, adv_mask=adv_mask, fairness='reg', prefix=prefix, lam=lam, eta=eta_reg, outputformat=outputformat)
    eval_ensemble(ds, sources, adv_mask=adv_mask, fairness='pp', prefix=prefix, lam=lam, outputformat=outputformat)
    eval_ensemble(ds, sources, adv_mask=adv_mask, fairness='adv', prefix=prefix, lam=lam, eta=eta_adv, outputformat=outputformat)
  eval_ensemble(ds, sources, adv_mask=adv_mask, fairness='resample', prefix=prefix, lam=lam, classifier=classifier, outputformat=outputformat)


def eval_ensemble(ds, sources, adv_mask=None, lam=0, eta=None, fairness='none', group_by='protected', prefix='', classifier="linear", outputformat="csv"):
  """Train and evaluate ensemble classifer for given setting and output result in tabular form:
     lam: L^2 regularization strength,  eta_reg/eta_adv: fairness trade-offs, extra: additional values to print"""
  df_test = ds.df_test
  
  if fairness == 'regularizer' or fairness == 'reg':
    prefix += '-fair_reg'
    all_thetas = [train_fairreg(ds, df, lam=lam, eta=eta) for df in sources]
  elif fairness == 'adversarial' or fairness == 'adv':
    prefix += '-fair_adv'
    all_thetas = [train_fairadv(ds, df, lam=lam, eta=eta) for df in sources]
  elif fairness == 'resample':
    prefix += '-fair_resample'
    all_thetas = [train_fairresample(ds, df, lam=lam, classifier=classifier) for df in sources]
  elif fairness == 'postprocessing' or fairness == 'pp':
    all_thetas = [train_classifier(ds, df, lam=lam) for df in sources]
  else: # fairness == 'none'
    all_thetas = [train_classifier(ds, df, lam=lam, classifier=classifier) for df in sources]
  
  if fairness == 'postprocessing' or fairness == 'pp':
    prefix += '-fair_pp'
    all_thetas = [postprocess_for_demographic_parity(ds, df, theta, group_by) for theta,df in zip(all_thetas,sources)]
    
  e = evaluate_ensemble(ds, df_test, all_thetas, adv_mask=adv_mask, classifier=classifier)
  print_output(stage='all', prefix=prefix, e=e, extra=e[6], outputformat=outputformat)

  class_ratio = df_test[group_by].value_counts(normalize=True)
  evals = []
  for label,df_subset in df_test.groupby(group_by):
      e = evaluate_ensemble(ds, df_subset, all_thetas, classifier=classifier)
      print_output(stage=f'group-{label}', prefix=prefix, e=e, extra=class_ratio[label], outputformat=outputformat)
      evals.append(e)
  
  diff = evals[0]-evals[1]
  print_output(stage='delta', prefix=prefix, e=diff, extra=np.abs(diff[4:6]).max(), outputformat=outputformat)
  return 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', '-d', type=str, default="adult", help="Dataset", choices=['adult', 'compas', 'germancredit', 'drugs', 'folktables'])
    parser.add_argument('--classifier', '-c', type=str, default='linear', choices=['linear','xgboost'], help="Classifier type")
    parser.add_argument('--nsources', '-n', type=int, default=5, help="Number of sources (N)")
    parser.add_argument('--nclean', '-K', type=int, default=None, help="Number of unperturbed sources to assume (default: N/2+1 )")
    parser.add_argument('--nadv', '-N', type=int, default=None, help="Number of perturbed sources (default: (N-1)/2 ) )")
    parser.add_argument('--adversary', '-a', type=str, default='none', help="Adversary type")
    parser.add_argument('--eta', '-e', type=float, default=0.5, help="Accuracy/fairness tradeoff (default: 0.5)")
    parser.add_argument('--t', '-t', type=float, default=-2, help="Outer tilting parameter for hierarchical TERM (default: -2)")
    parser.add_argument('--tau', '-T', type=float, default=2, help="Inner tilting parameter for hierarchical TERMs (default: 2)")
    parser.add_argument('--lambda', '-l', type=float, dest='lam', default=0., help="Regularization strength (default: no regularization)")
    parser.add_argument('--format', '-F', type=str, default="csv", choices=['csv','json'], help="Output format for evaluation results (default: csv)")
    parser.add_argument('--onehotdrop', type=str, default="first", help="Compatify features (='first') or not ('none')")
    parser.add_argument('--beta', '-b', type=float, default=None, help="Bound on number of clean sources for algorithm (default: 1/2+1/2N")
    parser.add_argument('--aggregation', type=str, default="sum", choices=['max','sum'], help="Aggregation method for D-scores (default: sum)")
    parser.add_argument('--do_FLEA_ablation', action="store_true", help="Run in FLEA ablation study mode")
    
    parser.add_argument('--do_original', action="store_true", help="Run on original clean data")
    parser.add_argument('--do_clean', action="store_true", help="Run oracle")
    parser.add_argument('--do_unrelible', action="store_true", help="Run on all (unreliable) data")
    parser.add_argument('--do_FLEA', action="store_true", help="Run FLEA")
    parser.add_argument('--do_ensemble', action="store_true", help="Run robust ensemble")
    parser.add_argument('--do_DRO', action="store_true", help="Run DRO")
    parser.add_argument('--do_TERM', action="store_true", help="Run TERM")
    parser.add_argument('--do_konstantinov', action="store_true", help="Run [Konstantinov et al]")

    args = parser.parse_args()
    if args.nclean and args.nadv and (args.nclean+args.navd != args.nsources):
        print("Illegal combination of nclean/nadv/nsources")
        raise SystemExit
    elif args.nclean is None and args.nadv is not None:
        args.nclean = args.nsources - args.nadv
    elif args.nclean is not None and args.nadv is None:
        args.nadv = args.nsources - args.nclean
    if args.nclean is None:
      args.nclean = args.nsources//2+1
    if args.nadv is None:
      args.nadv = (args.nsources-1)//2

    if args.do_original or args.do_clean or args.do_unrelible or args.do_FLEA or args.do_ensemble or args.do_DRO or args.do_TERM or args.do_konstantinov:
        args.do_all = False
    else:
        args.do_all = True # without argument: do all

    return args

def idx_to_mask(idx, size):
  mask = np.zeros(size, dtype=bool)
  mask[idx] = True
  return mask

def mask_to_idx(mask ):
  return np.where(mask)[0]

def FLEA(ds, sources, n_preserve=None, adv_mask=None, use_disb=True, use_disc=True, use_disp=True, aggregation='sum'): # adv_mask is only used to collect statistics
    n_sources = len(sources)
    if adv_mask is None:
      adv_mask = np.zeros(n_sources, bool)
    n_adv = adv_mask.sum()
    if n_preserve is None:
      n_preserve = n_sources-n_adv
    
    stats = {} # store some statistics for later printing

    if use_disb:
      # compute disbalance
      disb_matrix = estimate_disb_matrix(ds, sources)
      disb_q_score = np.partition(disb_matrix, kth=n_preserve-1, axis=1)[:,n_preserve-1]
      disb_select_idx = np.argsort(disb_q_score)[:n_preserve]
      stats['disb'] = adv_mask[disb_select_idx].mean()
    else:
      disb_matrix = np.zeros((n_sources, n_sources), np.float32)
      stats['disb'] = 0.
    
    if use_disc:
      # compute discrepancy
      disc_matrix = estimate_disc_matrix(ds, sources)
      disc_q_score = np.partition(disc_matrix, kth=n_preserve-1, axis=1)[:,n_preserve-1]
      disc_select_idx = np.argsort(disc_q_score)[:n_preserve]
      stats['disc'] = adv_mask[disc_select_idx].mean()
    else:
      disc_matrix = np.zeros((n_sources, n_sources), np.float32)
      stats['disc'] = 0.
    
    if use_disp:
      # compute disparity
      disp_matrix = estimate_disp_matrix(ds, sources)
      disp_q_score = np.partition(disp_matrix, kth=n_preserve-1, axis=1)[:,n_preserve-1]
      disp_select_idx = np.argsort(disp_q_score)[:n_preserve]
      stats['disp'] = adv_mask[disp_select_idx].mean()
    else:
      disp_matrix = np.zeros((n_sources, n_sources), np.float32)
      stats['disp'] = 0.
    
    # D-score aggregates the values of disparity, discrepancy and disbalance
    if aggregation == 'max':
        D_matrix = np.maximum.reduce([disc_matrix, disp_matrix, disb_matrix])
    else: # aggregation == 'sum'
        D_matrix = np.add.reduce([disc_matrix, disp_matrix, disb_matrix])

    q_scores = np.partition(D_matrix, kth=n_preserve-1, axis=1)[:,n_preserve-1]
    FLEA_select_idx = np.argsort(q_scores)[:n_preserve]
    stats['FLEA'] = adv_mask[FLEA_select_idx].mean()
    
    return FLEA_select_idx, stats

def estimate_disc_thresholds(ds, sources, delta=0.1):
    n_sources = len(sources)
    disc_threshold_matrix = np.eye(n_sources)
    data_shapes = [ds.features(s)[0].shape for s in sources]
    for i in range(1,n_sources):
        mi,dim = data_shapes[i]
        for j in range(i):
          mj,_ = data_shapes[j]
          m = (mi+mj)/2
          # threshold based on tighest bound we could find (from Mohri et al, "Foundation of ML")
          thr_term1 = 8*dim*(1+np.log(2*m/dim))
          thr_term2 = 8*np.log(8*n_sources/delta)
          thr = ((thr_term1+thr_term2)/m)**0.5
          disc_threshold_matrix[i,j] = thr

    disc_threshold_matrix += disc_threshold_matrix.T
    return disc_threshold_matrix
  
def konstantinov_filter(ds, sources, adv_mask=None):
    n_sources = len(sources)
    if adv_mask is None:
      adv_mask = np.zeros(n_sources, bool)
    n_adv = adv_mask.sum()
    
    stats = {} # store some statistics to be printed out later

    # compute discrepancy
    disc_matrix = estimate_disc_matrix(ds, sources)
    # compute threshold for reliable exclusion (from [Konstantinov, ICML 2020])
    disc_matrix_thresholds = estimate_disc_thresholds(ds, sources)
    
    row_scores = (disc_matrix<=disc_matrix_thresholds).sum(axis=1)
    select_mask = (row_scores>n_adv)
    if not select_mask.any(): # if no source at all is selected, fall back to everything
      select_mask = ~select_mask
    select_idx = mask_to_idx(select_mask)
    stats['konstantinov'] = adv_mask[select_idx].mean()
    return select_idx, stats

def main():
  global args
  args = parse_arguments()
  np.random.seed(args.seed)
  
  if args.do_FLEA_ablation:
    do_original = False
    do_clean = False
    do_unrelible = False
    do_FLEA = False # FLEA is part of ablation anyway
    do_ensemble = False
    do_DRO = False
    do_TERM = False
    do_konstantinov=False
  elif args.do_all:
    do_original = True
    do_clean = True
    do_unrelible = True
    do_FLEA = True
    do_ensemble = True
    do_DRO = True
    do_TERM = True
    do_konstantinov = True
  else:
    do_original = args.do_original
    do_clean = args.do_clean
    do_unrelible = args.do_unrelible
    do_FLEA = args.do_FLEA
    do_ensemble = args.do_ensemble
    do_DRO = args.do_DRO
    do_TERM = args.do_TERM
    do_konstantinov = args.do_konstantinov
      
  # load and preprocess data
  ds = DatasetInfo(args)
  ds.train_test_split(test_size=0.2, random_state=args.seed) # always 80%-20% for simplicity
  n_sources,n_clean,n_adv = args.nsources,args.nclean,args.nadv
  sources = ds.make_sources(n_sources)

  # header for CSV output
  if args.format == 'csv':
    print("method,acc,auc,data_pr,cls_pr,tpr,tnr,extra")

  # train/eval on original iid data (no manipulations)
  if do_original:
    df = pd.concat(sources)
    eval_setting(ds, df=df, prefix='original', lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, classifier=args.classifier, outputformat=args.format)

  # fix random subset of adversarial sources 
  rng = np.random.default_rng(seed=args.seed)
  adv_indices = rng.choice(range(n_sources), size=n_adv, replace=False)
  adv_mask = idx_to_mask(adv_indices, size=n_sources)
  clean_idx = mask_to_idx(~adv_mask)
  
  # train/eval union of all clean sources
  if do_clean:
    df = pd.concat([sources[i] for i in clean_idx])
    eval_setting(ds, df=df, prefix='adv-clean', lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, classifier=args.classifier, outputformat=args.format)

  unreliable_sources = manipulate(ds, sources, adv_idx=adv_indices, strategy=args.adversary)
  
  # train/eval union of all sources, including manipulated (no protection)
  if do_unrelible:
    df = pd.concat(unreliable_sources)
    prefix = f'adv-all'
    eval_setting(ds, df=df, prefix=prefix, lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, classifier=args.classifier, outputformat=args.format)

  
  # train/eval [konstantinov]
  if do_konstantinov:
    prefix = f'adv-konstantinov'
    select_idx, stats = konstantinov_filter(ds, unreliable_sources, adv_mask=adv_mask)
    df = pd.concat([unreliable_sources[i] for i in select_idx])
    extra = {'none':stats['konstantinov'], 'reg':0, 'pp':0, 'adv':0, 'resample':0} # where to print which value
    eval_setting(ds, df=df, prefix=prefix, lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, extra=extra, classifier=args.classifier, outputformat=args.format)

  # train/eval FLEA
  if do_FLEA:
    prefix = f'adv-selected'
    if args.beta is not None:
      n_preserve = int(math.ceil(args.beta*n_sources))
    else:
      n_preserve = n_clean
    select_idx, stats = FLEA(ds, unreliable_sources, n_preserve=n_preserve, adv_mask=adv_mask, use_disb=True, use_disc=True, use_disp=True, aggregation=args.aggregation)
    df = pd.concat([unreliable_sources[i] for i in select_idx])
    extra = {'none':stats['FLEA'], 'reg':stats['disc'], 'pp':stats['disp'], 'adv':stats['disb']} # where to print which value
    eval_setting(ds, df=df, prefix=prefix, lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, extra=extra, classifier=args.classifier, outputformat=args.format)
  
  if args.do_FLEA_ablation:
    if args.beta is not None:
      n_preserve = int(math.ceil(args.beta*n_sources))
    else:
      n_preserve = n_clean
    for use_disb in [True,False]:
      for use_disc in [True,False]:
        for use_disp in [True,False]:
          prefix = f'adv-FLEAablation{int(use_disb)}{int(use_disc)}{int(use_disp)}'
          select_idx, stats = FLEA(ds, unreliable_sources, n_preserve=n_preserve, adv_mask=adv_mask, use_disb=use_disb, use_disc=use_disc, use_disp=use_disp, aggregation=args.aggregation)
          df = pd.concat([unreliable_sources[i] for i in select_idx])
          extra = {'none':stats['FLEA'], 'reg':stats['disc'], 'pp':stats['disp'], 'adv':stats['disb']} # where to print which value
          eval_setting_ablation(ds, df=df, prefix=prefix, lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, extra=extra, outputformat=args.format)

  # train/eval ensemble classifier using median of scores
  if do_ensemble:
    prefix = f'adv-voting'
    eval_setting_ensemble(ds, unreliable_sources, adv_mask=adv_mask, prefix=prefix, lam=args.lam, eta_reg=args.eta, eta_adv=args.eta, classifier=args.classifier, outputformat=args.format)

  # train/eval hierarchical tilted ERM 
  if do_TERM:
    prefix = f'adv-TERM'
    eval_setting_hierarchical(ds, unreliable_sources, adv_mask=adv_mask, prefix=prefix, lam=args.lam, t=args.t, tau=args.tau, classifier=args.classifier, outputformat=args.format)

  if do_DRO and args.classifier=="linear":
    df = pd.concat(unreliable_sources)
    prefix = f'adv-DRO'
    eval_classifier(ds, df=df, fairness='reg', prefix=prefix, lam=args.lam, eta=args.eta, outputformat=args.format)

if __name__ == "__main__":
  main()
