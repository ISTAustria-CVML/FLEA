#!/usr/bin/env python3
# coding: utf-8
# Licensed under the terms of the MIT license, see LICENSE.md
# make barplots of FLEA results

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
try:
  from tqdm import tqdm
except ImportError:
  tqdm = lambda x: x
sns.set_theme(style="whitegrid")

figsize=(16, 3.6)

results_path = './results/'

extra='' 
#extra='-cxgboost' # different filenames for results of non-linear classifier

if extra == '-cxgboost': 
    all_fairalgo = ['none','resample']   # only some combinations possible for nonlinear classifier 
    all_methods = {'none':['voting','konstantinov','FLEA'], 
                   'resample':['voting','konstantinov','FLEA']}
else:
    all_fairalgo = ['none','reg','pp','adv', 'resample']
    all_methods = { 'none':['voting','konstantinov','FLEA'],
                    'reg':['voting','DRO','konstantinov','FLEA'],
                    'pp':['voting','TERM','konstantinov','FLEA'],
                    'adv': ['voting','konstantinov','FLEA'],
                    'resample':['voting','TERM','konstantinov','FLEA']}

filepattern = results_path+'{dataset}-n{nsources}-N{nadv}-a{adversary}'+extra+'-s{seed}.txt'
imagepattern = results_path+'/suppl-{dataset}-n{nsources}-N{nadv}-{classifier}'+extra+'{suffix}'

all_nsources = {'adult':[(5,2)], 'compas':[(5,2)], 'drugs':[(5,2)], 'germancredit':[(5,2)], 'folktables':[(51,5),(51,10),(51,15),(51,20),(51,25)]} 
if extra == '-cxgboost': 
    del all_nsources['folktables']
    
all_adversaries = ['none', 'flip#protected',  'flip#target', 'flip#protected#target', 'shuffle#protected','copy#protected#target','copy#target#protected','resample#1', 'random', 'randomanchor#0', 'randomanchor#1'] 
adversaries_label = {'none': 'ID', 'flip#protected':'FP', 'flip#target':'FL', 'flip#protected#target':'FB', 'shuffle#protected':'SP', 'copy#protected#target':'OP', 'copy#target#protected':'OL','resample#1':'RP' , 'random':'RND', 'randomanchor#0':'RA0', 'randomanchor#1':'RA1'}

all_seeds = range(0,10)

def demographic_parity(data, pattern): 
  return data[data.index.str.contains(f"{pattern}-delta")]['cls_pr'].abs()

def accuracy(data, pattern):
  return data[data.index.str.contains(f"{pattern}-all")]['acc']

def mean_std(x):
  return x.mean(),x.std()

def do_plot(data, dataset, algo, n, nadv):
  if algo == 'none':
    suffix = ''
  else:
    suffix = f'-fair_{algo}'
  
  acc, dp = {}, {}
  adv = 'none' # doesn't matter which one for clean data
  
  acc['adv-all'] = [accuracy(data[adv], pattern=f'adv-clean')]
  dp['adv-all'] = [1-demographic_parity(data[adv], pattern=f'adv-clean')]
  
  acc['adv-all-fair'] = [accuracy(data[adv], pattern=f'adv-clean{suffix}')]
  dp['adv-all-fair'] = [1-demographic_parity(data[adv], pattern=f'adv-clean{suffix}')]
  
  if 'FLEA' in all_methods[algo]:
      acc['adv-FLEA-fair'] = [accuracy(data[adv], pattern=f'adv-clean{suffix}')]
      dp['adv-FLEA-fair'] = [1-demographic_parity(data[adv], pattern=f'adv-clean{suffix}')]

  if 'TERM' in all_methods[algo]:
      acc['adv-TERM-fair'] = [accuracy(data[adv], pattern=f'adv-TERM{suffix}')]
      dp['adv-TERM-fair'] = [1-demographic_parity(data[adv], pattern=f'adv-TERM{suffix}')]

  if 'DRO' in all_methods[algo]:
      acc['adv-DRO-fair'] = [accuracy(data[adv], pattern=f'adv-clean{suffix}')]
      dp['adv-DRO-fair'] = [1-demographic_parity(data[adv], pattern=f'adv-clean{suffix}')]

  if 'voting' in all_methods[algo]:
      acc['adv-voting-fair'] = [accuracy(data[adv], pattern=f'adv-clean{suffix}')]
      dp['adv-voting-fair'] = [1-demographic_parity(data[adv], pattern=f'adv-clean{suffix}')]
  
  if 'konstantinov' in all_methods[algo]:
      acc['adv-konstantinov-fair'] = [accuracy(data[adv], pattern=f'adv-clean{suffix}')]
      dp['adv-konstantinov-fair'] = [1-demographic_parity(data[adv], pattern=f'adv-clean{suffix}')]
  
  for adv in all_adversaries:
    acc['adv-all'].append( accuracy(data[adv], pattern=f'adv-all') )
    dp['adv-all'].append( 1-demographic_parity(data[adv], pattern=f'adv-all') )
    acc['adv-all-fair'].append( accuracy(data[adv], pattern=f'adv-all{suffix}') )
    dp['adv-all-fair'].append( 1-demographic_parity(data[adv], pattern=f'adv-all{suffix}') )
    if 'FLEA' in all_methods[algo]:
        acc['adv-FLEA-fair'].append( accuracy(data[adv], pattern=f'adv-selected{suffix}') )
        dp['adv-FLEA-fair'].append( 1-demographic_parity(data[adv], pattern=f'adv-selected{suffix}') )
    if 'TERM' in all_methods[algo]:
        acc['adv-TERM-fair'].append( accuracy(data[adv], pattern=f'adv-TERM{suffix}') )
        dp['adv-TERM-fair'].append( 1-demographic_parity(data[adv], pattern=f'adv-TERM{suffix}') )
    if 'DRO' in all_methods[algo]:
        acc['adv-DRO-fair'].append( accuracy(data[adv], pattern=f'adv-DRO{suffix}') )
        dp['adv-DRO-fair'].append( 1-demographic_parity(data[adv], pattern=f'adv-DRO{suffix}') )
    if 'voting' in all_methods[algo]:
        acc['adv-voting-fair'].append( accuracy(data[adv], pattern=f'adv-voting{suffix}') )
        dp['adv-voting-fair'].append( 1-demographic_parity(data[adv], pattern=f'adv-voting{suffix}') )
    if 'konstantinov' in all_methods[algo]:
        acc['adv-konstantinov-fair'].append( accuracy(data[adv], pattern=f'adv-konstantinov{suffix}') )
        dp['adv-konstantinov-fair'].append( 1-demographic_parity(data[adv], pattern=f'adv-konstantinov{suffix}') )

  fig, axes = plt.subplots(2, len(all_methods[algo])+1, figsize=figsize)
  col=0
  
  sns.barplot(data=acc['adv-all-fair'], ax=axes[0,col])
  col +=1 
  if 'voting' in all_methods[algo]:
    sns.barplot(data=acc['adv-voting-fair'], ax=axes[0,col])
    col +=1 
  if 'DRO' in all_methods[algo]:
    sns.barplot(data=acc['adv-DRO-fair'], ax=axes[0,col])
    col +=1 
  if 'TERM' in all_methods[algo]:
    sns.barplot(data=acc['adv-TERM-fair'], ax=axes[0,col])
    col +=1 
  if 'konstantinov' in all_methods[algo]:
    sns.barplot(data=acc['adv-konstantinov-fair'], ax=axes[0,col])
    col +=1 
  if 'FLEA' in all_methods[algo]:
    sns.barplot(data=acc['adv-FLEA-fair'], ax=axes[0,col])
    col +=1 

  print(f"% all clean {' '.join(all_methods[algo])}")

  print(f"{dataset}", end="")
  acc_clean = 100*np.asarray(acc['adv-all-fair'][0]) # first entry is oracle
  m_clean,s_clean = mean_std(acc_clean)
  print("& ${:.1f}_{{\pm {:.1f}}}$".format(m_clean,s_clean), end="")
  acc_ref = 100*np.asarray(acc['adv-all-fair'][1:]) #   n_of_advmethods x n_seeds
  m,s = mean_std(acc_ref.min(axis=0)) # min across adv, mean/std across 
  print("& ${:.1f}_{{\pm {:.1f}}}$".format(m,s), end="")

  mean_of_minacc,std_of_minacc={},{}
  m_ref,s_ref = 0,0
  for method in all_methods[algo]:
    acc_matrix = 100*np.asarray(acc[f'adv-{method}-fair'][1:]) #   n_of_advmethods  x n_seeds
    mean_of_minacc[method],std_of_minacc[method] = mean_std(acc_matrix.min(axis=0))
    if mean_of_minacc[method] > m_ref:
        m_ref,s_ref = mean_of_minacc[method],std_of_minacc[method]
  
  for method in all_methods[algo]:
    m,s = mean_of_minacc[method],std_of_minacc[method]
    print("& ${:.1f}_{{\pm {:.1f}}}$".format(m, s), end="")
  print("\\\\")
  
  #sns.barplot(data=dp['adv-all'], ax=axes[1,0])
  col = 0
  sns.barplot(data=dp['adv-all-fair'], ax=axes[1,col])
  col += 1
  if 'voting' in all_methods[algo]:
    sns.barplot(data=dp['adv-voting-fair'], ax=axes[1,col])
    col += 1
  if 'DRO' in all_methods[algo]:
    sns.barplot(data=dp['adv-DRO-fair'], ax=axes[1,col])
    col += 1
  if 'TERM' in all_methods[algo]:
    sns.barplot(data=dp['adv-TERM-fair'], ax=axes[1,col])
    col += 1
  if 'konstantinov' in all_methods[algo]:
    sns.barplot(data=dp['adv-konstantinov-fair'], ax=axes[1,col])
    col += 1
  if 'FLEA' in all_methods[algo]:
    sns.barplot(data=dp['adv-FLEA-fair'], ax=axes[1,col])
    col += 1

  print(f"{algo}(N={n} K={n-nadv})", end="")
  dp_clean = 100*np.asarray(dp['adv-all-fair'][0]) # first entry is oracle
  m_clean,s_clean = mean_std(dp_clean)
  print("& ${:.1f}_{{\pm {:.1f}}}$".format(m_clean,s_clean), end="")
  dp_ref = 100*np.asarray(dp['adv-all-fair'][1:]) # all adversaries, no protection
  m,s = mean_std(dp_ref.min(axis=0))
  print("& ${:.1f}_{{\pm {:.1f}}}$".format(m,s), end="")
  
  mean_of_mindp,std_of_mindp={},{}
  m_ref,s_ref = 0,0
  for method in all_methods[algo]:
    dp_matrix = 100*np.asarray(dp[f'adv-{method}-fair'][1:]) #   n_of_advmethods  x n_seeds
    mean_of_mindp[method],std_of_mindp[method] = mean_std(dp_matrix.min(axis=0))
    if mean_of_mindp[method] > m_ref:
        m_ref,s_ref = mean_of_mindp[method],std_of_mindp[method]
  
  for method in all_methods[algo]:
    m,s = mean_of_mindp[method],std_of_mindp[method]
    print("& ${:.1f}_{{\pm {:.1f}}}$".format(m, s), end="")

  print("\\\\\hline")
  
  for ax in axes.flatten():
    ax.axvline(x=0.5, color='k', linewidth=1, linestyle='--')
  for ax in axes[0]:
    ax.set(ylim=(0.5, 0.9))
    ax.set_xticklabels(labels=[]) # ['clean']+
  for ax in axes[1]: 
    ax.set(ylim=(0, 1.1))
  
  col=0
  if algo == 'none':
    axes[0,col].set_title('fairness-unaware training')
  else:
    axes[0,col].set_title('ordinary fair training')
  col+=1
  
  if 'voting' in all_methods[algo]:
    axes[0,col].set_title('robust ensemble')
    col+=1
  if 'DRO' in all_methods[algo]:
    axes[0,col].set_title('DRO [Wang et al, NeurIPS 2020]')
    col+=1
  if 'TERM' in all_methods[algo]:
    axes[0,col].set_title('hTERM [Li et al, ICLR 2021]')
    col+=1
  if 'konstantinov' in all_methods[algo]:
    axes[0,col].set_title('[Konstantinov et al, ICML 2020]')
    col+=1
  if 'FLEA' in all_methods[algo]:
    axes[0,col].set_title('FLEA (proposed)')
    col+=1

  for ax in axes[1,:]:
    ax.set_xticklabels(labels=['oracle']+[adversaries_label[a] for a in all_adversaries], rotation=90) # ['clean']+
  plt.subplots_adjust(bottom=0.3)
  
  axes[0,0].set_ylabel("accuracy")
  axes[1,0].set_ylabel("fairness")
  plt.tight_layout()

def main():  
  all_datasets = sys.argv[1:]
  if all_datasets == [] or all_datasets == ["all"]:
    all_datasets = all_nsources.keys()
  
  for fairalgo in all_fairalgo:
    print("\\begin{table} % ",  fairalgo)
    for dataset in all_datasets:
      print("\\begin{subfigure}[b]{.95\\textwidth}\\centering")
      print("\\caption{",dataset,"}")
      print("\\begin{tabular}{ccccccc}")
      for (n,nadv) in all_nsources[dataset]:
        print(f"Dataset {dataset} N={n} nadv={nadv}", file=sys.stderr)
        data = defaultdict(list)
        for adv in tqdm(all_adversaries):
          for seed in all_seeds:
            filename = filepattern.format(dataset=dataset, nsources=n, adversary=adv, nadv=nadv, seed=seed)
            try:
              data[adv].append(pd.read_csv(filename, sep=',', index_col = 'method'))
            except pd.errors.EmptyDataError:
              print("Empty file?", filename, file=sys.stderr)
          data[adv] = pd.concat(data[adv], axis=0) # 10 seeds in 1 dataframe
        
        print("fairalgo=",fairalgo, file=sys.stderr)
        do_plot(data, dataset, fairalgo, n, nadv)
        
        imagefile = imagepattern.format(dataset=dataset, nsources=n, classifier=f'fair-{fairalgo}', nadv=nadv, suffix='.svg')
        print("Saving to", imagefile, file=sys.stderr)
        plt.savefig(imagefile)
        imagefile = imagepattern.format(dataset=dataset, nsources=n, classifier=f'fair-{fairalgo}', nadv=nadv, suffix='.pdf')
        print("Also saving to", imagefile, file=sys.stderr)
        plt.savefig(imagefile)
        plt.close()
      print("\\\\\\hline")
      print("\\end{tabular}")
      print("\\end{subfigure}")
    
    print("\\end{table} % ",  fairalgo, dataset)

if __name__ == "__main__":
  main()
