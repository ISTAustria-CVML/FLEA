#!/usr/bin/env python3
# coding: utf-8
# Licensed under the terms of the MIT license, see LICENSE.md

import folktables
import jax.numpy as jnp
import numpy as np
import pandas as pd

from folktables import ACSDataSource, ACSIncome
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# load data and store information
class DatasetInfo():
  def __init__(self, args):
    self.name = args.dataset
    if self.name == 'adult':
      self.dataset_filename = "./data/adult.csv"
      self.protected_column = 'gender'
      self.target_column = 'income'
      self.categorial_cols = ['workclass','education','hours_cat','age_cat','native-country','race','gender']
      self.numeric_cols = [] 
    elif self.name == 'compas':
      self.dataset_filename = "./data/compas-nonames.csv"
      self.protected_column = 'sex'
      self.target_column = 'two_year_recid'
      self.categorial_cols = ['c_charge_degree','age_cat','race','sex']
      self.numeric_cols = ['priors_count']
    elif self.name == 'germancredit':
      self.dataset_filename = "./data/german_credit_data.csv"
      self.protected_column = 'Sex'
      self.target_column = 'Risk'
      self.categorial_cols = ['Age_cat', 'Sex', 'Saving accounts', 'Checking account']
      self.numeric_cols = ['Duration','Credit amount']
    elif self.name == 'drugs':
      self.dataset_filename = "./data/drug_consumption.csv"
      self.protected_column = 'Gender'
      self.target_column = 'Coke' # 'CL0' means never, 'CL1' means over a Decade Ago, etc.
      self.categorial_cols = []
      self.numeric_cols = ['Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS']
    elif self.name == 'folktables': # v0.0.11 (Dec 5, 2021)
      self.dataset_filename = None
      self.protected_column = 'SEX'
      self.target_column = 'label'
      self.categorial_cols = ['AGEP_CAT','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP_CAT','SEX','RAC1P'] # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf
      self.numeric_cols = [] 
    else:
      print(f"Unknown datasets {name}")
      raise SystemExit
    
    if self.protected_column in self.categorial_cols:
      self.categorial_cols.remove(self.protected_column)
    if self.protected_column in self.numeric_cols:
      self.numeric_cols.remove(self.protected_column)
    self.categorial_cols += ['protected'] # put at last position 
    
    self.data = self.load_dataset()
    self.data = self.prepare_dataset()
    self.onehotenc = OneHotEncoder(sparse=False, drop=args.onehotdrop).fit(self.data[self.categorial_cols])
    self.feature_names = ['const']
    for cat,vals in zip(self.categorial_cols,self.onehotenc.categories_):
        if args.onehotdrop == 'first':
          self.feature_names += [f"{cat}#{v}" for v in vals[1:]]
        else:
          self.feature_names += [f"{cat}#{v}" for v in vals]
    self.feature_names += self.numeric_cols
    
  def load_dataset(self):
    if self.dataset_filename:
      data = pd.read_csv(self.dataset_filename, index_col=False)
    elif self.name == 'folktables':
      data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
      data = {}
      for state in folktables.state_list: # load state-by-state to simplify subsampling
        acs_data = data_source.get_data(states=[state], download=True)
        if len(acs_data) > 10000:
          acs_data = acs_data.sample(10000)
        features, label, _ = ACSIncome.df_to_numpy(acs_data) # different group than default
        new_data = pd.DataFrame(features, columns=ACSIncome.features)
        new_data['label'] = label
        data[state] = new_data
      data = pd.concat(data)
    return data
  
  def prepare_dataset(self):
    if self.name == 'adult':
      age_bins = [15,25,35,45,55,65,99]
      self.data['age_cat'] = pd.cut(self.data['age'], bins=age_bins)
      hours_bins = [0,20,30,40,99]
      self.data['hours_cat'] = pd.cut(self.data['hours-per-week'], bins=hours_bins)
      
      self.data.loc[(self.data['native-country'] != "United-States"), 'native-country']='Other'

      target_dict = {'<=50K':0, '>50K':1}
      if self.protected_column == 'gender':
        protected_dict = {'Male':0, 'Female':1}
      else:
        print("Unknown protected_column", protected_column)
        raise SystemExit

    elif self.name=='compas':
      for col in ['name','first','last','dob']:
        if col in self.data:
          del self.data[col]    # remove personal data if present

      idx = (self.data['days_b_screening_arrest'] <= 30) & \
            (self.data['days_b_screening_arrest'] >= -30) & \
            (self.data['is_recid'] != -1) & \
            (self.data['c_charge_degree'] != "O") & \
            (self.data['score_text'] != "N/A" )
      self.data = self.data[idx] # filter like ProPublica did

      self.data.loc[self.data['race']=='Native American', 'race']='Other'
      self.data.loc[self.data['race']=='Asian', 'race']='Other'
      
      if self.protected_column == 'sex':
        protected_dict = {'Male':0, 'Female':1}
      else:
        print("Unknown protected_column", protected_column)
        raise SystemExit
      target_dict = {} # no conversion necessary
      
    elif self.name=='germancredit':
      age_bins = [15,25,35,45,55,65,99]
      self.data['Age_cat'] = pd.cut(self.data['Age'], bins=age_bins)
      self.data['Credit amount'] /= 1000. # scale to reasonable range
      
      protected_dict = {'male':0, 'female':1}
      target_dict = {'bad':0, 'good':1}
    
    elif self.name=='drugs':
      target_dict = {'CL0':0, 'CL1':1, 'CL2':1, 'CL3':1, 'CL4':1, 'CL5':1, 'CL6':1} # CL0 = "never"

    elif self.name=='folktables':
      age_bins = [15,25,35,45,55,65,99]
      self.data['AGEP_CAT'] = pd.cut(self.data['AGEP'], bins=age_bins)
      del self.data['AGEP']
      hours_bins = [0,20,30,40,99]
      self.data['WKHP_CAT'] = pd.cut(self.data['WKHP'], bins=hours_bins)
      del self.data['WKHP']
      self.data.loc[(self.data['POBP'] < 100), 'POBP'] = 0 # USA
      self.data.loc[(self.data['POBP'] >= 100), 'POBP'] = 1 # Other
      self.data['OCCP'] //= 1000    # keep only major category of 4-digit code
      
      target_dict = {False:0, True:1}
      protected_dict = {1.0:0, 2.0:1} 
      
    # create 'target' column and make sure entries are 0/1
    self.data['target'] = self.data[self.target_column].replace(target_dict).astype(int).values
    del self.data[self.target_column] # no reason to keep
    
    if self.name == 'drugs':
      self.data['protected'] = 1*(self.data[self.protected_column]>0).values
    else: 
      self.data['protected'] = self.data[self.protected_column].replace(protected_dict).astype(int).values
    del self.data[self.protected_column] # no reason to keep
    return self.data
  
  def train_test_split(self, train_size=None, test_size=.2, random_state=0):
    if test_size is not None:
      self.df_train, self.df_test = train_test_split(self.data, test_size=test_size, random_state=random_state)
    elif train_size is not None:
      self.df_train, self.df_test = train_test_split(self.data, train_size=train_size, random_state=random_state)
    else:
      print("Error: must specifiy either train_size or test_size for train/test split")
      raise SystemExit

  def make_sources(self, n_sources, random_state=0):
    if self.name == 'folktables':
      sources = [df for _,df in self.df_train.groupby(level=0)] # group by first element of multiindex (=state)
      sources = sources[:n_sources]
    else:
      sources = np.array_split(self.df_train, indices_or_sections=n_sources)
    return sources
  
  def features(self, df=None):
    if df is None:
      df = self.df_train
    X1 = self.onehotenc.transform(df[self.categorial_cols])
    X2 = df[self.numeric_cols].to_numpy()
    if X2.shape[1] == 0:
      X = X1
    else:
      X = jnp.hstack([X1, X2])
    y = jnp.asarray(df['target'])
    
    self.dim = X.shape[-1]
    return X,y
