#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 06:44:19 2018

@author: katou
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import seaborn as sns 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import seaborn as sns 
from sklearn.decomposition import PCA
import warnings;warnings.filterwarnings('ignore')
from pandas import Series, DataFrame
from scipy.optimize import minimize
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn import  covariance

G_LassoCV = covariance.GraphLassoCV(cv=5)

def cov_mat(df_ret,lookbak):
    return pd.DataFrame.cov(ret0[-1*lookbak:])
    

def cov_mat_L1p(df_ret,lookbak):
    G_cov = G_LassoCV.fit(ret0[-1*lookbak:].dropna(axis=0))
    return  pd.DataFrame(G_cov.covariance_)* float((lookbak)/(lookbak-1))
