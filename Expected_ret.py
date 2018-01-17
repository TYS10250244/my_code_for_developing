# -*- coding: utf-8 -*-
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

def Expected_return_SMA(DF_return,window_for_ER,min_window_for_ER) :
    Exp_ret = pd.rolling_mean(DF_return,window=window_for_ER,\
                                                 min_periods=min_window_for_ER)  
    return(Exp_ret)
 
def Expected_return_EMA1(x, halflife):
    return pd.DataFrame(x).ewm(halflife=halflife).mean()

def Expected_return_EMA(Df_ret,input_Look_back,input_Min_Look_back,\
                                                         input_halflife_for_ER):
    Expected_ret_EMA = pd.DataFrame()
    for date in Df_ret[input_Min_Look_back:].resample("M").index:
        ret0 = Df_ret[:date][-1*input_Look_back:]
        halflife = input_halflife_for_ER
        if len(ret0.index) < input_halflife_for_ER:
            halflife = int(len(ret0.index)/2)
        Expected_ret_EMA0 = Expected_return_EMA1(ret0,halflife)[-1:] 
        Expected_ret_EMA0['date'] = date
        Expected_ret_EMA  = pd.concat([Expected_ret_EMA ,Expected_ret_EMA0],\
                                                                         axis=0)  
    Expected_ret_EMA.index = Expected_ret_EMA['date']
    Expected_ret_EMA = Expected_ret_EMA.drop('date',axis=1)       
    DF_Expected_ret_EMA = pd.DataFrame.from_dict(Expected_ret_EMA).transpose()    
    return DF_Expected_ret_EMA

