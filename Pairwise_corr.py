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


def Calc_Pairwise_Corr(input_ret_M,Look_back,Min_Look_back):
    Rolling_corr = pd.rolling_corr(input_ret_M,window=Look_back,\
                                            min_periods=Min_Look_back) 
    tmp = Rolling_corr.apply(lambda x: np.fill_diagonal(x.values, None), \
                                                                     axis=(1,2)) 
    apc = Rolling_corr.apply(lambda x: x.unstack().mean(skipna=True),\
                                                                     axis=(1,2))
    return apc
