
import numpy as np
import pandas as pd
import pyfolio.plotting as plotting
import pyfolio.timeseries as timeseries
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import seaborn as sns 
from sklearn.decomposition import PCA
import warnings;warnings.filterwarnings('ignore')
from pandas import Series, DataFrame
from scipy.optimize import minimize
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn import  covariance

import sys
sys.path.append('K:\\KB600\\AssetAllocation\\Research_Platform\\python\\Portfolio')
# sys.path
import Daily.Out_performance as Daily_Out_performance
import Daily.Expected_ret as Daily_Expected_ret
import Daily.Method_for_portfolio_construction as Daily_portfolio

Out_prf = Daily_Out_performance.out_performacne
ER_EMA = Daily_Expected_ret.Expected_return_EMA
MaxSharpe_withL1P = Daily_portfolio.calc_historical_max_sharpe_weight_EMAwithL1P
ERC = Daily_portfolio.calc_historical_RP_weight
G_LassoCV = covariance.GraphLassoCV(cv=5)

Data_direct = "K:\\KB600\\AssetAllocation\\Sawagashira\Kato\\Marketing\\DL\\data\\"

ret_data = pd.read_csv(Data_direct+"Return_Data_for_DL_daily.csv",index_col=0)
ret_data.index = pd.to_datetime(ret_data.index)
Index_value=100*((1+ret_data).cumprod())
Index_value = Index_value.resample("D",how='last').dropna(axis=0)
base_return = Index_value/Index_value.shift(1)-1

"""
For Return Adjust
"""
Look_back = 250*3
Min_Look_back = 250
target_vol = 0.05

"""
For portforio construction
"""
input_Look_back = 250*3
input_Min_Look_back = 250


"""
 bounds for portforio construction
"""
set_bounds = (0.00, 1.0)

"""
 Expected return
"""
input_Look_back_for_Expected_ret = 250*3
input_Min_Look_back_for_Expected_ret = 250
input_halflife_for_ER = 250

"""
 Selected index
"""
num_Selected = 3

"""
 Trade(T+(x-1))
"""
lag_trade = 4

ExpRet = ER_EMA(base_return,input_Look_back,input_Min_Look_back,\
                                                          input_halflife_for_ER)


base_return = base_return.drop('Hedge',axis=1)
bounds = [set_bounds for i in base_return.columns]
port_weight = MaxSharpe_withL1P(base_return,input_Look_back,\
                                               input_Min_Look_back,ExpRet,bounds)
# port_weight = ERC (base_return,input_Look_back,input_Min_Look_back)

port_weight = pd.DataFrame.from_dict(port_weight).transpose()
port_weight.columns = base_return.columns 

def adj_weight(DF_ret,DF_weight):
    date_tmp = pd.DataFrame()
    date_tmp['tmp_date'] = DF_ret.ix[:,1]
    
    DF_weight = pd.concat([date_tmp,DF_weight],axis=1).\
                                  fillna(method='ffill').drop('tmp_date',axis=1)
    DF_weight = pd.concat([date_tmp,DF_weight],axis=1,\
                             join_axes=[date_tmp.index]).drop('tmp_date',axis=1)  
    return pd.DataFrame(DF_weight)

DF_port_weight = adj_weight(base_return,port_weight.resample("M").last()).dropna(axis=0)
P_ret  = pd.DataFrame(np.sum((base_return*DF_port_weight.\
                                     shift(lag_trade-1)).dropna(axis=0),axis=1))
P_ret.columns = ['portfolio return']

%matplotlib inline
Out_prf(P_ret)
