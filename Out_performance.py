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


   
"""     
-----------------------------------------------------------------------------------
For Output Performance Summary
-----------------------------------------------------------------------------------
""" 

def out_performacne(DF_Base_data):
    ADJ_MF_base =  DF_Base_data.copy()

    Vol_adj_ret = pd.DataFrame(np.std(ADJ_MF_base, ddof=1)*np.sqrt(250)).transpose()
    Vol_adj_ret.index = ['Volatility']

    AunRet_adj_ret = pd.DataFrame(((1+ADJ_MF_base).\
                                cumprod()[-1:])**(250/len(ADJ_MF_base.index))-1)
    AunRet_adj_ret.index = ['Return']

    AunRet_SR = pd.DataFrame(np.array(AunRet_adj_ret)/np.array(Vol_adj_ret))
    AunRet_SR.index = ['Sharpe']
    AunRet_SR.columns = AunRet_adj_ret.columns
    
    equity_curve = 100*((1+ADJ_MF_base).cumprod())
       
    Roll_Max = pd.rolling_max(equity_curve,window=len(equity_curve.index), \
                                                                  min_periods=1)
    Historical_Drawdown = equity_curve/Roll_Max - 1.0      
    Max_Drawdown = pd.rolling_min(Historical_Drawdown, \
                             window=len(equity_curve.index), min_periods=1)[-1:]
    Max_Drawdown.index = ['MAX DD']
    
    Skew = pd.DataFrame(ADJ_MF_base.skew()).transpose()
    Skew.index = ['Skew']
    
    Kurtosis = pd.DataFrame(ADJ_MF_base.kurtosis()).transpose()
    Kurtosis.index = ['Kurtosis']
    
    Performance = pd.concat([AunRet_adj_ret,Vol_adj_ret,\
                                   AunRet_SR,Max_Drawdown,Skew,Kurtosis],axis=0)

    def specific_term_Perf(DF_ret,term): 
        ADJ_MF_base_st = DF_ret[-1*term:]
        Vol_adj_ret_st = pd.DataFrame(np.std(ADJ_MF_base_st, \
                                               ddof=1)*np.sqrt(250)).transpose()
        Vol_adj_ret_st.index = ['Volatility']

        AunRet_adj_ret_st = pd.DataFrame(((1+ADJ_MF_base_st).cumprod()[-1:])**\
                                              (250/len(ADJ_MF_base_st.index))-1)
        AunRet_adj_ret_st.index = ['Return']

        AunRet_SR_st = pd.DataFrame(np.array(AunRet_adj_ret_st)/np.array(Vol_adj_ret_st))
        AunRet_SR_st.index = ['Sharpe']
        AunRet_SR_st.columns = AunRet_adj_ret.columns

        equity_curve_st = 100*((1+ADJ_MF_base_st).cumprod())
       
        Roll_Max_st = pd.rolling_max(equity_curve_st,\
                                 window=len(equity_curve.index),  min_periods=1)
        Historical_Drawdown_st = equity_curve_st/Roll_Max_st - 1.0      
        Max_Drawdown_st = pd.rolling_min(Historical_Drawdown_st, \
                             window=len(equity_curve.index), min_periods=1)[-1:]
        Max_Drawdown_st.index = ['MAX DD']   
        
        Performance_st = pd.concat([AunRet_adj_ret_st,Vol_adj_ret_st,\
                                              AunRet_SR_st,Max_Drawdown],axis=0)      
                       
        return Performance_st
    
    Performance_1y = specific_term_Perf(ADJ_MF_base,250*1) 
    Performance_3y = specific_term_Perf(ADJ_MF_base,250*3) 
    Performance_5y = specific_term_Perf(ADJ_MF_base,250*5) 

    equity_curve = 100*((1+ADJ_MF_base).cumprod())

    Roll_Max_12m = pd.rolling_max(equity_curve, window=250, min_periods=250)
    Historical_Drawdown_12m = equity_curve/Roll_Max_12m - 1.0    
 
    sns.set_palette("Set1", len(equity_curve.columns))

    plt.figure(figsize=(10, 5), dpi=80)
    plt.title('Equity Curve')
    plt.plot(equity_curve.index,equity_curve)
    plt.legend(equity_curve.columns,loc="upper center",bbox_to_anchor=(1.3,0.8)) 
    plt.suptitle('')
    plt.show()
    
    plt.figure(figsize=(10, 5), dpi=80)
    plt.title('Max DD (250-days rolling)')
    plt.plot(Historical_Drawdown_12m.index,Historical_Drawdown_12m)
    plt.legend(Historical_Drawdown_12m.columns,loc="upper center",\
                                                       bbox_to_anchor=(1.3,0.8)) 
    plt.suptitle('')
    plt.show()    
    

    print('----------------Statistics(Whole Period)---------------------------')    
    ax1 = plt.subplot(111)
    plt.axis('off')
    tbl = table(ax1, np.round(Performance.transpose(),4), loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.show() 
    
    print('----------------Statistics(1 year)---------------------------')
    ax1 = plt.subplot(111)      
    plt.axis('off') 
    tbl = table(ax1, np.round(Performance_1y.transpose(),4), loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.show() 

    print('----------------Statistics(3 year)---------------------------')    
    ax1 = plt.subplot(111)
    plt.axis('off')   
    tbl = table(ax1, np.round(Performance_3y.transpose(),4), loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.show() 

    print('----------------Statistics(5 year)---------------------------') 
    ax1 = plt.subplot(111)
    plt.axis('off')  
    tbl = table(ax1, np.round(Performance_5y.transpose(),4), loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.show() 
            
    
    CORR_Base_data = ADJ_MF_base.corr()   
    plt.figure(figsize=(6, 3), dpi=80)
    heatmap = sns.heatmap(CORR_Base_data,cbar=False,annot=True,\
                                                       cmap='Blues_r',fmt='.3f')
    plt.suptitle('Correlation')
    plt.show()
   


    ret_a = equity_curve.resample("A",how='last').pct_change()
    ret_a['year'] = (ret_a.index).year
    ret_a.index = ret_a['year']

    ret_a[DF_Base_data.columns].plot(kind='bar',alpha=0.8,figsize=(11, 5));
    plt.legend(loc=(1.0,0.4))
    plt.suptitle('Calendar Year Return')
    plt.show()
    
    return