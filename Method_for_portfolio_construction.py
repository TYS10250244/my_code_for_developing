# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
from sklearn.decomposition import PCA
from pandas import Series, DataFrame
from scipy.optimize import minimize
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn import  covariance

G_LassoCV = covariance.GraphLassoCV(cv=5)

def adj_return(input_ret):
    input_ret.index = pd.to_datetime(input_ret.index)
    a_ret = input_ret.copy()
    rets = target_vol * a_ret/(np.std(a_ret, ddof=1)*np.sqrt(250))
    return rets

def weight_sum_constraint(x) :
    return(x.sum() - 1.0 )


def weight_longonly(x) :
    return(x)

def RC(weight, covmat) :
    weight = np.array(weight)
    variance = weight.T @ covmat @ weight
    sigma = variance ** 0.5
    mrc = 1/sigma * (covmat @ weight)
    rc = weight * mrc
    rc = rc / rc.sum()
    return(rc)
    
def RiskParity_objective(x) :
    variance = x.T @ covmat @ x
    sigma = variance ** 0.5
    mrc = 1/sigma * (covmat @ x)
    rc = x * mrc
    a = np.reshape(rc, (len(rc), 1))
    risk_diffs = a - a.T
    sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
    return (sum_risk_diffs_squared)    
        
def Minimum_variance(x) :
    variance = x.T @ covmat @ x
    risk = variance 
    return (risk)                  


def Expected_return_SMA(DF_return,window_for_ER,min_window_for_ER) :
    Exp_ret = pd.rolling_mean(DF_return,window=window_for_ER,\
                                                 min_periods=min_window_for_ER)  
    return(Exp_ret)
 
def Expected_return_EMA1(x, halflife):
    return pd.DataFrame(x).ewm(halflife=halflife).mean()
        
def Max_sharpe_objdect(x) :
    variance = x.T @ covmat @ x
    sigma = variance ** 0.5
    ret =  expcted_ret @ x 
    expeted_sharpe = (ret/sigma)[0]  
    return (-1 * expeted_sharpe)                                                 

def RiskParity(covmat) :
    
    x0 = np.repeat(1/covmat.shape[1], covmat.shape[1]) 
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                  {'type': 'ineq', 'fun': weight_longonly})
    options = {'ftol': 1e-20, 'maxiter': 800}
    result = minimize(fun = RiskParity_objective,
                      x0 = x0,
                      method = 'SLSQP',
                      constraints = constraints,
                      options = options)                                 
    return(result.x)                     


def MinimumVariance(covmat,bounds) :
    # global bounds
    x0 = np.repeat(0, covmat.shape[1]) 
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                  {'type': 'ineq', 'fun': weight_longonly})
    options = {'ftol': 1e-20, 'maxiter': 10000}
    result = minimize(fun = Minimum_variance,
                      x0 = x0,
                      method = 'SLSQP',
                      constraints = constraints,
                      bounds = bounds,
                      options = options)
    # print(result.success)                                  
    return(result.x)     


def Max_sharpe(covmat, expcted_ret,bounds) :
    # global bounds
    x0 = np.repeat(1/covmat.shape[1], covmat.shape[1]) 
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                  {'type': 'ineq', 'fun': weight_longonly})
    options = {'ftol': 1e-20, 'maxiter': 10000}
    result = minimize(fun = Max_sharpe_objdect,
                      x0 = x0,
                      method = 'SLSQP',
                      constraints = constraints,
                      bounds = bounds,
                      options = options)                                 
    return(result.x)    


covmat = pd.DataFrame()
def calc_historical_RP_weight(df_ret,lookbak,min_lookbak):
    global covmat
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:
    # for d in df_ret.index:

        # print('*--------------------'+str(d)+'*----------------------------')        
        ret0 = df_ret[:d][-1*lookbak:]
        covmat = DataFrame.cov(ret0)
        # print(covmat)            
         
        result_weight[d] = RiskParity(covmat)
        # print(RiskParity(covmat)) 
        
    return result_weight


covmat = pd.DataFrame()
def calc_historical_RP_weight_withL1P(df_ret,lookbak,min_lookbak):
    global covmat
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:
        ret0 = df_ret[:d][-1*lookbak:]
        G_cov = G_LassoCV.fit(ret0.dropna(axis=0))
        covmat = pd.DataFrame(G_cov.covariance_) * float((lookbak)/(lookbak-1))  
         
        result_weight[d] = RiskParity(covmat)
        
    return result_weight



covmat = pd.DataFrame()
def calc_historical_MV_weight(df_ret,lookbak,min_lookbak, bounds):
    global covmat
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:
    # for d in df_ret.index:

        # print('*---------------------'+str(d)+'*----------------------------')        
        ret0 = df_ret[:d][-1*lookbak:]
        covmat = DataFrame.cov(ret0)
        result_weight[d] = MinimumVariance(covmat, bounds)
        
    return result_weight
    
    
def calc_historical_InvVol_weight(df_ret,lookbak,min_lookbak):
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:

        ret0 = df_ret[:d][-1*lookbak:]
        Inv_vol = 1/np.std(ret0)
        result_weight[d] = Inv_vol/np.sum(Inv_vol,axis=0)
        
    return result_weight
    

covmat = pd.DataFrame()
expcted_ret = pd.DataFrame()
def calc_historical_max_sharpe_weight_SMA(df_ret,lookbak,min_lookbak,\
    input_Look_back_for_Expected_ret,input_min_Look_back_for_Expected_ret,bounds):
    global covmat, expcted_ret
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:
        ret0 = df_ret[:d][-1*lookbak:]
        covmat = DataFrame.cov(ret0)
        expcted_ret = Expected_return_SMA(df_ret[:d],\
                            input_Look_back_for_Expected_ret,\
                            input_min_Look_back_for_Expected_ret)[-1:].fillna(0)
        result_weight[d] = Max_sharpe(covmat,expcted_ret,bounds)
    return result_weight



covmat = pd.DataFrame()
expcted_ret = pd.DataFrame()
def calc_historical_max_sharpe_weight_EMAwithL1P(df_ret,lookbak,min_lookbak,\
                                                            input_df_ER,bounds):
    global covmat, expcted_ret
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:
        ret0 = df_ret[:d][-1*lookbak:]
        G_cov = G_LassoCV.fit(ret0.dropna(axis=0))
        covmat = pd.DataFrame(G_cov.covariance_)* float((lookbak)/(lookbak-1))  
        expcted_ret = input_df_ER[:d][-1:][df_ret.columns]
        result_weight[d] = Max_sharpe(covmat,expcted_ret,bounds)
    return result_weight     
   
def calc_historical_InvVol_weight_selected(df_ret,lookbak,min_lookbak,\
                                                            input_num_Select):
    result_weight = pd.DataFrame()
    
    for d in df_ret[min_lookbak:].resample("M").index:
        ret0 = df_ret[:d][-1*lookbak:]
        ret_vol = pd.DataFrame(np.mean(ret0)/np.std(ret0))
        ret_vol['rank'] = ret_vol.rank(ascending=False)
        selected_index = ret0[ret_vol[ret_vol['rank'] <=input_num_Select].index]
        Inv_vol = 1/np.std(selected_index)
        result_weight0 = pd.DataFrame(Inv_vol/np.sum(Inv_vol,axis=0)).transpose()
        result_weight0['date'] = d
        result_weight = pd.concat([result_weight,result_weight0],axis=0).fillna(0)
    result_weight.index = result_weight['date']
    result_weight = result_weight.drop('date',axis=1)  
    return result_weight        

def calc_robust_covariance(df_ret,alpha,lookbak,min_lookbak,rbstlookbak,d):
    ret0 = df_ret[:d][-1*lookbak:]
    normal_covmat = DataFrame.cov(ret0)                      

    ret_for_rbst =df_ret[:d][-1*rbstlookbak:]
    quant = ret_for_rbst.quantile([alpha/2, (1-alpha/2)])
    q_l = quant.loc[alpha/2,:]
    q_u = quant.loc[(1-alpha/2),:]

    ret_for_rbst_L = np.sign(ret_for_rbst[ret_for_rbst.\
                                  apply(lambda x: x-q_l ,axis=1)<0]).fillna(0)
    ret_for_rbst_U = np.sign(ret_for_rbst[ret_for_rbst.\
                                  apply(lambda x: x-q_u ,axis=1)>0]).fillna(0)
    ab_normal_signal = np.abs((ret_for_rbst_L + ret_for_rbst_U)\
                                         [(ret_for_rbst_L + ret_for_rbst_U)!=0])
    ab_normal_ret = ret_for_rbst *  ab_normal_signal                                                        
      
    normal_signal = ab_normal_signal.copy()
    normal_signal = np.abs(ab_normal_signal.fillna(0) -1)                                             
    normal_signal = normal_signal[normal_signal>0]

    normal_ret = ret0 *  normal_signal 
    rdst_std = np.sqrt(alpha * (np.std(abs(ab_normal_ret),\
                  ddof=1 ))**2 + (1-alpha) * (np.std(normal_ret, ddof=1 ) )**2)

    covmat = DataFrame.cov(ret0)
    rbst_covmat = covmat.copy()
    np.fill_diagonal(rbst_covmat.values, rdst_std**2)
                                                                                                                                                                                                          
    return rbst_covmat
                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

covmat = pd.DataFrame()
rbst_covmat = pd.DataFrame()
expcted_ret = pd.DataFrame()
def calc_historical_max_sharpe_weight_EMA_robust(df_ret,lookbak,min_lookbak,\
                                                    input_Expected_ret, bounds):
    global covmat, expcted_ret, d 
    result_weight = {}
    
    for d in df_ret[min_lookbak:].resample("M").index:
        ret0 = df_ret[:d][-1*lookbak:]
        covmat = calc_robust_covariance(df_ret,0.05,lookbak,min_lookbak,120,d)
        expcted_ret = input_Expected_ret[:d][-1:][df_ret.columns]
        result_weight[d] = Max_sharpe(covmat,expcted_ret, bounds)
    return result_weight
 
                
