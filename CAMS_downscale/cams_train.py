'''
Author: jyyd23@mails.tsinghua.edu.cn
Date: 2024-01-02 14:33:02
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-13 20:16:40
FilePath: \code\finalcode\CAMS_downscale\cams_train.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''


import pandas as pd
import numpy as np
import netCDF4 as nc
import scipy.io as sio
import time
import dataframe_image as dfi
import os, sys
os.chdir(sys.path[0])
from tqdm import trange, tqdm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'figure.dpi': 300}) # set dpi

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import neighbors, svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor

from joblib import dump, load
import concurrent.futures
from functools import partial

import preprocessing as pre



def load_trainData(pollution:str='NOX', describeFlag=True):
    '''
    description: to load the trainData from downScaleConc.m when predictionFlag=0
    param {str} pollution
    return {traindata}
    ''' 
    trainpath = '../dataset/trainpredata/trainData/'+ pollution + '_trainData.csv'
    traindata = pd.read_csv(trainpath)
    if describeFlag:
        print('Before data wash size: ', traindata.shape)
        traindata = traindata.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
        print('After data wash size: ', traindata.shape)
        traindata.head(2)
        traindata_heatmap_path = '../out/figure/pollution_traindata_heatmap/'
        pre.create_directories(traindata_heatmap_path )
        pre.corr_plot(traindata,  traindata_heatmap_path + pollution)
        print('load data....')
    else:
        traindata = traindata.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    return traindata


def train_data(pollution:str='NOX', describeFlag=True):
    '''
    description: split the x_train and y_train(measurements)
    param {str} pollution
    return {X_train_transformed, y_train, scaler}
    --------------------------------------------------------------------------
    features: cams,radiation,temperature,precipitation,humidity,Speed,road,hour,month,weekday,measurements
    '''
    traindata = load_trainData(pollution, describeFlag)
    traindata = traindata.apply(pd.to_numeric,errors="ignore")
    traindata = traindata.replace([np.inf, -np.inf], np.nan).dropna()
    traindata = traindata.dropna(how='any', axis=0)
    if pollution == 'NOX':
        traindata['road_log'] = np.exp(traindata['road'])
        traindata = traindata.drop(['road'],axis=1)
        traindata = traindata[['cams', 'radiation', 'temperature', 'precipitation', 'humidity',
            'Speed', 'road_log','hour', 'month', 'weekday', 'measurements']]
    else:
        traindata['road_log'] = (traindata['road'])**2
        traindata = traindata.drop(['road'],axis=1)
        traindata = traindata[['cams', 'radiation', 'temperature', 'precipitation', 'humidity',
            'Speed', 'road_log','hour', 'month', 'weekday', 'measurements']]

    X = traindata.iloc[:, :-1]
    Y = traindata.iloc[:, -1] # measurements

    scaler = StandardScaler().fit(X)
    X_train_transformed = scaler.transform(X)
    y_train = Y

    return X_train_transformed, y_train, scaler
    

def select_train_model(pollution:str='NOX', model_name:str='lightGBM', describeFlag=True):
    '''
    description: 
    param {str} pollution
    param {str} model_name
    return {*}
    suggestion: NOX and PM10 select lightGBM, for other pollution select GradientBoosting
    '''
    X_train_transformed, y_train, scaler = train_data(pollution, describeFlag)
    if model_name == 'RandomForest':
        regr = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap=False, n_jobs=20).fit(X_train_transformed, y_train)
    elif model_name == 'lightGBM':
        regr = lgb.LGBMRegressor(random_state=42, n_jobs=20).fit(X_train_transformed, y_train)
    elif model_name == 'GradientBoosting':
        regr = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train_transformed, y_train)
    elif model_name == 'Lasso':
        # linear_reg = LinearRegression()
        regr = Lasso(alpha=80).fit(X_train_transformed, y_train)
    elif model_name == 'SVM':
        regr = svm.SVR(C=100, gamma=0.5).fit(X_train_transformed, y_train)
    elif model_name == 'KNeighbors':
        regr = neighbors.KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=20).fit(X_train_transformed, y_train)
    elif model_name == 'DecisionTree':
        regr = DecisionTreeRegressor(criterion='absolute_error', max_depth=10).fit(X_train_transformed, y_train)
    elif model_name == 'AdaBoost':
        regr = AdaBoostRegressor(random_state=42, n_estimators=10, learning_rate=0.01, loss='square').fit(X_train_transformed, y_train)
    elif model_name == 'Stacking':
        # linear_reg = LinearRegression().fit(X_train_transformed, y_train)
        # lasso_reg = Lasso(alpha=80).fit(X_train_transformed, y_train)
        svm_reg = svm.SVR(C=100, gamma=0.5).fit(X_train_transformed, y_train)
        knn_reg = neighbors.KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=20).fit(X_train_transformed, y_train)
        # tree_reg = DecisionTreeRegressor(criterion='absolute_error', max_depth=10).fit(X_train_transformed, y_train)
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_features=4, bootstrap=False, n_jobs=20).fit(X_train_transformed, y_train)
        # ada_reg = AdaBoostRegressor(random_state=42, n_estimators=10, learning_rate=0.01, loss='square').fit(X_train_transformed, y_train)
        # gbr_reg = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train_transformed, y_train)
        lgb_reg = lgb.LGBMRegressor(random_state=42, n_jobs=20).fit(X_train_transformed, y_train)
        estimators = [("knn", knn_reg),("rf", rf_reg),("lgb", lgb_reg),("svm", svm_reg)]
        final_estimator = Lasso()
        regr = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
    model_path = '../out/model/cams_pred_model/'
    pre.create_directories(model_path)
    dump(regr, model_path + model_name + pollution + '_trainedModel.joblib')
    print('Sucess dump model!')
    return scaler, regr

def get_annual_data(pollution:str='NOX', model_name:str='RandomForest', describeFlag=True):
    start_time = time.time()
    # Calculate the results of downScaleConc.m and return the mean of the results
    n = 8760
    # n = 8784
    year = 2020
    scaler, regr = select_train_model(pollution, model_name, describeFlag)

    def process_hour(hourId):
        data = np.fromfile('../dataset/trainpredata/allbin/' + pollution + 'trainbin/' + str(hourId) + '_' + pollution + '_predData.bin', dtype=np.float32)
        data = np.reshape(data, (-1, 112041))
        data = pd.DataFrame(np.transpose(data))
        dataPred = data.apply(pd.to_numeric,errors="ignore")
        dataPred.fillna(0, inplace=True)

        cams = data.iloc[:, 0]
        XPred = dataPred.iloc[:, :]

        X_pred_transformed = scaler.transform(XPred)
        Y_pred = regr.predict(X_pred_transformed)
        pred = np.multiply(cams, Y_pred)
        pred = np.array(pred).reshape(112041, 1)
        return pred

    # # method of map
    # results = list(tqdm(map(process_hour, range(n)), total=n))
    
    # # method of multiprocessing
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        results = list(tqdm(executor.map(process_hour, range(n), chunksize=1000), total=n))
    
    pred_sum = np.mean(results, axis=0)
    print('The pred size: ', pred_sum.shape)
    end_time = time.time()
    cost_time = end_time - start_time
    print('time cost : %.5f sec' %cost_time)
    downscale_path = '../dataset/trainpredata/allbin/ave2020/' + pollution + '/'
    pre.create_directories(downscale_path)
    np.savetxt(downscale_path + str(year) +'_downScale_' + pollution + model_name + '.csv', pred_sum, delimiter=',')
    return cost_time

def get_matdata(pollution:str='NOX',  model_name:str='RandomForest'):

    Delta_x = 0.01
    Delta_y = 0.01

    nf = nc.Dataset('../dataset//CAMS/CAMS_European_airquality_forecasts/SingleLevel_202101.nc')
    lon = np.array(nf.variables['longitude'][:]).reshape(-1, 1)
    lat = np.array(nf.variables['latitude'][:]).reshape(-1, 1)

    lonNew = np.arange(lon[0][0], lon[-1][0], Delta_x)
    latNew = np.arange(lat[0][0], lat[-1][0], -Delta_y)
    xnew, ynew = np.meshgrid(lonNew, latNew)
    print(lonNew.shape)
    print(latNew.shape)

    # 20230723
    avgnoxfilename = '../dataset/trainpredata/allbin/ave2020/' + pollution + '/' + '2020_downScale_' + pollution + model_name + '.csv'
    temp= pd.read_csv(avgnoxfilename, header=None)
    value = np.array(temp).reshape(531, 211).T


    avgConc = {
        'lonNew': np.array(lonNew),
        'latNew': np.array(latNew),
        'avgConc': np.array(value),
    }
    year = 2020
    filename = '../pncEstimator-main/src/postProcessing/matdata/pollution/'+ str(year) + pollution + 'avgConc_'+ model_name + '.mat'
    sio.savemat(filename, avgConc)

def main():
    pollution_list = ['NOX', 'NO2', 'PM10', 'PM2.5', 'O3']
    model_list = ['lightGBM', 'GradientBoosting']
    cost_times = np.zeros((len(pollution_list), len(model_list)))
    for i in range(len(pollution_list)):
        for j in range(len(model_list)):
            cost_time = get_annual_data(pollution_list[i], model_list[j], describeFlag=False)
            get_matdata(pollution_list[i], model_list[j])
            cost_times[i, j] = cost_time
    cost_times = pd.DataFrame(cost_times)
    cost_times.columns = model_list
    cost_times.index = pollution_list
    dfi.export(obj = cost_times, filename='../out/figure/CAMS_cost_times.png', fontsize = 14)

if __name__ == "__main__":
    main()