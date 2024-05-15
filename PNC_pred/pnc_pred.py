'''
Author: jyyd23@mails.tsinghua.edu.cn
Date: 2024-01-02 14:43:25
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-09 16:40:44
FilePath: PNC_pred\pnc_pred.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''


import os, sys
os.chdir(sys.path[0])
import pandas as pd
import numpy as np
import seaborn as sns
import netCDF4 as nc
import scipy.io as sio
import time
import matplotlib.pyplot as plt
import dataframe_image as dfi
from tqdm import trange, tqdm
import itertools
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={'figure.dpi': 300}) # set dpi=300

from sklearn.preprocessing import StandardScaler, MinMaxScaler


from joblib import dump, load
import concurrent.futures
import multiprocessing

import CAMS_downscale.preprocessing as pre
import CAMS_downscale.cams_train as camstrain


def load_and_split_pnc_data():
    '''
    description: Load the pnc feature data and split the data by year, this part of the data is used to 
                 get the scaler of pnc data to pred the pullotion_predData.
                 This part of the function has already run in model_train.ipynb, it's not easy to import ipynb
                 file, so I use a copy job.
    return {pncdata2016_2019, pncdata2020, pnc_x_train, pnc_x_test, pnc_y_train, pnc_y_test}
    '''
    pnc_feature_path = '../dataset/NABEL/feature_data/'
    pncdata2016_2019 = pd.read_csv(pnc_feature_path + 'feature_data_2016_2019.csv')
    pncdata2020 = pd.read_csv(pnc_feature_path + 'feature_data_2020.csv')
    pncdata2016_2019 = pncdata2016_2019.iloc[:,1:]
    pncdata2020 = pncdata2020.iloc[:,1:]
    pncdata2016_2019 = pncdata2016_2019[['Date/time', 'station', 'CPC [1/cm3]', 'NOX [ug/m3 eq. NO2]',
                   'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]',
                   'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]',
                   'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']]
    pncdata2020 = pncdata2020[['Date/time', 'station', 'CPC [1/cm3]', 'NOX [ug/m3 eq. NO2]',
                   'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]',
                   'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]',
                   'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']]
    
    # split data and the ratio
    pnc_x_train = pncdata2016_2019.iloc[:,3:].values
    pnc_y_train = pncdata2016_2019.iloc[:,2].values
    pnc_x_test = pncdata2020.iloc[:,3:].values
    pnc_y_test = pncdata2020.iloc[:,2].values
    # print('--------------------------------------------------')
    # print(pnc_y_train.shape, pnc_y_test.shape)
    # print('The train dataset ratio = ', pnc_y_train.shape[0]/ (pnc_y_test.shape[0]+pnc_y_train.shape[0]))
    # print('The test dataset ratio = ', pnc_y_test.shape[0]/ (pnc_y_test.shape[0]+pnc_y_train.shape[0]))
    return pncdata2016_2019, pncdata2020, pnc_x_train, pnc_x_test, pnc_y_train, pnc_y_test

def get_pnc_scaler():
    '''
    description: get the pnc scaler.
    return {*}
    '''
    pncdata2016_2019, pncdata2020, pnc_x_train, pnc_x_test, pnc_y_train, pnc_y_test = load_and_split_pnc_data()
    scaler_std = StandardScaler()
    pnc_x_all = np.vstack((pnc_x_train, pnc_x_test))
    # print('Check the pnc data nums: ', pnc_x_all.shape, pnc_x_train.shape[0], pnc_x_test.shape[0])
    pnc_x_all_scaler = scaler_std.fit(pnc_x_all)
    pnc_x_train = pnc_x_all_scaler.transform(pnc_x_train)
    pnc_x_test = pnc_x_all_scaler.transform(pnc_x_test)
    # print('Load the pnc scaler!')
    return pnc_x_all_scaler, pnc_x_train, pnc_x_test, pnc_y_train, pnc_y_test

def load_pnc_joblib(pnc_model_name:str='RandomForest'):
    '''
    description: Load the model.joblib train in model_train.ipynb.
    param {str} pnc_model_name
    return {pnc_regr}
    '''
    joblib_path = '../out/model/pnc_pred_model/'
    if pnc_model_name == 'RandomForest':
        pnc_regr = load(joblib_path + 'rf_PNC_trainedModel.joblib')
    elif pnc_model_name == 'lightGBM':
        pnc_regr = load(joblib_path + 'lgb_PNC_trainedModel.joblib')
    elif pnc_model_name == 'Stacking':
        pnc_regr = load(joblib_path + 'stack_trainedModel.joblib')
    else:
        print('Please check the model has been dumped to local!')
    # print('Load the pnc regr joblib!')
    return pnc_regr

def cams_train_data(pollution_data, pollution:str='NOX', downFlag=True):
    '''
    description: 
    param {*} pollution_data
    param {str} cams_model_name
    param {str} pollution
    param {*} downFlag
    return {*}
    '''
    pollution_data = np.reshape(pollution_data, (-1, 112041))
    pollution_data = pd.DataFrame(np.transpose(pollution_data))
    pollution_dataPred = pollution_data.apply(pd.to_numeric,errors="ignore")
    pollution_dataPred.fillna(0, inplace=True)
    pollution_dataPred.columns = ['cams', 'radiation', 'temperature', 'precipitation', 'humidity', 'Speed',
                    'road', 'hour',  'month', 'weekday']
    if pollution == 'NOX':
        pollution_dataPred['road_log'] = np.exp(pollution_dataPred['road'])
    else:
        pollution_dataPred['road_log'] = (pollution_dataPred['road'])**2
    pollution_dataPred = pollution_dataPred.drop(['road'],axis=1)
    pollution_dataPred = pollution_dataPred[['cams', 'radiation', 'temperature', 'precipitation', 'humidity',
        'Speed', 'road_log','hour', 'month', 'weekday']]

    cams = pollution_data.iloc[:, 0]
    
    if downFlag:
        trainData_model_path = '../out/model/cams_pred_model/'
        # print(trainData_model_path + cams_model_name + pollution + '_trainedModel.joblib')
        if pollution == 'NOX':
            regr_cams_trainData = load(trainData_model_path + pollution + 'lgbexp_trainedModel.joblib')
        elif pollution == 'NO2':
            regr_cams_trainData = load(trainData_model_path + pollution + 'gbrsquare_trainedModel.joblib')
        elif pollution == 'PM10':
            regr_cams_trainData = load(trainData_model_path + pollution + 'lgbsquare_trainedModel.joblib')
        elif pollution == 'PM2.5':
            regr_cams_trainData = load(trainData_model_path + pollution + 'gbrsquare_trainedModel.joblib')
        others = pollution_data.iloc[:, 1:]
        cams_XPred = pollution_dataPred.iloc[:, :]
        _, _, cams_scaler = camstrain.train_data(pollution, describeFlag=False)
        cams_X_pred_transformed = cams_scaler.transform(cams_XPred)
        cams_Y_pred = regr_cams_trainData.predict(cams_X_pred_transformed)
        cams_pred = np.multiply(cams, cams_Y_pred)
        cams_pred = np.array(cams_pred).reshape(112041, 1)
        return cams_pred, others
    else:
        camsnp = np.array(pollution_data.iloc[:, 0]).reshape(112041, 1)
        return camsnp

def get_pnc_process_hour_data(args):
    '''
    description: get ../dataset/trainpredata/allbin/PNCtrainbin/
    param {*} args
    return {*}pred_out_pnc
    '''
    # hourId, cams_model_name, pnc_model_name = args
    hourId, pnc_model_name = args
    pnc_x_all_scaler, _, _, _, _ = get_pnc_scaler()
    pnc_regr = load_pnc_joblib(pnc_model_name)
    noxdata = np.fromfile('../dataset/trainpredata/allbin/NOXtrainbin/' + str(hourId) + '_NOX_predData.bin', dtype=np.float32)
    no2data = np.fromfile('../dataset/trainpredata/allbin/NO2trainbin/' + str(hourId) + '_NO2_predData.bin', dtype=np.float32)
    pm10data = np.fromfile('../dataset/trainpredata/allbin/PM10trainbin/' + str(hourId) + '_PM10_predData.bin', dtype=np.float32)
    pm25data = np.fromfile('../dataset/trainpredata/allbin/PM2.5trainbin/' + str(hourId) + '_PM2.5_predData.bin', dtype=np.float32)
    o3data = np.fromfile('../dataset/trainpredata/allbin/O3trainbin/' + str(hourId) + '_O3_predData.bin', dtype=np.float32)
    # prednox, others = cams_train_data(noxdata, cams_model_name, pollution='NOX', downFlag=True)
    # predno2, _ = cams_train_data(no2data, cams_model_name, pollution='NOX', downFlag=True)
    # camspm10 = cams_train_data(pm10data, cams_model_name, pollution='PM10', downFlag=False)
    # camspm25 = cams_train_data(pm25data, cams_model_name, pollution='PM2.5', downFlag=False)
    # camso3 = cams_train_data(o3data, cams_model_name, pollution='O3', downFlag=False)
    prednox, others = cams_train_data(noxdata, pollution='NOX', downFlag=True)
    predno2, _ = cams_train_data(no2data, pollution='NO2', downFlag=True)
    camspm10 = cams_train_data(pm10data, pollution='PM10', downFlag=False)
    camspm25 = cams_train_data(pm25data, pollution='PM2.5', downFlag=False)
    camso3 = cams_train_data(o3data, pollution='O3', downFlag=False)
    pred_pnc = np.concatenate((prednox, predno2, camspm10, camspm25, camso3, others), axis=1)
    pred_pnc = pd.DataFrame(pred_pnc)
    pred_pnc.columns = ['NOX [ug/m3 eq. NO2]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]', 'O3 [ug/m3]', 'Radiation[W/m2]', 'Temperature',
                        'Precipitation[mm]', 'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']
    pred_pnc['PM2.5/PM10 ratio'] = pred_pnc['PM2.5 [ug/m3]'] / pred_pnc['PM10 [ug/m3]']
    pred_pnc = pred_pnc[['NOX [ug/m3 eq. NO2]' ,'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]', 'Radiation[W/m2]', 'Temperature',
                        'Precipitation[mm]', 'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']]
    pred_pnc.fillna(0, inplace=True)
    pred_pnc = np.array(pred_pnc)
    x_pred_pnc = pnc_x_all_scaler.transform(pred_pnc)
    pred_out_pnc = pnc_regr.predict(x_pred_pnc)
    pred_out_pnc = np.array(pred_out_pnc).reshape(112041, 1)
    return pred_out_pnc


def get_matdata_pnc(pnc_model_name:str='RandomForest', part_num:int=5):
    '''
    description: 
    param {str} cams_model_name
    param {str} pnc_model_name
    return {*}
    '''

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

    avg_result = np.zeros((112041, part_num))
    for m in range(part_num):
        avgfilename = '../dataset/trainpredata/allbin/ave2020/PNC/PNCtest/' + '2020_downScale_PNC_' + \
            pnc_model_name + '_part' + str(m) + '.csv'
        avgtemp = pd.read_csv(avgfilename, header=None)
        avgtemp = np.array(avgtemp).reshape(-1, )
        avg_result[:, m] = avgtemp
    avgnox = np.mean(avg_result, axis=1).reshape(112041, 1)
    # temp= pd.read_csv(avgnoxfilename, header=None)
    value = np.array(avgnox).reshape(531, 211).T


    avgConc = {
        'lonNew': np.array(lonNew),
        'latNew': np.array(latNew),
        'avgConc': np.array(value),
    }
    year = 2020
    filename = '../pncEstimator-main/src/postProcessing/matdata/PNCtest/'+ str(year) + 'PNC_avgConc_'+ pnc_model_name + '.mat'
    sio.savemat(filename, avgConc)


def main():
    n = 8760
    part_num = 4
    temp_n = n // part_num

    year = 2020
    pnc_model_list = ['Stacking']
    cost_times = np.zeros((1, len(pnc_model_list)))
    for j in range(len(pnc_model_list)):
        print('pnc pres model: ', pnc_model_list[j])
        for m in range(part_num):
            start_num = int(m * temp_n)
            end_num = int((m+1) * temp_n)
            params = itertools.product(range(start_num, end_num), [pnc_model_list[j]])
            start_time = time.time()
            # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            #         results = list(tqdm(executor.map(get_pnc_process_hour_data, params), total=temp_n, desc="Processing hours"))
            with multiprocessing.Pool(processes=20) as pool:
                results = list(tqdm(pool.imap(get_pnc_process_hour_data, params), total=temp_n, 
                                    desc="Processing hours"))


            # 1. When using multiprocessing on Windows, parallel code must be placed under the protection of 
            # if __name__ == '__main__':, otherwise you may encounter problems with recursive creation processes.
            # with multiprocessing.Pool(processes=30) as pool:
            #     results = list(tqdm(pool.imap(get_pnc_process_hour_data, params), total=n, desc="Processing hours"))

            average_value = np.array(sum(results) / temp_n)
            print('The shape of average pnc data: ', average_value.shape)
            np.savetxt('../dataset/trainpredata/allbin/ave2020/PNC/PNCtest/' + str(year) + \
                       '_downScale_PNC_' + pnc_model_list[j] + '_part' + str(m) + '.csv', average_value, delimiter=',')

        get_matdata_pnc(pnc_model_name=pnc_model_list[j], part_num = part_num)
        end_time = time.time()
        cost_time = end_time - start_time
        cost_times[0, j] = cost_time
        print('time cost : %.5f sec' % cost_time)
            
    cost_times = pd.DataFrame(cost_times)
    cost_times.index = pnc_model_list
    dfi.export(obj = cost_times, filename='../out/figure/pnc_cost_times1129.png', fontsize = 14)


if __name__ == "__main__":
    main()

