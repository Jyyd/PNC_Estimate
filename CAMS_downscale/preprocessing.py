'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2023-11-25 14:24:43
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-03-07 13:54:35
FilePath: CAMS_downscale\preprocessing.py
Description: 

'''


import os, sys
os.chdir(sys.path[0])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
# import dataframe_image as dfi
import warnings

warnings.filterwarnings("ignore")
sns.set(rc={'figure.dpi': 300}) # set dpi

'''
The preprocessing.py functions are:
    1.create_directories(aim_folder_name:str)
        input: aim_folder_name
        output: None
        function: Creat the file.
    2.data_describe(data)
        input: data
        output: data_describe Tab
        function: Based on the data.describe() add 'skew' and 'kurt'.
    3.data_describe_plot(data, savepath:str, station:str)
        input: data, savepath:str, station:str
        output: None
        function: Plot the Histrogram, scatter, Autocorrelation and qq-plot of data, save the figure.
    4.corr_plot(data, figurename:str)
        input: data, figurename:str
        output: None
        function: Plot the spearman Heatmap of data, save the figure.
    5.get_NABEL_data()
        input: None
        output: NABELdata
        function: Load the NABEL data and add needed features.
    6.get_meteo_data()
        input: None
        output: meteodata
        function: Load the meoteo data and add needed features.
    7.merge_data()
        input: None
        output: pncdata, pncdata_washed_data
        function: Merge the NABELdata and meteodata, and wash the data.
    8.get_corrlist()
        input: None
        output: None
        function: Put all the Spearman's correlation coefficients of stations together and save them as a figure.
    9.get_feature_data()
        input: None
        output: None
        function: Divide the merge_data by date/time into two parts, and save the feature data.
    10.main()
        input: None
        output: None
        function: Run the needed functions.
'''


def create_directories(aim_folder_name:str):
    '''
    description: 
    param {str} aim_folder_name
    return {*}
    '''
    if not os.path.exists(aim_folder_name):
        os.makedirs(aim_folder_name)
    
    
def data_describe(data):
    # print('----------------------------')
    # print('The shape of data: ', data.shape)
    # print('----------------------------')
    # print('The data type and infos: ', data.info())
    # print('----------------------------')
    # print('The missing data: ', data.isnull().sum())
    # print('----------------------------')
    data_describe = data.describe()
    data_skew = pd.DataFrame(data.skew()).T
    data_kurt = pd.DataFrame(data.kurt()).T
    data_describe = pd.concat([data_describe, data_skew], axis=0)
    data_describe = pd.concat([data_describe, data_kurt], axis=0)
    data_describe.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt']
    # print('The describe infos of data: ', data_describe)
    return data_describe

def data_describe_plot(data, savepath:str, station:str):
    try:
        # plot the Histogram of data feature
        data = data.iloc[:, 2:]
        num_f = data.shape[1]
        # colname = data.columns
        colname = ['CPC', 'NOX', 'ratio_NO2_NOX', 'PM10', 'ratio_PM2_5_PM10', 'O3', 'Radiation', 'Temperature', 'Precipitation',
                   'Relative_Humidity', 'Wind_spped', 'trafficVol', 'hour', 'month', 'weekend']
        color = ['SteelBlue', 'olive', 'gold', 'teal', 'red', 'g', 'peru', 'DarkViolet', 'b',
                 'SteelBlue', 'olive', 'gold', 'teal', 'red', 'g', 'peru', 'DarkViolet', 'b']
        if len(color)<len(colname):
            print('The length of the color list needs to be more data!')
        else:
            print('OK!')

        for i in range(num_f):
            sns.histplot(data = data, x = data.iloc[:, i], color=color[i], kde=True, bins=20)
            # sns.histplot(data = data, x = data.iloc[:, i], color=color[i], kde=True)
            title_name = station + '_Histogram_of_' + colname[i]
            plt.title(title_name)
            # plt.savefig(savepath + title_name + '.png')
            # plt.show()

        # plot the scatter figure but may cost too much time
        # the scatter figure x:features, y:CPC
        # for i in range(num_f):
        #     plt.figure(figsize=(10, 2), dpi=300)
        #     plt.plot(data.iloc[:, 0], data.iloc[:, i+1], color=color[i])
        #     plt.scatter(data.iloc[:, 0], data.iloc[:, i+1], alpha=0.4, c='red', label=data.columns[i+1])
        #     plt.xlabel("Date")
        #     plt.ylabel(data.columns[i+1])
        #     plt.legend()
        #     title_name = station + '_Scatter line chart ' + colname[i+1].split(' ')[0]
        #     plt.title(title_name)
        #     plt.savefig(savepath + title_name + '.png')
        #    # plt.show()

        # 绘制时序自相关图
        # for i in range(num_f):
        #     title_name_au = station + "_Autocorrelation_" + colname[i]
        #     plot_acf(data.iloc[:, i], title=title_name_au)
        #     plt.figure(figsize=(12, 5))
        #     # plt.show()
        #     plt.savefig(savepath + title_name_au + '.png')

        # for i in range(num_f):
        #     sm.qqplot(data.iloc[:, i], line='s')
        #     title_name_qq = station + '_qq_' + colname[i]
        #     plt.title(title_name)
        #     # plt.show()
        #     plt.savefig(savepath + title_name_qq + '.png')
    except:
        print('Pay attention to the input format, the first column is y, and the features are put behind the first column!')

def corr_plot(data, figurename:str):  
    corr_data = data.corr(method='spearman')
    figure, ax = plt.subplots(figsize=(12, 12), dpi=300)
    sns.heatmap(corr_data, square=True, annot=True, ax=ax, cmap="RdBu_r")
    plt.savefig(figurename + '.png')
    
def get_NABEL_data():
    path_0 = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    path_1 = path_0 + '/pncEstimator-main'
    filePath = path_1 + "/dataset/NABEL/raw_data/"
    filelist = os.listdir(filePath)
    print('The number of NABEL stations: ', len(filelist))
    NABELdata = []
    for file in filelist:
        filename = filePath + file
        # print(filename)
        data = pd.read_csv(filename, skiprows=6, sep=';') # load data
        file_colname = data.columns
        if 'CPC [1/cm3]' in file_colname:
            data_pnc = data[['Date/time', 'CPC [1/cm3]', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]', 'NOX [ug/m3 eq. NO2]']]
            data_pnc['NO2/NOX ratio'] = data_pnc['NO2 [ug/m3]'] / data_pnc['NOX [ug/m3 eq. NO2]']
            data_pnc['PM2.5/PM10 ratio'] = data_pnc['PM2.5 [ug/m3]'] / data_pnc['PM10 [ug/m3]']
            station = [file[0:3]] * len(data_pnc)
            station = pd.DataFrame(station)
            data_pnc = pd.concat([data_pnc, station], axis=1)
            data_pnc.columns = ['Date/time', 'CPC [1/cm3]', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]',
                                'PM2.5 [ug/m3]', 'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio', 'PM2.5/PM10 ratio',
                                'station']
            data_pnc = data_pnc[['Date/time', 'station', 'CPC [1/cm3]', 'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                'PM10 [ug/m3]','PM2.5/PM10 ratio', 'O3 [ug/m3]']]
            NABELdata.append(data_pnc)
    NABELdata = pd.concat(NABELdata, axis=0)
    NABELdata = NABELdata.reset_index(drop=True)
    NABELdata['Date/time'] = pd.to_datetime(NABELdata['Date/time'], format='%d.%m.%Y %H:%M')
    print('++++++++++++++++++NABELdata++++++++++++++++++')
    print(NABELdata.head(2))
    return NABELdata

def get_NABEL_data_PM():
    filelist = os.listdir('../../PNC/code/pncEstimator-main/data/NABEL/raw_data/')
    print('The number of NABEL stations: ', len(filelist))
    NABELdata = []
    for file in filelist:
        filename = '../../PNC/code/pncEstimator-main/data/NABEL/raw_data/' + file
        if filename.endswith('.csv'):
            # print(filename)
            data = pd.read_csv(filename, skiprows=6, sep=';') # load data
            file_colname = data.columns
            if 'PM2.5 [ug/m3]' in file_colname:
                data_pnc = data[['Date/time', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]', 'NOX [ug/m3 eq. NO2]']]
                data_pnc['NO2/NOX ratio'] = data_pnc['NO2 [ug/m3]'] / data_pnc['NOX [ug/m3 eq. NO2]']
                data_pnc['PM2.5/PM10 ratio'] = data_pnc['PM2.5 [ug/m3]'] / data_pnc['PM10 [ug/m3]']
                station = [file[0:3]] * len(data_pnc)
                station = pd.DataFrame(station)
                data_pnc = pd.concat([data_pnc, station], axis=1)
                data_pnc.columns = ['Date/time', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]',
                                    'PM2.5 [ug/m3]', 'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio', 'PM2.5/PM10 ratio',
                                    'station']
                data_pnc = data_pnc[['Date/time', 'station', 'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                    'PM10 [ug/m3]','PM2.5/PM10 ratio', 'O3 [ug/m3]']]
                NABELdata.append(data_pnc)
    NABELdata = pd.concat(NABELdata, axis=0)
    NABELdata = NABELdata.reset_index(drop=True)
    NABELdata['Date/time'] = pd.to_datetime(NABELdata['Date/time'], format='%d.%m.%Y %H:%M')
    print('++++++++++++++++++NABELdata++++++++++++++++++')
    print(NABELdata.head(2))
    return NABELdata

def get_meteo_data():
    path_0 = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    path_1 = path_0 + '/pncEstimator-main'
    meteopath = path_1 + '/dataset/meteo/meteo/'
    meteodata = pd.read_csv(meteopath + 'meteodata.txt', sep=';', dtype={1: str})
    meteodata.columns = ['stn', 'time', 'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]', 'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol']
    meteodata['time'] = meteodata['time'].str.slice(0, 4) + '-' + meteodata['time'].str.slice(4, 6) + \
                        '-' + meteodata['time'].str.slice(6, 8) + ' ' + meteodata['time'].str.slice(8, 10)+':00'
    meteodata['stn'] = meteodata['stn'].replace({'NABRIG': 'RIG'})
    meteodata['stn'] = meteodata['stn'].replace({'NABBER': 'BER'})
    meteodata['stn'] = meteodata['stn'].replace({'NABHAE': 'HAE'})
    meteodata['time'] = pd.to_datetime(meteodata['time'], format='%Y-%m-%d %H',errors='ignore')
    print('++++++++++++++++++meteodata++++++++++++++++++')
    print(meteodata.head(2))
    return meteodata

def merge_data():
    NABELdata = get_NABEL_data()
    meteodata = get_meteo_data()
    pncdata = pd.merge(NABELdata, meteodata, how='inner', left_on=['station', 'Date/time'], right_on=['stn', 'time'])
    pncdata = pncdata[['Date/time', 'station', 'CPC [1/cm3]', 'NOX [ug/m3 eq. NO2]',
                       'NO2/NOX ratio', 'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]',
                       'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]',
                       'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol']]
    pncdata['hour'] = pncdata['Date/time'].dt.hour
    pncdata['month'] = pncdata['Date/time'].dt.month
    pncdata['weekday'] = pncdata['Date/time'].dt.dayofweek
    pncdata = pncdata[(pncdata['Date/time']>='2016-01-01 01:00')&(pncdata['Date/time']<'2021-01-01 01:00')]
    pncdata = pncdata.reset_index(drop=True)
    data_describe_figure_name = '/out/figure/data_describe/Tab/'
    create_directories(data_describe_figure_name)

    pncdata_data_describe = data_describe(pncdata)
    dfi.export(obj = pncdata_data_describe, filename=data_describe_figure_name + 'Tab_pnc_merge_data_describe.png', fontsize = 14)

    pncdata_washed = pncdata.dropna(axis=0, how='any')
    pncdata_washed = pncdata_washed.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    pncdata_washed_data_describe = data_describe(pncdata)
    dfi.export(obj = pncdata_washed_data_describe, filename=data_describe_figure_name + 'Tab_pnc_washed_merge_data_describe.png', fontsize = 14)
    return pncdata, pncdata_washed

def merge_data_PM():
    NABELdata = get_NABEL_data_PM()
    meteodata = get_meteo_data()
    pncdata = pd.merge(NABELdata, meteodata, how='inner', left_on=['station', 'Date/time'], right_on=['stn', 'time'])
    pncdata = pncdata[['Date/time', 'station', 'NOX [ug/m3 eq. NO2]',
                       'NO2/NOX ratio', 'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]',
                       'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]',
                       'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol']]
    pncdata['hour'] = pncdata['Date/time'].dt.hour
    pncdata['month'] = pncdata['Date/time'].dt.month
    pncdata['weekday'] = pncdata['Date/time'].dt.dayofweek
    pncdata = pncdata[(pncdata['Date/time']>='2016-01-01 01:00')&(pncdata['Date/time']<'2021-01-01 01:00')]
    pncdata = pncdata.reset_index(drop=True)
    data_describe_figure_name = '/out/figure/data_describe/Tab/'
    create_directories(data_describe_figure_name)

    pncdata_data_describe = data_describe(pncdata)
    dfi.export(obj = pncdata_data_describe, filename=data_describe_figure_name + 'Tab_pnc_merge_data_describe.png', fontsize = 14)

    pncdata_washed = pncdata.dropna(axis=0, how='any')
    pncdata_washed = pncdata_washed.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    pncdata_washed_data_describe = data_describe(pncdata)
    dfi.export(obj = pncdata_washed_data_describe, filename=data_describe_figure_name + 'Tab_pnc_washed_merge_data_describe.png', fontsize = 14)
    return pncdata, pncdata_washed

def get_corrlist():
    corr_station_figure_name = '/out/figure/NABEL_station_heatmap/'
    create_directories(corr_station_figure_name)
    data_describe_tab_figure_name = '/out/figure/data_describe/Tab/'
    create_directories(data_describe_tab_figure_name)
    corrlist = []
    pncdata, pncdata_washed = merge_data()
    stations = ['BAS', 'BER', 'HAE', 'LUG', 'RIG']
    for station in stations:
        temp = pncdata_washed[pncdata_washed['station'] == station]
        station_describe = data_describe(temp)
        dfi.export(obj = station_describe, filename=data_describe_tab_figure_name + 'Tab_'+ station +'_data_describe.png', fontsize = 14)
        # data_describe_figure_name = '../out/figure/data_describe/' + station + '/'
        # create_directories(data_describe_figure_name)
        # data_describe_plot(temp, data_describe_figure_name, station)
        corr_plot(temp, corr_station_figure_name + station)
        corr = temp.corr(method='spearman')
        corrlist.append(corr.iloc[0, :])
    corrtable = pd.DataFrame(corrlist)
    corrtable.index = ['Basel_Binningen','Bern_Bollwerk','Harkingen_A1','Lugano_Universita','Rigi_Seebodenalp']
    dfi.export(obj = corrtable, filename=corr_station_figure_name + 'Tab_station_corrlist.png', fontsize = 14)
    
def get_feature_data():
    path_0 = os.path.abspath(os.path.join(os.getcwd(),"../.."))
    path_1 = path_0 + '/pncEstimator-main'
    pncdata, pncdata_washed = merge_data()
    pncdata2016_2019 = pncdata_washed[(pncdata_washed['Date/time']>='2016-01-01 01:00')&(pncdata_washed['Date/time']<'2020-01-01 01:00')]
    pncdata2020 = pncdata_washed[(pncdata_washed['Date/time']>='2020-01-01 01:00')&(pncdata_washed['Date/time']<'2021-01-01 01:00')]
    pncdata2016_2019 = pncdata2016_2019.reset_index(drop=True)
    pncdata2020 = pncdata2020.reset_index(drop=True)
    pncdata2016_2019.to_csv(path_1 + '/dataset/NABEL/feature_data/feature_data_2016_2019.csv')
    pncdata2020.to_csv(path_1 + '/dataset/NABEL/feature_data/feature_data_2020.csv')
    return pncdata2016_2019, pncdata2020

def main():
    get_corrlist()
    get_feature_data()
    
if __name__ == "__main__":
    main()