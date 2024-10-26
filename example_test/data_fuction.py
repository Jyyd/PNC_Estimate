from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os, sys
os.chdir(sys.path[0]) # relative path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import time
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from joblib import dump, load
import netCDF4 as nc
import scipy.io as sio
from tqdm import trange

from sklearn.linear_model import LinearRegression, Lasso
from sklearn import neighbors, svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor


def get_train_test_data(feature_data_2016_2019, feature_data_2020):
    x_train = feature_data_2016_2019.iloc[:,3:].values
    y_train = feature_data_2016_2019.iloc[:,2].values
    x_test = feature_data_2020.iloc[:,3:].values
    y_test = feature_data_2020.iloc[:,2].values
    print('The train dataset ratio = ', y_train.shape[0]/ (y_test.shape[0]+y_train.shape[0]))
    print('The test dataset ratio = ', y_test.shape[0]/ (y_test.shape[0]+y_train.shape[0]))
    scaler_std = StandardScaler()
    x_all = np.vstack((x_train, x_test))
    print(x_all.shape, x_train.shape[0], x_test.shape[0])
    x_train = scaler_std.fit_transform(x_train)
    x_test = scaler_std.transform(x_test)
    return x_train, y_train, x_test, y_test, scaler_std


class ModelEvaluator():    
    def __init__(self):
        self.evsall = []
        self.maeall = []
        self.mseall = []
        self.r2all = []


    def pred_plot(self, y_test, y_pred):
        print('----------------------------------')
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        print('MSE: ', mse)
        print('MAE: ', mae)
        print('r2 score: ', r2)
        print('Explained_variance: ', evs)
        return mse, mae, r2, evs

    def out_table(self, mse, mae, r2, evs):
        self.evsall.append(evs)
        self.maeall.append(mae)
        self.mseall.append(mse)
        self.r2all.append(r2)

    def predpnc(self, model_fit, x_test, y_test, pncdata2020):
        pred = model_fit.predict(x_test)
        resid = pred - y_test
        mse, mae, r2, evs = self.pred_plot(y_test, pred)
        self.out_table(mse, mae, r2, evs)
        pred = pd.DataFrame(pred, columns=['Prediction'])
        resid = pd.DataFrame(resid, columns=['Residual'])
        data2020_pred = pd.concat([pncdata2020, pred, resid], axis=1)
        return data2020_pred

    def table_result(self):
        evsall = pd.DataFrame(self.evsall, columns=['evs'])
        maeall = pd.DataFrame(self.maeall, columns=['mae'])
        mseall = pd.DataFrame(self.mseall, columns=['mse'])
        r2all = pd.DataFrame(self.r2all, columns=['r2'])
        table_result = pd.concat([evsall, maeall, mseall, r2all], axis=1)
        return table_result

def grid_search(x_train, y_train):
    # this function is used to find the best hyperparameters for the model
    param_grid = [{'alpha':[0.1, 1, 10, 25, 50, 80, 100, 500, 1000]}]
    lasso_reg = Lasso()
    grid_search = GridSearchCV(lasso_reg, param_grid, cv=5)
    grid_result = grid_search.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    print('-------------------------------------')
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))

def get_model_reg(x_train, y_train, n_jobs_nums=2):
    # this function is used to train the model, and return the trained model
    # you can change the hyperparameters in the model
    linear_reg = LinearRegression().fit(x_train, y_train)
    lasso_reg = Lasso(alpha=80).fit(x_train, y_train)
    svm_reg = svm.SVR(C=100, gamma=0.5).fit(x_train, y_train)
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=n_jobs_nums).fit(x_train, y_train)
    tree_reg = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20, max_features=4,
                                   bootstrap=False, n_jobs=n_jobs_nums).fit(x_train, y_train)
    ada_reg = AdaBoostRegressor(random_state=42, n_estimators=50, learning_rate=0.05, loss='exponential').fit(x_train, y_train)
    gbr_reg = GradientBoostingRegressor(n_estimators=200, random_state=42, learning_rate=0.1,
                                        max_depth=4, min_samples_leaf=2, min_samples_split=6,
                                        subsample=0.8).fit(x_train, y_train)
    lgb_reg = lgb.LGBMRegressor(random_state=42, n_estimators=60, num_leaves=61, objective='regression',
                                n_jobs=n_jobs_nums).fit(x_train, y_train)
    return linear_reg, lasso_reg, svm_reg, knn_reg, tree_reg, rf_reg, ada_reg, gbr_reg, lgb_reg


def save_model_results(linear_reg, lasso_reg, svm_reg, knn_reg, tree_reg, rf_reg, ada_reg, gbr_reg, lgb_reg,
                       x_test, y_test, pncdata2020):
    # this function is used to save the prediction results
    evaluator = ModelEvaluator()
    data2020_pred1 = evaluator.predpnc(linear_reg, x_test, y_test, pncdata2020)
    data2020_pred2 = evaluator.predpnc(lasso_reg, x_test, y_test, data2020_pred1)
    data2020_pred3 = evaluator.predpnc(svm_reg, x_test, y_test, data2020_pred2)
    data2020_pred4 = evaluator.predpnc(knn_reg, x_test, y_test, data2020_pred3)
    data2020_pred5 = evaluator.predpnc(tree_reg, x_test, y_test, data2020_pred4)
    data2020_pred6 = evaluator.predpnc(rf_reg, x_test, y_test, data2020_pred5)
    data2020_pred7 = evaluator.predpnc(ada_reg, x_test, y_test, data2020_pred6)
    data2020_pred8 = evaluator.predpnc(gbr_reg, x_test, y_test, data2020_pred7)
    data2020_pred9 = evaluator.predpnc(lgb_reg, x_test, y_test, data2020_pred8)
    data2020_pred9.columns = ['Date/time', 'station', 'CPC [1/cm3]', 'NOX [ug/m3 eq. NO2]',
                            'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]',
                            'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]',
                            'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour',
                            'month', 'weekday', 'linear_pred','linear_resid','lasso_pred','lasso_resid',
                            'svm_pred','svm_resid','knn_pred','knn_resid','tree_pred','tree_resid',
                            'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
                            'lgb_pred','lgb_resid']
    pred_table_path = './out/pred_table/'
    if not os.path.exists(pred_table_path):
        os.makedirs(pred_table_path)
    # data2020_pred9.to_csv(pred_table_path + 'pncdata2020_pred_test.csv')

def get_stacking_results(linear_reg, lasso_reg, svm_reg, knn_reg, tree_reg,
                         rf_reg, ada_reg, gbr_reg, lgb_reg,
                         x_train, y_train, x_test, y_test, pncdata2020):
    ## 1. use the local model
    estimators = [("rf", rf_reg),("lgb", lgb_reg),("gbr",gbr_reg)]

    ## 2. use the model from our trained model
    # if you use model on https://huggingface.co/jyyd23/PNC_Estimate/blob/main/stack_trainedModel.joblib
    # you can use the following code
    # estimators = [("knn", knn_reg),("rf", rf_reg),("lgb", lgb_reg),("tree", tree_reg)]

    final_estimator = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', 
                                   solver='adam', random_state=42, max_iter=100)

    stack_reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
    stack_pred = stack_reg.fit(x_train, y_train).predict(x_test)
    resid = stack_pred - y_test
    evaluator = ModelEvaluator()
    # mse, mae, r2, evs = evaluator.pred_plot(y_test, stack_pred, resid)

    stack_fit = stack_reg.fit(x_train, y_train)
    data2020_pred16 = evaluator.predpnc(stack_fit, x_test, y_test, pncdata2020)
    other_pred = pd.read_csv('./out/pred_table/pncdata2020_pred_test.csv')
    select_other_pred = other_pred[['linear_pred','linear_resid','lasso_pred','lasso_resid',
                            'svm_pred','svm_resid','knn_pred','knn_resid','tree_pred','tree_resid',
                            'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
                            'lgb_pred','lgb_resid']]
    data2020_pred16 = pd.concat([data2020_pred16, select_other_pred], axis=1)
    data2020_pred16.columns = ['Date/time', 'station', 'CPC [1/cm3]', 'NOX [ug/m3 eq. NO2]',
                            'PM10 [ug/m3]', 'PM2.5/PM10 ratio', 'O3 [ug/m3]',
                            'Radiation[W/m2]', 'Temperature', 'Precipitation[mm]',
                            'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour',
                            'month', 'weekday', 'linear_pred','linear_resid','lasso_pred','lasso_resid',
                            'svm_pred','svm_resid','knn_pred','knn_resid','tree_pred','tree_resid',
                            'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
                            'lgb_pred','lgb_resid', 'stack_pred','stack_resid']
    pred_table_path = './out/pred_table/'
    data2020_pred16.to_csv(pred_table_path + '/pncdata2020_pred.csv')
    pnc_model_folder = './out/pnc_model/'
    if not os.path.exists(pnc_model_folder):
        os.makedirs(pnc_model_folder)
    dump(stack_fit, pnc_model_folder + '/stack_trainedModel.joblib')



def cams_load_trainData(pollution:str='NOX'):
    trainpath = './dataset/trainData/'+ pollution + '_trainData.csv'
    traindata = pd.read_csv(trainpath)
    traindata = traindata.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    return traindata


def cams_train_data(pollution_data, pollution:str='NOX'):
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
    others = pollution_data.iloc[:, 1:]
    camsnp = np.array(pollution_data.iloc[:, 0]).reshape(112041, 1)
    return camsnp, others

def get_pnc_process_hour_data(args):

    # hourId, cams_model_name, pnc_model_name = args
    hourId, pnc_x_all_scaler = args
    ## 1.use the local one
    # use the trained model to predict the pnc data
    pnc_regr = load('./out/pnc_model/stack_trainedModel.joblib')

    ## 2.use the trained model from our trained model
    # you can use the trained model to predict the pnc data from
    # https://huggingface.co/jyyd23/PNC_Estimate/blob/main/stack_trainedModel.joblib
    # use your download path to load the model
    # pnc_regr = load('../../../mypred/out/model/stack_trainedModel.joblib')

    noxdata = np.fromfile('./dataset/allbin/NOXtrainbin/' + str(hourId) + '_NOX_predData.bin', dtype=np.float32)
    no2data = np.fromfile('./dataset/allbin/NO2trainbin/' + str(hourId) + '_NO2_predData.bin', dtype=np.float32)
    pm10data = np.fromfile('./dataset/allbin/PM10trainbin/' + str(hourId) + '_PM10_predData.bin', dtype=np.float32)
    pm25data = np.fromfile('./dataset/allbin/PM2.5trainbin/' + str(hourId) + '_PM2.5_predData.bin', dtype=np.float32)
    o3data = np.fromfile('./dataset/allbin/O3trainbin/' + str(hourId) + '_O3_predData.bin', dtype=np.float32)
    prednox, others = cams_train_data(noxdata, pollution='NOX')
    predno2, _ = cams_train_data(no2data, pollution='NO2')
    camspm10, _  = cams_train_data(pm10data, pollution='PM10')
    camspm25, _  = cams_train_data(pm25data, pollution='PM2.5')
    camso3, _  = cams_train_data(o3data, pollution='O3')
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


def get_matdata_pnc(hourID):
    path1 = './dataset/allbin/PNC2020/'
    path2 = './dataset/allbin/PNC2020_mat/'
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)

    Delta_x = 0.01
    Delta_y = 0.01

    nf = nc.Dataset('./dataset/CAMS/SingleLevel_202101.nc')
    lon = np.array(nf.variables['longitude'][:]).reshape(-1, 1)
    lat = np.array(nf.variables['latitude'][:]).reshape(-1, 1)

    lonNew = np.arange(lon[0][0], lon[-1][0], Delta_x)
    latNew = np.arange(lat[0][0], lat[-1][0], -Delta_y)
    xnew, ynew = np.meshgrid(lonNew, latNew)
    # print(lonNew.shape)
    # print(latNew.shape)

    avg_result = np.zeros((112041, hourID))
    for m in range(hourID):
        avgfilename = f'./dataset/allbin/PNC2020/2020_downScale_PNC_stack_{m}.csv'
        avgtemp = pd.read_csv(avgfilename, header=None)
        avgtemp = np.array(avgtemp).reshape(-1, )
        avg_result[:, m] = avgtemp
    avgnox = np.mean(avg_result, axis=1).reshape(112041, 1)
    value = np.array(avgnox).reshape(531, 211).T

    avgConc = {
        'lonNew': np.array(lonNew),
        'latNew': np.array(latNew),
        'avgConc': np.array(value),
    }
    filename = f'./dataset/allbin/PNC2020_mat/2020_PNC_avgConc_stack.mat'
    sio.savemat(filename, avgConc)

def get_results_all(pnc_x_all_scaler):
    n = 24
    for hourId in trange(0, n):
        args = hourId, pnc_x_all_scaler
        results = get_pnc_process_hour_data(args)
        np.savetxt(f'./dataset/allbin/PNC2020/2020_downScale_PNC_stack_{hourId}.csv',
                   results, delimiter=',')
        get_matdata_pnc(hourId)
