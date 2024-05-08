'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-04-30 17:32:22
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-08 20:23:25
FilePath: \finalcode\pop_PNC\pop_data_processing.py
Description: 

'''

import os, sys
os.chdir(sys.path[0])
import rasterio
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import griddata
import geopandas as gpd
from rasterio import features
from affine import Affine
from matplotlib.path import Path
from scipy.io import savemat, loadmat
from scipy import interpolate
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

'''
This file has functions:
1. load_pnc_annual_data: Load PNC data
2. load_pm_data: Load PM data
3. load_pnc_hourly_data: Load PNC hourly data
4. get_swiss_pop_data: Load population data
5. cal_area: Calculate area
6. find_nearest: Find nearest value
7. calculate_overlap_area: Calculate overlap area
8. calculate_newpop: Calculate new population
9. load_mask_pop_data: Load population data
10. print_pop_data_details: Print population data details
11. get_pop_with_pnc: Get population data with PNC
12. get_pop_with_pnc_hourly: Get population data with PNC hourly
13. save_tiff: Save tiff file
14. get_annual_tiff_file: Get annual tiff file
'''

def load_pnc_annual_data():

    # load PNC data
    pncmat_data = loadmat('../../pncEstimator-main/src/postProcessing/matdata/PNCtest/0102/2020PNC_avgConc_Stacking.mat')
    matlon = pncmat_data['lonNew'][0][:]
    matlat = pncmat_data['latNew'][0][:]
    avgPNC = pncmat_data['avgConc']
    matlon = np.append(matlon, [matlon[-1] + 0.01])
    matlat = np.append(matlat, [matlat[-1] + 0.01])
    matlat = matlat[::-1]

    return matlon, matlat, avgPNC, pncmat_data

def load_pm_data(pm_type:str='PM10'):
    if pm_type == 'PM10':
        file_path = '../../pncEstimator-main/src/postProcessing/matdata/PM/2020PM10_avgConc_test.mat'
    elif pm_type == 'PM2.5':
        file_path = '../../pncEstimator-main/src/postProcessing/matdata/PM/2020PM25_avgConc_test.mat'
    else:
        raise ValueError('Invalid PM type. Please choose between PM10 and PM2.5')

    pmmat_data = loadmat(file_path)
    matlon = pmmat_data['lonNew'][0][:]
    matlat = pmmat_data['latNew'][0][:]
    avgPM = pmmat_data['avgConc']
    matlon = np.append(matlon, [matlon[-1] + 0.01])
    matlat = np.append(matlat, [matlat[-1] + 0.01])
    matlat = matlat[::-1]

    return matlon, matlat, avgPM, pmmat_data

def load_pnc_hourly_data(pnc_hourly_filename:str):
    pncmat_data = loadmat(pnc_hourly_filename)
    matlon = pncmat_data['lonNew'][0][:]
    matlat = pncmat_data['latNew'][0][:]
    avgPNC = pncmat_data['avgConc']
    matlon = np.append(matlon, [matlon[-1] + 0.01])
    matlat = np.append(matlat, [matlat[-1] + 0.01])
    matlat = matlat[::-1]

    return matlon, matlat, avgPNC, pncmat_data


def get_swiss_pop_data(pop_filename):
    # load pop data
    popData = rasterio.open(pop_filename)
    pop = popData.read(1)
    popCoord = popData.transform
    cellLon = popCoord[0]
    cellLat = -popCoord[4]
    lonPop = np.arange(popCoord[2] + cellLon / 2, popCoord[0], cellLon/2)
    latPop = np.arange(popCoord[5] - cellLat / 2, popCoord[1], -cellLat/2)
    origin_lon = lonPop.reshape(-1, 1).T
    origin_lat = latPop.reshape(-1, 1).T
    origin_pop = pop
    # transform coordinates of pop
    def convert_coordinates(lon_array, lat_array):
        lon_array_converted = lon_array*2 + 180
        lat_array_converted = lat_array*2 - 90
        return lon_array_converted, lat_array_converted
    trans_origin_lon, trans_origin_lat = convert_coordinates(origin_lon, origin_lat)

    # select rectangular area around swiss
    lonfield = [5.9, 10.6]
    latfield = [45.75, 47.85]
    lon_indices = np.where((trans_origin_lon >= lonfield[0]) & (trans_origin_lon <= lonfield[1]))[1]
    lat_indices = np.where((trans_origin_lat >= latfield[0]) & (trans_origin_lat <= latfield[1]))[1]
    select_lon = trans_origin_lon[:, lon_indices]
    select_lat = trans_origin_lat[:, lat_indices]
    select_pop = origin_pop[np.ix_(lat_indices, lon_indices)] # use index select area
    
    # select the pop data in swiss
    switzerland = gpd.read_file('../../pncEstimator-main/data/geoshp/gadm36_CHE_0.shp')
    switzerland_polygon = switzerland.unary_union
    paths = []
    if switzerland_polygon.geom_type == 'MultiPolygon':
        for polygon in switzerland_polygon.geoms:
            paths.append(Path(np.array(polygon.exterior.coords)))
    else:
        paths.append(Path(np.array(switzerland_polygon.exterior.coords)))

    select_lon_grid, select_lat_grid = np.meshgrid(select_lon, select_lat)
    select_lon_lat_points = np.vstack((select_lon_grid.flatten(), select_lat_grid.flatten())).T
    select_inside_swiss_mask = np.zeros(len(select_lon_lat_points), dtype=bool)
    for path in paths:
        select_inside_swiss_mask |= path.contains_points(select_lon_lat_points)
    select_inside_swiss_mask = select_inside_swiss_mask.reshape(select_pop.shape)
    inside_swiss_pop = np.where(select_inside_swiss_mask, select_pop, 0)
    select_lat = select_lat.flatten()
    select_lon = select_lon.flatten()
    select_lat = select_lat[::-1]
    
    return select_lon, select_lat, inside_swiss_pop

def cal_area(lat, lon):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        # make sure the input is in radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        distance = R * c
        return distance
    area_matrix = np.zeros((len(lat), len(lon)))
    for i in range(len(lat) - 1):
        for j in range(len(lon) - 1):
            x_distance = haversine(lat[i], lon[j], lat[i], lon[j + 1])
            y_distance = haversine(lat[i], lon[j], lat[i + 1], lon[j])
            area_matrix[i, j] = x_distance * y_distance
    # print(area_matrix.shape)
    return area_matrix

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def calculate_overlap_area(pnc_lon, pnc_lat, pop_lon, pop_lat):
    pop_half_grid = 0.04167 / 2 # degree
    pnc_half_grid = 0.01 / 2 # degree

    left = max(pnc_lon - pnc_half_grid, pop_lon - pop_half_grid)
    right = min(pnc_lon + pnc_half_grid, pop_lon + pop_half_grid)
    bottom = max(pnc_lat - pnc_half_grid, pop_lat - pop_half_grid)
    top = min(pnc_lat + pnc_half_grid, pop_lat + pop_half_grid)

    overlap_width_lon = max(0, right - left)
    overlap_height_lat = max(0, top - bottom)

    return overlap_width_lon * overlap_height_lat # degree^2
    

def calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens, poplat_dens, pop_data_dens, adjust_ratio):
    area_matrix_pnc = cal_area(pnclat, pnclon)
    pnc_area = area_matrix_pnc.mean() # km^2
    diff_pnc = 0.01*0.01 # degree^2
    newpop_values = np.zeros(pnc_data.shape)
    temp = []
    for i, pnc_lat in enumerate(pnclat):
        for j, pnc_lon in enumerate(pnclon):

            lon_idx, _ = find_nearest(poplon_dens, pnc_lon)
            lat_idx, _ = find_nearest(poplat_dens, pnc_lat)

            total_weighted_value = 0
            total_overlap_area = 0

            for di in [0, 1]:
                for dj in [0, 1]:
                    pop_i = lat_idx + di - 1
                    pop_j = lon_idx + dj - 1
                    if 0 <= pop_i < pop_data_dens.shape[0] and 0 <= pop_j < pop_data_dens.shape[1]:
                    # if 0 <= pop_i < len(poplat_dens) and 0 <= pop_j < len(poplon_dens):
                        overlap_area = calculate_overlap_area(pnc_lon, pnc_lat, poplon_dens[pop_j], poplat_dens[pop_i])
                        overlap_area = overlap_area * (pnc_area/diff_pnc)
                        
                        if overlap_area > 0:
                            weight = overlap_area
                            weighted_pop = pop_data_dens[pop_i, pop_j] * overlap_area * adjust_ratio
                            total_weighted_value += weighted_pop
                            total_overlap_area += overlap_area
            
            if total_overlap_area > 0.4:
                temp.append(total_overlap_area)
                newpop_values[i, j] = total_weighted_value / total_overlap_area * adjust_ratio
            else:
                newpop_values[i, j] = 0
    return newpop_values, total_overlap_area, temp

def load_mask_pop_data():
    # # density pop
    # total pop
    (
        poplon_dens_bt,
        poplat_dens_bt,
        pop_data_dens_bt
    ) = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopbt_2010_dens_2pt5_min.tif')
    # male pop
    (
        poplon_dens_mt,
        poplat_dens_mt,
        pop_data_dens_mt
    ) = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopmt_2010_dens_2pt5_min.tif')
    # famale pop
    (
        poplon_dens_ft,
        poplat_dens_ft,
        pop_data_dens_ft
    ) = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopft_2010_dens_2pt5_min.tif')
    # age group pop
    # 0-14
    (
        poplon_dens_bt_0_14,
        poplat_dens_bt_0_14,
        pop_data_dens_bt_0_14) = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a000_014bt_2010_dens_2pt5_min.tif')
    # 15-64
    (
        poplon_dens_bt_15_64,
        poplat_dens_bt_15_64,
        pop_data_dens_bt_15_64
    ) = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a015_064bt_2010_dens_2pt5_min.tif')
    # 65+
    (
        poplon_dens_bt_65,
        poplat_dens_bt_65,
        pop_data_dens_bt_65
    ) = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a065plusbt_2010_dens_2pt5_min.tif')
    
    # # conut pop
    # total pop
    poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopbt_2010_cntm_2pt5_min.tif')
    # male pop
    poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopmt_2010_cntm_2pt5_min.tif')
    # famale pop
    poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopft_2010_cntm_2pt5_min.tif')
    # age group pop
    # 0-14
    poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14 = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a000_014bt_2010_cntm_2pt5_min.tif')
    # 15-64
    poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64 = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a015_064bt_2010_cntm_2pt5_min.tif')
    # 65+
    poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65 = get_swiss_pop_data('../../pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a065plusbt_2010_cntm_2pt5_min.tif')

    return (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    )

def print_pop_data_details(print_details:bool=False, pollution_type:str='PNC'):

    pnclon, pnclat, pnc_data, pncmat_data = None, None, None, None
    (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    ) = load_mask_pop_data()

    valid_pollution_types = ['PNC', 'PM2.5', 'PM10']
    if pollution_type not in valid_pollution_types:
        raise ValueError(f"Invalid pollution_type. Expected one of: {valid_pollution_types}")
    elif pollution_type == 'PNC':
        pnclon, pnclat, pnc_data, pncmat_data = load_pnc_annual_data()
    elif pollution_type == 'PM2.5':
        pnclon, pnclat, pnc_data, pncmat_data = load_pm_data(pm_type='PM2.5')
    elif pollution_type == 'PM10':
        pnclon, pnclat, pnc_data, pncmat_data = load_pm_data(pm_type='PM10')

    area_matrix_pnc = cal_area(pnclat, pnclon)
    pnc_area = area_matrix_pnc.mean() # km^2
    diff_pnc = 0.01*0.01 # degree^2
    pop_grid = 0.04167
    diff_pop = pop_grid*pop_grid # degree^2
    # bt
    area_matrix_pop_bt = cal_area(poplat_dens_bt, poplon_dens_bt)
    pop_area_bt = area_matrix_pop_bt.mean() # km^2
    adjust_ratio_bt = np.sum(pop_data_cntm_bt)/np.sum(pop_data_dens_bt*pop_area_bt)
    # mt
    area_matrix_pop_mt = cal_area(poplat_dens_mt, poplon_dens_mt)
    pop_area_mt = area_matrix_pop_mt.mean() # km^2
    adjust_ratio_mt = np.sum(pop_data_cntm_mt)/np.sum(pop_data_dens_mt*pop_area_mt)
    # ft
    area_matrix_pop_ft = cal_area(poplat_dens_ft, poplon_dens_ft)
    pop_area_ft = area_matrix_pop_ft.mean() # km^2
    adjust_ratio_ft = np.sum(pop_data_cntm_ft)/np.sum(pop_data_dens_ft*pop_area_ft)
    # bt 0-14
    area_matrix_pop_bt_0_14 = cal_area(poplat_dens_bt_0_14, poplon_dens_bt_0_14)
    pop_area_bt_0_14 = area_matrix_pop_bt_0_14.mean() # km^2
    adjust_ratio_bt_0_14 = np.sum(pop_data_cntm_bt_0_14)/np.sum(pop_data_dens_bt_0_14*pop_area_bt_0_14)
    # bt 15-64
    area_matrix_pop_bt_15_64 = cal_area(poplat_dens_bt_15_64, poplon_dens_bt_15_64)
    pop_area_bt_15_64 = area_matrix_pop_bt_15_64.mean() # km^2
    adjust_ratio_bt_15_64 = np.sum(pop_data_cntm_bt_15_64)/np.sum(pop_data_dens_bt_15_64*pop_area_bt_15_64)
    # bt 65+
    area_matrix_pop_bt_65 = cal_area(poplat_dens_bt_65, poplon_dens_bt_65)
    pop_area_bt_65 = area_matrix_pop_bt_65.mean() # km^2
    adjust_ratio_bt_65 = np.sum(pop_data_cntm_bt_65)/np.sum(pop_data_dens_bt_65*pop_area_bt_65)

    if print_details == True:
        print('pnc mean area : ', pnc_area)
        print('pnc km^2/degree^2 : ', pnc_area/diff_pnc)
        print('--------------------------------')
        # bt
        print('pop_area_unit_bt: ', pop_area_bt)
        print('pop cntm bt: ', np.sum(pop_data_cntm_bt)) # people
        print('pop_data_dens_bt*pop_area_bt: ', np.sum(pop_data_dens_bt*pop_area_bt))
        print('adjust_ratio_bt: ', adjust_ratio_bt)
        print('diff_bt ratio(%) : ', (np.sum(pop_data_dens_bt*pop_area_bt)-np.sum(pop_data_cntm_bt))/np.sum(pop_data_cntm_bt)*100)
        print('--------------------------------')
        # mt
        print('pop_area_unit_mt: ', pop_area_mt)
        print('pop cntm mt: ', np.sum(pop_data_cntm_mt)) # people
        print('pop_data_dens_mt*pop_area_mt: ', np.sum(pop_data_dens_mt*pop_area_mt))
        print('adjust_ratio_mt: ', adjust_ratio_mt)
        print('diff_mt ratio(%): ', (np.sum(pop_data_dens_mt*pop_area_mt)-np.sum(pop_data_cntm_mt))/np.sum(pop_data_cntm_mt)*100)
        print('--------------------------------')
        # ft
        print('pop_area_unit_ft: ', pop_area_ft)
        print('pop cntm ft: ', np.sum(pop_data_cntm_ft)) # people
        print('pop_data_dens_ft*pop_area_ft: ', np.sum(pop_data_dens_ft*pop_area_ft))
        print('adjust_ratio_ft: ', adjust_ratio_ft)
        print('diff_ft ratio(%): ', (np.sum(pop_data_dens_ft*pop_area_ft)-np.sum(pop_data_cntm_ft))/np.sum(pop_data_cntm_ft)*100)
        print('--------------------------------')
        # bt 0-14
        print('pop_area_unit_bt_0_14: ', pop_area_bt_0_14)
        print('pop cntm bt_0_14: ', np.sum(pop_data_cntm_bt_0_14)) # people
        print('pop_data_dens_bt_0_14*pop_area_bt_0_14: ', np.sum(pop_data_dens_bt_0_14*pop_area_bt_0_14))
        print('adjust_ratio_bt_0_14: ', adjust_ratio_bt_0_14)
        print('diff_bt_0_14 ratio(%): ', (np.sum(pop_data_dens_bt_0_14*pop_area_bt_0_14)-np.sum(pop_data_cntm_bt_0_14))/np.sum(pop_data_cntm_bt_0_14)*100)
        print('--------------------------------')
        # bt 15-64
        print('pop_area_unit_bt_15_64: ', pop_area_bt_15_64)
        print('pop cntm bt_15_64: ', np.sum(pop_data_cntm_bt_15_64)) # people
        print('pop_data_dens_bt_15_64*pop_area_bt_15_64: ', np.sum(pop_data_dens_bt_15_64*pop_area_bt_15_64))
        print('adjust_ratio_bt_15_64: ', adjust_ratio_bt_15_64)
        print('diff_bt_15_64 ratio(%): ', (np.sum(pop_data_dens_bt_15_64*pop_area_bt_15_64)-np.sum(pop_data_cntm_bt_15_64))/np.sum(pop_data_cntm_bt_15_64)*100)
        print('--------------------------------')
        # bt 65+
        print('pop_area_unit_bt_65: ', pop_area_bt_65)
        print('pop cntm bt_65: ', np.sum(pop_data_cntm_bt_65)) # people
        print('pop_data_dens_bt_65*pop_area_bt_65: ', np.sum(pop_data_dens_bt_65*pop_area_bt_65))
        print('adjust_ratio_bt_65: ', adjust_ratio_bt_65)
        print('diff_bt_65 ratio(%): ', (np.sum(pop_data_dens_bt_65*pop_area_bt_65)-np.sum(pop_data_cntm_bt_65))/np.sum(pop_data_cntm_bt_65)*100)
        print('--------------------------------')
    else:
        print('Details not printed. Set print_details=True to print details')

        return adjust_ratio_bt, adjust_ratio_mt, adjust_ratio_ft, adjust_ratio_bt_0_14, adjust_ratio_bt_15_64, adjust_ratio_bt_65


def get_pop_with_pnc(pollution_type:str='PNC'):

    pnclon, pnclat, pnc_data, pncmat_data = None, None, None, None
    (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    ) = load_mask_pop_data()

    (
        adjust_ratio_bt, adjust_ratio_mt, adjust_ratio_ft,
        adjust_ratio_bt_0_14, adjust_ratio_bt_15_64, adjust_ratio_bt_65
    ) = print_pop_data_details(print_details=False, pollution_type=pollution_type)

    valid_pollution_types = ['PNC', 'PM2.5', 'PM10']
    if pollution_type not in valid_pollution_types:
        raise ValueError(f"Invalid pollution_type. Expected one of: {valid_pollution_types}")
    elif pollution_type == 'PNC':
        pnclon, pnclat, pnc_data, pncmat_data = load_pnc_annual_data()
    elif pollution_type == 'PM2.5':
        pnclon, pnclat, pnc_data, pncmat_data = load_pm_data(pm_type='PM2.5')
    elif pollution_type == 'PM10':
        pnclon, pnclat, pnc_data, pncmat_data = load_pm_data(pm_type='PM10')

    newpop_values_bt, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, adjust_ratio_bt)
    print('pop total shape and sum : ', newpop_values_bt.shape, np.sum(newpop_values_bt))
    newpop_values_mt, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, adjust_ratio_mt)
    print('male pop shape and sum : ', newpop_values_mt.shape, np.sum(newpop_values_mt))
    newpop_values_ft, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, adjust_ratio_ft)
    print('female pop shape and sum : ', newpop_values_ft.shape, np.sum(newpop_values_ft))
    newpop_values_bt_0_14, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, adjust_ratio_bt_0_14)
    print('age 0-14 pop shape and sum : ', newpop_values_bt_0_14.shape, np.sum(newpop_values_bt_0_14))
    newpop_values_bt_15_64, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, adjust_ratio_bt_15_64)
    print('age 15-64 pop shape and sum : ', newpop_values_bt_15_64.shape, np.sum(newpop_values_bt_15_64))
    newpop_values_bt_65, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, adjust_ratio_bt_65)
    print('age 65+ pop shape and sum : ', newpop_values_bt_65.shape, np.sum(newpop_values_bt_65))

    data_save = np.stack((pnc_data, newpop_values_bt, newpop_values_mt,
                        newpop_values_ft, newpop_values_bt_0_14, newpop_values_bt_15_64,
                        newpop_values_bt_65), axis=0)
    
    return data_save

def get_pop_with_pnc_hourly():
    # THis fuction need to run on server
    (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    ) = load_mask_pop_data()

    (
        adjust_ratio_bt, adjust_ratio_mt, adjust_ratio_ft,
        adjust_ratio_bt_0_14, adjust_ratio_bt_15_64, adjust_ratio_bt_65
    ) = print_pop_data_details(print_details=False, pollution_type='PNC')

    hourId = 8760
    # data_save_all = np.array((hourId, 7, 211, 531))
    data_save_all = []

    for i in trange(hourId):
        data_save = np.array((7, 211, 531))
        pnc_hourly_filename = '../../pncEstimator-main/src/postProcessing/matdata/pncall3/2020PNC_avgConc_Stacking'+ str(i) +'.mat'
        pnclon, pnclat, pnc_data, pncmat_data = load_pnc_hourly_data(pnc_hourly_filename)
        
        newpop_values_bt, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, adjust_ratio_bt)
        newpop_values_mt, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, adjust_ratio_mt)
        newpop_values_ft, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, adjust_ratio_ft)
        newpop_values_bt_0_14, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, adjust_ratio_bt_0_14)
        newpop_values_bt_15_64, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, adjust_ratio_bt_15_64)
        newpop_values_bt_65, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, adjust_ratio_bt_65)

        data_save = np.stack((pnc_data, newpop_values_bt, newpop_values_mt,
                            newpop_values_ft, newpop_values_bt_0_14, newpop_values_bt_15_64,
                            newpop_values_bt_65), axis=0)

        data_save_all.append(data_save)
        np.save('../../pop/pop_PNC/npy_file/hourId/pnc_pop_data'+ str(i) +'.npy', data_save)
    data_save_all = np.array(data_save_all)
    np.save('../../pop/pop_PNC/npy_file/pnc_pop_data.npy', data_save_all)

def save_tiff(save_tiff_name:str, save_tiff_data, lon_grid, lat_grid):
    transform = Affine.translation(lon_grid[0][0], lat_grid[0][0]) * Affine.scale(lon_grid[0,1]-lon_grid[0,0],
                                                                              lat_grid[1,0]-lat_grid[0,0])
    dataset = rasterio.open(
        save_tiff_name, 'w',
        driver='GTiff',
        height=lon_grid.shape[0],
        width=lat_grid.shape[1],
        count=1,
        dtype=save_tiff_data.dtype,
        crs='+proj=latlong',
        transform=transform,
    )
    dataset.write(save_tiff_data, 1)
    dataset.close()

def get_annual_tiff_file(pnc_pop_npy_filename:str):
    pnc2020_pop_data = np.load(pnc_pop_npy_filename)
    pop_bt = pnc2020_pop_data[1, :, :]
    pop_mt = pnc2020_pop_data[2, :, :]
    pop_ft = pnc2020_pop_data[3, :, :]
    pop_bt_0_14 = pnc2020_pop_data[4, :, :]
    pop_bt_15_64 = pnc2020_pop_data[5, :, :]
    pop_bt_65 = pnc2020_pop_data[6, :, :]

    pncmat_data = loadmat('../../pop/pop_PNC/mat_file/avgConc_with_all_data.mat')
    matlon = pncmat_data['lonNew'][0][:]
    matlat = pncmat_data['latNew'][0][:]
    pnc = pncmat_data['avgConc']
    pnc2020_data  = np.nan_to_num(pnc, nan=0.0)
    pop = pncmat_data['interpolated_pop']
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)

    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc2020.tif', pnc, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pop_bt.tif', pop_bt, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pop_mt.tif', pop_mt, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pop_ft.tif', pop_ft, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pop_bt_0_14.tif', pop_bt_0_14, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pop_bt_15_64.tif', pop_bt_15_64, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pop_bt_65.tif', pop_bt_65, lon_grid, lat_grid)

    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt.tif', pop_bt*pnc2020_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc_pop_mt.tif', pop_mt*pnc2020_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc_pop_ft.tif', pop_ft*pnc2020_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt_0_14.tif', pop_bt_0_14*pnc2020_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt_15_64.tif', pop_bt_15_64*pnc2020_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt_65.tif', pop_bt_65*pnc2020_data, lon_grid, lat_grid)

def get_hours_tiff_file():
    hour_low_pnc_data = np.load('.../../pop/pop_PNC/npy_file/1hour_low_pnc_data.npy')
    hour_high_pnc_data = np.load('../../pop/pop_PNC/npy_file/1hour_high_pnc_data.npy')

    day_low_pnc_data = np.load('../../pop/pop_PNC/npy_file/24hours_low_pnc_data.npy')
    day_middle_pnc_data = np.load('../../pop/pop_PNC/npy_file/24hours_middle_pnc_data.npy')
    day_high_pnc_data = np.load('../../pop/pop_PNC/npy_file/24hours_high_pnc_data.npy')

    pncmat_data = loadmat('../../pop/pop_PNC/mat_file/avgConc_with_all_data.mat')
    matlon = pncmat_data['lonNew'][0][:]
    matlat = pncmat_data['latNew'][0][:]
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)

    save_tiff('../../pop/pop_PNC/tif_file/hour/hour_low_pnc.tif', hour_low_pnc_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/hour/hour_high_pnc.tif', hour_high_pnc_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/hour/day_low_pnc.tif', day_low_pnc_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/hour/day_middle_pnc.tif', day_middle_pnc_data, lon_grid, lat_grid)
    save_tiff('../../pop/pop_PNC/tif_file/hour/day_high_pnc.tif', day_high_pnc_data, lon_grid, lat_grid)


def get_district_data(tiff_filename:str, value_name:str, data):
    tiff_dataset = rasterio.open(tiff_filename)
    gdf = gpd.read_file('../../pncEstimator-main/data/geoshp/gadm36_CHE_3.shp')
    
    weighted_data_by_district_mean = {}
    weighted_data_by_district_sum = {}
    for idx, row in gdf.iterrows():
        district_name = row['NAME_3']
        district_shape = row['geometry']

        mask = features.geometry_mask([district_shape], transform=tiff_dataset.transform, invert=True,
                                    out_shape=(tiff_dataset.height, tiff_dataset.width), all_touched=True)

        district_value = data[mask]
        weighted_data_by_district_sum[district_name] = np.sum(district_value)
        weighted_data_by_district_mean[district_name] = np.mean(district_value)
    
    weighted_district_mean = []
    weight_value_mean = []
    weight_value_sum = []
    for district, value in weighted_data_by_district_mean.items():
        weighted_district_mean.append(district)
        weight_value_mean.append(value)
    for district, value in weighted_data_by_district_sum.items():
        weight_value_sum.append(value)
    
    tiff_dataset.close()

    weighted_data = pd.DataFrame(weighted_district_mean, columns=['District'])
    weighted_data[value_name + '_mean'] = weight_value_mean
    weighted_data[value_name + '_sum'] = weight_value_sum

    return weighted_data

def get_distirct_hour_data_with_tiff():
    hour_low_pnc_data = np.load('../../pop/pop_PNC/npy_file/1hour_low_pnc_data.npy')
    hour_high_pnc_data = np.load('../../pop/pop_PNC/npy_file/1hour_high_pnc_data.npy')

    day_low_pnc_data = np.load('../../pop/pop_PNC/npy_file/24hours_low_pnc_data.npy')
    day_middle_pnc_data = np.load('../../pop/pop_PNC/npy_file/24hours_middle_pnc_data.npy')
    day_high_pnc_data = np.load('../../pop/pop_PNC/npy_file/24hours_high_pnc_data.npy')

    weighted_hour_low_pnc = get_district_data('../../pop/pop_PNC/tif_file/hour/hour_low_pnc.tif', 'hour_low_pnc', hour_low_pnc_data)
    weighted_hour_high_pnc = get_district_data('../../pop/pop_PNC/tif_file/hour/hour_high_pnc.tif', 'hour_high_pnc', hour_high_pnc_data)
    weighted_day_low_pnc = get_district_data('../../pop/pop_PNC/tif_file/hour/day_low_pnc.tif', 'day_low_pnc', day_low_pnc_data)
    weighted_day_middle_pnc = get_district_data('../../pop/pop_PNC/tif_file/hour/day_middle_pnc.tif', 'day_middle_pnc', day_middle_pnc_data)
    weighted_day_high_pnc = get_district_data('../../pop/pop_PNC/tif_file/hour/day_high_pnc.tif', 'day_high_pnc', day_high_pnc_data)

    dataframes = [weighted_hour_low_pnc, weighted_hour_high_pnc,
                  weighted_day_low_pnc, weighted_day_middle_pnc, weighted_day_high_pnc]
    weighted_data_all = dataframes[0]

    for df in dataframes[1:]:
        weighted_data_all = pd.merge(weighted_data_all, df, on='District')

    return weighted_data_all

def get_distirct_data_with_tiff(pnc_pop_npy_filename):
    pnc2020_pop_data = np.load(pnc_pop_npy_filename)
    pncmat_data = loadmat('../../pop/pop_PNC/mat_file/avgConc_with_all_data.mat')
    pnc = pncmat_data['avgConc']
    pnc2020_data  = np.nan_to_num(pnc, nan=0.0)
    pop = pncmat_data['interpolated_pop']
    pop_bt = pnc2020_pop_data[1, :, :]
    pop_mt = pnc2020_pop_data[2, :, :]
    pop_ft = pnc2020_pop_data[3, :, :]
    pop_bt_0_14 = pnc2020_pop_data[4, :, :]
    pop_bt_15_64 = pnc2020_pop_data[5, :, :]
    pop_bt_65 = pnc2020_pop_data[6, :, :]

    weighted_pnc = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc2020.tif', 'pnc2020', pnc2020_data)
    weighted_pop_bt = get_district_data('../../pop/pop_PNC/tif_file/annual/pop_bt.tif', 'pop_bt', pop_bt)
    weighted_pop_mt = get_district_data('../../pop/pop_PNC/tif_file/annual/pop_mt.tif', 'pop_mt', pop_mt)
    weighted_pop_ft = get_district_data('../../pop/pop_PNC/tif_file/annual/pop_ft.tif', 'pop_ft', pop_ft)
    weighted_pop_bt_0_14 = get_district_data('../../pop/pop_PNC/tif_file/annual/pop_bt_0_14.tif', 'pop_bt_0_14', pop_bt_0_14)
    weighted_pop_bt_15_64 = get_district_data('../../pop/pop_PNC/tif_file/annual/pop_bt_15_64.tif', 'pop_bt_15_64', pop_bt_15_64)
    weighted_pop_bt_65 = get_district_data('../../pop/pop_PNC/tif_file/annual/pop_bt_65.tif', 'pop_bt_65', pop_bt_65)

    weighted_pnc_pop_bt = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt.tif', 'pnc_pop_bt', pop_bt*pnc2020_data)
    weighted_pnc_pop_mt = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc_pop_mt.tif', 'pnc_pop_mt', pop_mt*pnc2020_data)
    weighted_pnc_pop_ft = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc_pop_ft.tif', 'pnc_pop_ft', pop_ft*pnc2020_data)
    weighted_pnc_pop_bt_0_14 = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt_0_14.tif', 'pnc_pop_bt_0_14', pop_bt_0_14*pnc2020_data)
    weighted_pnc_pop_bt_15_64 = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt_15_64.tif', 'pnc_pop_bt_15_64', pop_bt_15_64*pnc2020_data)
    weighted_pnc_pop_bt_65 = get_district_data('../../pop/pop_PNC/tif_file/annual/pnc_pop_bt_65.tif', 'pnc_pop_bt_65', pop_bt_65*pnc2020_data)
    dataframes = [weighted_pnc, weighted_pnc_pop_bt, weighted_pnc_pop_mt,
                  weighted_pnc_pop_ft, weighted_pnc_pop_bt_0_14, weighted_pnc_pop_bt_15_64,
                  weighted_pnc_pop_bt_65, weighted_pop_bt, weighted_pop_mt,
                  weighted_pop_ft, weighted_pop_bt_0_14, weighted_pop_bt_15_64,
                  weighted_pop_bt_65]
    weighted_data_all = dataframes[0]

    for df in dataframes[1:]:
        weighted_data_all = pd.merge(weighted_data_all, df, on='District')

    return weighted_data_all

def get_select_area_data_with_tiff():
    select_area_data = np.load('../../pop/pop_PNC/npy_file/select_hour_area.npy')
    pncmat_data = loadmat('../../pop/pop_PNC/mat_file/avgConc_with_all_data.mat')
    matlon = pncmat_data['lonNew'][0][:]
    matlat = pncmat_data['latNew'][0][:]
    matlat_temp = matlat[45:52]
    matlon_temp = matlon[285:292]
    lon_grid, lat_grid = np.meshgrid(matlon_temp, matlat_temp)
    weighted_temp_all = []
    # for i in trange(8760):
    for i in trange(0, 3):
        temp = select_area_data[i, :, :]
        save_tiff(f'../../pop/pop_PNC/tif_file/temp/temp{i}.tif', temp, lon_grid, lat_grid)
        weighted_temp = get_district_data(f'../../pop/pop_PNC/tif_file/temp/temp{i}.tif', 'select_area', temp)
        weighted_temp['iteration'] = i
        weighted_temp_all.append(weighted_temp)
    weighted_temp_all = pd.concat(weighted_temp_all, axis=1)
    weighted_temp_all.to_csv('../../pop/pop_PNC/tif_file/temp/select_area_data.csv', index=False)
# parrallel
# select_area_data = np.load('./npy_file/select_hour_area.npy')
# pncmat_data = loadmat('./mat_file/avgConc_with_all_data.mat')
# matlon = pncmat_data['lonNew'][0][:]
# matlat = pncmat_data['latNew'][0][:]
# matlat_temp = matlat[45:52]
# matlon_temp = matlon[285:292]
# lon_grid, lat_grid = np.meshgrid(matlon_temp, matlat_temp)

# def process_image(i):
#     temp = select_area_data[i, :, :]
#     tiff_filename = f'./tif_file/temp/temp{i}.tif'
#     save_tiff(tiff_filename, temp, lon_grid, lat_grid)
#     weighted_temp = get_district_data(tiff_filename, 'select_area', temp)
#     weighted_temp['iteration'] = i
#     return weighted_temp

# def main():
#     results = []
#     with ProcessPoolExecutor() as executor:
#         future_to_iter = {executor.submit(process_image, i): i for i in range(3)}
#         for future in as_completed(future_to_iter):
#             result = future.result()
#             results.append(result)

#     weighted_temp_all = pd.concat(results, axis=1)
#     weighted_temp_all.to_csv('./tif_file/temp/select_area_data.csv', index=False)

# if __name__ == '__main__':
#     main()

        
def main():
    # pnc_pop_annual_data = get_pop_with_pnc(pollution_type='PNC')
    # print(pnc_pop_annual_data.shape)
    # np.save('./npy_file/pnc2020_pop.npy', pnc_pop_annual_data)

    # pm25_pop_annual_data = get_pop_with_pnc(pollution_type='PM2.5')
    # print(pm25_pop_annual_data.shape)
    # np.save('../../../con_code/out/npy_file/pm25_2020_pop.npy', pm25_pop_annual_data)

    # pm10_pop_annual_data = get_pop_with_pnc(pollution_type='PM10')
    # print(pm10_pop_annual_data.shape)
    # np.save('../../../con_code/out/npy_file/pm10_2020_pop.npy', pm10_pop_annual_data)

    # get_annual_tiff_file('./npy_file/pnc2020_pop.npy')
    get_hours_tiff_file()
    get_select_area_data_with_tiff()

if __name__ == "__main__":
    main()