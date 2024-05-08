<!--
 * @Author: JYYD jyyd23@mails.tsinghua.edu.cn
 * @Date: 2024-05-08 19:46:19
 * @LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
 * @LastEditTime: 2024-05-08 20:25:37
 * @FilePath: \finalcode\README.md
 * @Description: 
 * 
-->

# my_project:PNC_Estimate

## 1.Description

Artificial Intelligence-Enhanced High-Resolution Exposure Assessment of Ultrafine Particles

## 2.File Structure
```
finalcode
├─ CAMS_downscale
│  ├─ CAMSpollutionDistribution.m
│  ├─ CAMS_downscale_plot.ipynb
│  ├─ cams_train.py
│  ├─ cams_train_pred.ipynb
│  ├─ downScaleConc.m
│  ├─ preprocessing.py
├─ PNC_estimate
│  ├─ diffplot.m
│  ├─ PNC_estimate_plot.ipynb
│  └─ swissPNCDistribution.m
├─ PNC_pred
│  ├─ model_train.ipynb
│  ├─ pnc_pred.py
│  ├─ PNC_pred_plot.ipynb
│  ├─ test_pnc_model
│  │  ├─ model_test.py
│  └─ trained model
├─ pop_PNC
│  ├─ pop_data_processing.py
│  ├─ pop_plot_final.ipynb
└─ SI_figure

```

## 3.File Function

### 2.1 CAMS_downscale
1. downScaleConc.m : If predictionFlag = 0 can get the pollution_trainData.csv, and if predictionFlag = 1 get 8760 hourId_pollutant_predData.bin files.
2. preprocessing.py : Contains functions cams_train.py may used.
3. cams_train.py : Use pollution_trainData.csv and hourId_pollutant_predData.bin files to the annual pollution high-resolution 1km grid data used machine learning methods.
4. cams_train_pred.ipynb : Get site-specific prediction results.
5. CAMS_downscale_plot.ipynb : Figure plot code for pollution compares.
6. CAMSpollutionDistribution.m : Figure plot code for DownScales pollution.

### 2.2 PNC_estimate
1. swissPNCDistribution.m : The code to plot 2020_DownScale_PNC.png.
2. PNC_estimate_plot.ipynb : The code to plot other figures.
3. diffplot.m : The code plot diff between matflies.

### 2.3 PNC_pred
1.  test_pnc_model/model_test.py : DL code for pnc_pred.py.
2.  pnc_pred.py : Get the pnc_estimate mat files.
3.  model_train.ipynb : Train the PNC models.

### 2.4 pop_PNC
1. pop_data_processing.py : The functions pop_plot_final.ipynb used.
2. pop_plot_final.ipynb : Get the population related figures.


## 4.Software and Hardware Environment and Version

### 4.1 Hardware

#### 4.1.1 Computer
+ **CPU**: Intel(R) Core(TM) i7-10750H
+ **GPU**: NVIDIA Geforce GTX 1650 Ti
+ **RAM**: 16GB

#### 4.1.2 Workstation
+ **CPU**: Intel(R) Xeon(R) Platinum 8383C
+ **RAM**: 256GB

### 4.1 Software

#### 4.2.1 Computer
+ **Operating System**: Windows 11
+ **Python Version**: Anaconda Python 3.9.7
+ **CUDA Version**: 11.7

#### 4.2.2 Workstation
+ **Operating System**: Windows 10
+ **Python Version**: Anaconda Python 3.11.4

### 4.3 Environment configuration commands

 ```cmd
    pip install -r requirements.txt
```
