<!--
 * @Author: JYYD jyyd23@mails.tsinghua.edu.cn
 * @Date: 2024-05-08 19:46:19
 * @LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
 * @LastEditTime: 2024-11-05 01:21:43
 * @FilePath: README.md
 * @Description: 
 * 
-->

# my_project:PNC_Estimate

:triangular_flag_on_post: For example on a small dataset that can run on a personal laptop in `example_test`. The `./example_test/dataset/` can download from [dataset](https://huggingface.co/jyyd23/PNC_Estimate/tree/main).

## 1.Description

**Machine Learning-Enhanced High-Resolution Exposure Assessment of Ultrafine Particles**

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
4. cams_train_pred.ipynb: Get site-specific prediction results.
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
1. pop_data_processing.py: The functions pop_plot_final.ipynb used.
2. pop_plot_final.ipynb: Get the population related figures.


## 4.Software and Hardware Environment and Version

### 4.1 Hardware

#### 4.1.1 Computer
+ **CPU：** Intel(R) Core(TM) i7-10750H
+ **GPU：** NVIDIA Geforce GTX 1650 Ti
+ **RAM：** 16GB

#### 4.1.2 Workstation CPU
+ **CPU：** Intel(R) Xeon(R) Platinum 8383C
+ **GPU：** NVIDIA Geforce GTX 3090
+ **RAM：** 256GB

#### 4.1.3 Workstation GPU
+ **CPU：** Intel(R) Xeon(R) Platinum 8488C 2.40 GHz
+ **GPU：** 4 x NVIDIA RTX A6000
+ **RAM：** 512GB

### 4.1 Software

#### 4.2.1 Computer
* **Operating System：** Windows 11
* **Python Version：** Anaconda Python 3.9.7
* **CUDA Version：** 11.7

#### 4.2.2 Workstation CPU
* **Operating System：** Windows 10
* **Python Version：** Anaconda Python 3.11.4
* **CUDA Version：** 12.2

#### 4.2.3 Workstation GPU
* **Operating System：** Windows 10
* **Python Version：** Python 3.12.6
* **CUDA Version：** 12.6

## 5.Usage

### 5.1 Environment configuration commands
- For convenience, execute the following command.

```
    pip install -r requirements.txt
```

### 5.2 Develop your own model.
- Start the dataset. A small dataset `./example_test/dataset/` can download from [dataset](https://huggingface.co/jyyd23/PNC_Estimate/blob/main/dataset) as an example.

- Use the `./PNC_pred/model_train.ipynb` to train your own model.


## 6.Citation

If you find this repo useful, please cite our paper.

- [1]Jianyao, Y., Yuan, H., Su, G. et al. Machine learning-enhanced high-resolution exposure assessment of ultrafine particles. Nat Commun 16, 1209 (2025). https://doi.org/10.1038/s41467-025-56581-8.
- [2]Jianyao, Y. & Zhang, X. Machine learning-enhanced high-resolution exposure assessment of ultra fine particles. Zenodo https://doi.org/10.5281/zenodo.14554168 (2024).
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14554168.svg)](https://doi.org/10.5281/zenodo.14554168 (2024))
- [3]Jianyao, Y. & Zhang, X. Machine learning-enhanced high-resolution exposure assessment of ultra fine particles (Mini Dataset + Stem-PNC weights). Zenodo https://doi.org/10.5281/zenodo.14634738 (2025).
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14634738.svg)](https://doi.org/10.5281/zenodo.14634738 (2025))


## 7.Contact
If you have any questions or suggestions, feel free to contact:
- Yudie Jianyao (Ph.D. student, jyyd23@mails.tsinghua.edu.cn)
