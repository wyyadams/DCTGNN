# DCTGNN
## How to use:
### Step1：
Download MODMA, HUSM
MODMA: [https://modma.lzu.edu.cn/data/index/](https://modma.lzu.edu.cn/data/index/)
HUSM: [https://figshare.com/articles/dataset/EEG_Data_New/4244171](https://figshare.com/articles/dataset/EEG_Data_New/4244171)

Tips: Only 19 electrodes are used in MODMA (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, C4, T4, T5, P3, Pz, P4, T6, O1, Oz, O2)

### Step2:
EEG preprocessing using the [MNE](https://mne.tools/stable/index.html) library 

(1) Filtering
1Hz high-pass filter, 100Hz low-pass filter, 50Hz notch filter

(2) Average Re-referencing

(3) ICA
ica = ICA(n_components=30, max_iter="auto", method="infomax", random_state=97, fit_params=dict(extended=True))

(4) Segment and Reshape
data:[n_subject, n_sample, n_electrode, n_subslices, n_timestamps\]
label:[n_subject\]  0-Depression, 1-Healthy Controls 
save as "MODMA_dataset.npz"/"HUSM_dataset.npz", revise the "dataset_path" in main.py

Note: MODMA_Dataset\MODMA_demo_dataset.npz is only used to illustrate the file format and array shape of the stored dataset.

### Step3:
run main.py
