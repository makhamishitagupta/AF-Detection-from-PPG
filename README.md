# Arrhythmia (Atrial Fibrillation) Detection from PPG Signals using LightGBM

An end-to-end machine learning project to detect **Atrial Fibrillation (AF)** — a prevalent cardiac arrhythmia — from **Photoplethysmography (PPG)** optical pulse waveforms using **LightGBM**.

---

## 1. Project Overview

Atrial Fibrillation (AF) is one of the most common cardiac arrhythmias, significantly increasing the risk of stroke, heart failure, and cardiovascular complications. While traditional 12-lead electrocardiography (ECG) is the clinical gold standard, it is usually intermittent and hospital-based, making it difficult to detect paroxysmal (episodic) and asymptomatic arrhythmias.

Consumer wearables (such as smartwatches and fitness bands) use optical **Photoplethysmography (PPG)** sensors to monitor blood volume pulse at the wrist. This project develops an end-to-end pipeline that processes continuous raw PPG signals, extracts 35 time/frequency/entropy features, and classifies arrhythmia episodes using **LightGBM**.

### Key Highlights
- **Signal Preprocessing:** 3rd-order zero-phase Butterworth bandpass filter ($0.5 - 5.0\text{ Hz}$, $30 - 300\text{ bpm}$) with adaptive systolic peak detection.
- **35 Tabular Features:** Pulse rate variability (RMSSD, pNN50, pNN20, CV IBI, IQR), non-linear Poincaré dynamics ($SD_1, SD_2$), Sample Entropy, Shannon Entropy, systolic amplitude dynamics, and Welch spectral power densities.
- **Grouped Train/Test Split:** Data is partitioned by `patient_id` using **5-Fold Stratified Group Cross-Validation** and an independent **holdout test set of 9 unseen patients** (711 windows), preventing patient data leakage.
- **Dual-Level Evaluation:** Real-time 30-second window-level detection and 20-minute patient-level aggregated classification.
- **Ready to Run:** Fully self-contained notebook and standalone script supporting both Google Colab and local Python environments.

---

## 2. Dataset

This project uses recordings from the **MIMIC PERform AF Dataset** ([Charlton et al., *Physiological Measurement*, 2022](https://ppg-beats.readthedocs.io/en/latest/)):
- **Total Patients:** 34 ICU subjects (19 with Atrial Fibrillation, 15 with Normal Sinus Rhythm).
- **Sampling Frequency ($f_s$):** $125\text{ Hz}$ ($\Delta t = 0.008\text{ s}$).
- **Recording Duration:** $1,200\text{ seconds}$ (20 continuous minutes per subject).
- **Segmented Windows:** 2,686 windows ($30\text{ s}$ duration with a $15\text{ s}$ stride / 50% overlap).

---

## 3. Pipeline Architecture

```
Raw PPG Signal (125 Hz)
       │
       ▼
1. Signal Preprocessing & Filtering
   (3rd-Order Butterworth Bandpass, 0.5 - 5.0 Hz)
       │
       ▼
2. Peak Detection & Inter-Beat Intervals (IBI)
   (Adaptive prominence threshold based on window standard deviation)
       │
       ▼
3. Feature Engineering (35 Features per 30s Window)
   ├── Pulse Rate Variability (PRV): Mean, SD, RMSSD, pNN50, pNN20, IQR
   ├── Poincaré Geometry: SD1, SD2, SD1/SD2 ratio, ellipse area
   ├── Entropy: Sample Entropy, Shannon Entropy
   ├── Amplitude Dynamics: Peak amplitude mean, std, CV, count
   └── Spectral Power (PSD): VLF, LF, HF, Cardiac power, Welch entropy
       │
       ▼
4. Patient-Level Group Split
   ├── Training Set: 25 Patients (1,975 windows, 5-Fold Stratified Group CV)
   └── Holdout Test Set: 9 Unseen Patients (711 windows)
       │
       ▼
5. LightGBM Classification & Evaluation
   ├── Window-Level Predictions (30-second continuous monitoring)
   └── Patient-Level Aggregation (Majority vote across 20-minute session)
```

---

## 4. Feature Engineering Breakdown

| Category | Extracted Features | Description |
| :--- | :--- | :--- |
| **Pulse Rate Variability (PRV)** | `mean_ibi`, `std_ibi`, `rmssd`, `pnn50`, `pnn20`, `cv_ibi`, `median_ibi`, `iqr_ibi`, `min_ibi`, `max_ibi`, `range_ibi`, `hr_bpm` | Captures heartbeat interval dispersion and beat-to-beat differences. |
| **Poincaré Plot Geometry** | `poincare_sd1`, `poincare_sd2`, `poincare_sd_ratio`, `poincare_area` | Non-linear phase space plotting ($IBI_n \text{ vs. } IBI_{n+1}$). $SD_1$ reflects short-term interval variability, and $SD_2$ reflects long-term rhythm trends. |
| **Complexity & Entropy** | `sample_entropy_ibi`, `shannon_entropy_ibi` | Measures randomness and irregularity in heartbeat timing sequences. |
| **Amplitude Dynamics** | `mean_peak_amp`, `std_peak_amp`, `cv_peak_amp`, `peak_count` | Reflects variations in pulse peak heights caused by irregular ventricular filling times. |
| **Waveform & Spectral Features** | `ppg_mean`, `ppg_std`, `ppg_skew`, `ppg_kurtosis`, `zero_crossing_rate`, `vlf_power`, `lf_power`, `hf_power`, `cardiac_power`, `lf_hf_ratio`, `total_spectral_power`, `spectral_entropy`, `dominant_freq` | Statistical moments of the waveform and frequency band power distributions from Welch periodograms. |

---

## 5. Model Performance & Evaluation

### A. 5-Fold Stratified Group Cross-Validation (25 Patients, 1,975 Windows)
*Grouped strictly by `patient_id` so windows from the same subject never appear in both train and validation sets.*

| Model | Window Accuracy | Window Recall (AF Sens.) | Window F1-Score | Window ROC-AUC | Patient Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | $81.56\% \pm 6.94\%$ | $86.53\% \pm 10.64\%$ | $82.78\% \pm 7.42\%$ | $0.8037 \pm 0.0827$ | $88.00\% \pm 16.00\%$ |
| **Random Forest (Baseline)** | $86.27\% \pm 10.45\%$ | $91.33\% \pm 12.06\%$ | $87.65\% \pm 9.68\%$ | $0.9760 \pm 0.0249$ | $88.00\% \pm 16.00\%$ |
| **LightGBM (Final Model)** | **$87.34\% \pm 11.93\%$** | **$91.22\% \pm 12.29\%$** | **$88.40\% \pm 11.06\%$** | **$0.9818 \pm 0.0214$** | **$88.00\% \pm 16.00\%$** |

### B. Holdout Test Set Evaluation (9 Unseen Patients, 711 Windows)
- **Window Accuracy:** **$88.33\%$**
- **AF Detection Sensitivity (Recall):** **$98.99\%$** ($391$ of $395$ arrhythmia windows correctly detected)
- **Window Precision:** **$83.19\%$**
- **Window F1-Score:** **$90.40\%$**
- **Window ROC-AUC:** **$0.9863$**
- **Window PR-AUC:** **$0.9895$**
- **Confusion Matrix:**
  - True Negatives (TN): 237
  - False Positives (FP): 79
  - False Negatives (FN): 4
  - True Positives (TP): 391
- **Patient-Level Accuracy:** **$88.89\%$** ($8$ of $9$ patients correctly classified).

### C. Top Discriminative Features
1. `iqr_ibi` (Interquartile Range of Inter-Beat Intervals) — measures overall pulse interval spread.
2. `pnn50` (Percentage of consecutive interval differences $> 50\text{ ms}$) — key marker of irregular ventricular conduction.
3. `spectral_entropy` (Welch PSD disorder) — measures power spectrum spread across frequencies.
4. `sample_entropy_ibi` — non-linear irregularity of heartbeat sequences.
5. `poincare_sd_ratio` ($SD_1/SD_2$) — balance between short-term and long-term interval variability.
6. `cv_peak_amp` (Coefficient of variation of pulse heights) — captures stroke volume fluctuations.

---

## 6. Visualizations

Generated plots are saved in the `figures/` directory:
- `figures/fig1_ppg_waveforms_peaks.png`: Normal vs. AF filtered PPG waveforms with detected systolic peaks.
- `figures/fig2_poincare_tachogram.png`: Poincaré scatter plots ($IBI_n \text{ vs. } IBI_{n+1}$) and IBI tachograms.
- `figures/fig3_roc_confusion_matrix.png`: ROC curve (5-Fold CV envelope + Holdout) and holdout confusion matrix heatmap.
- `figures/fig4_feature_importance.png`: Top 15 ranked features by LightGBM split importance.

---

## 7. Project Structure

```
Arrhythmia-detection-using-PPG/
├── Arrhythmia_detection_using_PPG.ipynb  # Primary interactive Jupyter Notebook
├── pipeline_lightgbm.py                 # Standalone command-line Python script
├── requirements.txt                     # Python package dependencies
├── README.md                            # Project documentation
├── .gitignore                           # Git ignore configuration
├── figures/                             # Generated diagnostic plots
│   ├── fig1_ppg_waveforms_peaks.png
│   ├── fig2_poincare_tachogram.png
│   ├── fig3_roc_confusion_matrix.png
│   └── fig4_feature_importance.png
├── mimic_perform_af_csv/                # Raw AF patient recordings
├── mimic_perform_non_af_csv/            # Raw Normal Sinus Rhythm patient recordings
└── output5.csv                          # Preprocessed patient recording CSV
```

---

## 8. Quickstart & How to Run

### Option 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Arrhythmia_detection_using_PPG.ipynb`.
3. Upload `output5.csv` (or the folders `mimic_perform_af_csv` and `mimic_perform_non_af_csv`) to `/content/`.
4. Click **Runtime -> Run all** (`Ctrl + F9`).

### Option 2: Local Python Environment
```bash
# 1. Clone repository
git clone <your-repository-url>
cd Arrhythmia-detection-using-PPG

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the Jupyter Notebook
jupyter notebook Arrhythmia_detection_using_PPG.ipynb

# Or run the standalone pipeline script
python pipeline_lightgbm.py
```