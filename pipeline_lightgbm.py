"""
Arrhythmia Detection Using Photoplethysmogram (PPG) Signals with LightGBM
Standalone Executable Pipeline
"""

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

def resolve_data_paths(base_dir="."):
    af_dir = os.path.join(base_dir, "mimic_perform_af_csv")
    non_af_dir = os.path.join(base_dir, "mimic_perform_non_af_csv")
    out5_path = os.path.join(base_dir, "output5.csv")
    
    has_dirs = os.path.isdir(af_dir) and os.path.isdir(non_af_dir)
    has_csv = os.path.isfile(out5_path)
    
    return {
        'has_dirs': has_dirs,
        'has_csv': has_csv,
        'af_dir': af_dir,
        'non_af_dir': non_af_dir,
        'out5_path': out5_path
    }

def bandpass_filter(sig, fs=125.0, low=0.5, high=5.0, order=3):
    if np.isnan(sig).any():
        nans = np.isnan(sig)
        x = np.arange(len(sig))
        sig = np.interp(x, x[~nans], sig[~nans])
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, sig)

def calc_sample_entropy(data, m=2, r=0.2):
    data = np.asarray(data)
    N = len(data)
    if N <= m + 2:
        return 0.0
    r_val = r * np.std(data)
    if r_val == 0:
        return 0.0

    def _phi(m_len):
        x = np.array([data[i:i + m_len] for i in range(N - m_len + 1)])
        diffs = np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
        count = (np.sum(diffs < r_val) - len(x)) / 2.0
        total = (len(x) * (len(x) - 1)) / 2.0
        return count / total if total > 0 else 0.0

    A = _phi(m + 1)
    B = _phi(m)
    if A > 0 and B > 0:
        return -np.log(A / B)
    return 0.0

def extract_window_features(window, fs=125.0):
    feats = {}
    feats['ppg_mean'] = float(np.mean(window))
    feats['ppg_std'] = float(np.std(window))
    feats['ppg_skew'] = float(stats.skew(window))
    feats['ppg_kurtosis'] = float(stats.kurtosis(window))
    zero_crossings = np.where(np.diff(np.signbit(window)))[0]
    feats['zero_crossing_rate'] = float(len(zero_crossings) / len(window))
    
    min_dist = int(0.27 * fs)
    prominence = 0.25 * (np.std(window) if np.std(window) > 1e-6 else 1.0)
    peaks, _ = signal.find_peaks(window, distance=min_dist, prominence=prominence)
    feats['peak_count'] = int(len(peaks))
    
    if len(peaks) >= 3:
        ibis = np.diff(peaks) / fs * 1000.0
        diff_ibis = np.diff(ibis)
        
        feats['mean_ibi'] = float(np.mean(ibis))
        feats['std_ibi'] = float(np.std(ibis, ddof=1) if len(ibis) > 1 else 0.0)
        feats['rmssd'] = float(np.sqrt(np.mean(diff_ibis**2)) if len(diff_ibis) > 0 else 0.0)
        feats['pnn50'] = float(np.sum(np.abs(diff_ibis) > 50.0) / len(diff_ibis) * 100.0 if len(diff_ibis) > 0 else 0.0)
        feats['pnn20'] = float(np.sum(np.abs(diff_ibis) > 20.0) / len(diff_ibis) * 100.0 if len(diff_ibis) > 0 else 0.0)
        feats['cv_ibi'] = float(feats['std_ibi'] / (feats['mean_ibi'] + 1e-6))
        feats['median_ibi'] = float(np.median(ibis))
        feats['iqr_ibi'] = float(np.percentile(ibis, 75) - np.percentile(ibis, 25))
        feats['min_ibi'] = float(np.min(ibis))
        feats['max_ibi'] = float(np.max(ibis))
        feats['range_ibi'] = float(feats['max_ibi'] - feats['min_ibi'])
        feats['hr_bpm'] = float(60000.0 / (feats['mean_ibi'] + 1e-6))
        
        if len(diff_ibis) >= 2:
            var_diff = np.var(diff_ibis, ddof=1)
            var_ibi = np.var(ibis, ddof=1)
            sd1 = np.sqrt(0.5 * var_diff)
            sd2_sq = 2.0 * var_ibi - 0.5 * var_diff
            sd2 = np.sqrt(max(0.0, sd2_sq))
            feats['poincare_sd1'] = float(sd1)
            feats['poincare_sd2'] = float(sd2)
            feats['poincare_sd_ratio'] = float(sd1 / (sd2 + 1e-6))
            feats['poincare_area'] = float(np.pi * sd1 * sd2)
        else:
            feats['poincare_sd1'] = 0.0
            feats['poincare_sd2'] = 0.0
            feats['poincare_sd_ratio'] = 0.0
            feats['poincare_area'] = 0.0
            
        hist, _ = np.histogram(ibis, bins=12, density=True)
        hist = hist[hist > 0]
        feats['shannon_entropy_ibi'] = float(-np.sum(hist * np.log2(hist + 1e-9)))
        feats['sample_entropy_ibi'] = float(calc_sample_entropy(ibis, m=2, r=0.2))
        
        peak_amps = window[peaks]
        feats['mean_peak_amp'] = float(np.mean(peak_amps))
        feats['std_peak_amp'] = float(np.std(peak_amps, ddof=1) if len(peak_amps) > 1 else 0.0)
        feats['cv_peak_amp'] = float(feats['std_peak_amp'] / (np.abs(feats['mean_peak_amp']) + 1e-6))
    else:
        for k in ['mean_ibi', 'std_ibi', 'rmssd', 'pnn50', 'pnn20', 'cv_ibi',
                  'median_ibi', 'iqr_ibi', 'min_ibi', 'max_ibi', 'range_ibi',
                  'hr_bpm', 'poincare_sd1', 'poincare_sd2', 'poincare_sd_ratio',
                  'poincare_area', 'shannon_entropy_ibi', 'sample_entropy_ibi',
                  'mean_peak_amp', 'std_peak_amp', 'cv_peak_amp']:
            feats[k] = 0.0
            
    freqs, psd = signal.welch(window, fs=fs, nperseg=min(len(window), int(4 * fs)))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    feats['spectral_entropy'] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    cardiac_mask = (freqs >= 0.5) & (freqs < 4.0)
    
    feats['vlf_power'] = float(np.sum(psd[vlf_mask]))
    feats['lf_power'] = float(np.sum(psd[lf_mask]))
    feats['hf_power'] = float(np.sum(psd[hf_mask]))
    feats['cardiac_power'] = float(np.sum(psd[cardiac_mask]))
    feats['lf_hf_ratio'] = float(feats['lf_power'] / (feats['hf_power'] + 1e-6))
    feats['total_spectral_power'] = float(np.sum(psd))
    
    cardiac_freqs = freqs[cardiac_mask]
    cardiac_psd = psd[cardiac_mask]
    feats['dominant_freq'] = float(cardiac_freqs[np.argmax(cardiac_psd)]) if len(cardiac_psd) > 0 else 0.0
    
    return feats

def load_patient_signals(base_dir="."):
    paths = resolve_data_paths(base_dir)
    patients = []
    fs = 125.0
    
    if paths['has_dirs']:
        af_files = sorted(glob.glob(os.path.join(paths['af_dir'], "*.csv")))
        non_af_files = sorted(glob.glob(os.path.join(paths['non_af_dir'], "*.csv")))
        for f in af_files:
            pid = os.path.basename(f).replace("_data.csv", "")
            df = pd.read_csv(f)
            patients.append({'patient_id': pid, 'label': 1, 'ppg': df['PPG'].values.astype(float), 'fs': fs})
        for f in non_af_files:
            pid = os.path.basename(f).replace("_data.csv", "")
            df = pd.read_csv(f)
            patients.append({'patient_id': pid, 'label': 0, 'ppg': df['PPG'].values.astype(float), 'fs': fs})
    elif paths['has_csv']:
        df = pd.read_csv(paths['out5_path'])
        valid_df = df.dropna(subset=['results']).copy()
        af_count, non_af_count = 1, 1
        for _, row in valid_df.iterrows():
            orig_val = int(row['results'])
            clinical_label = 1 if orig_val == 0 else 0
            if clinical_label == 1:
                pid = f"mimic_perform_af_{af_count:03d}"
                af_count += 1
            else:
                pid = f"mimic_perform_non_af_{non_af_count:03d}"
                non_af_count += 1
            sig = row.drop('results').values.astype(float)
            patients.append({'patient_id': pid, 'label': clinical_label, 'ppg': sig, 'fs': fs})
    else:
        raise FileNotFoundError("No valid data sources found!")
    return patients

def main():
    parser = argparse.ArgumentParser(description="Arrhythmia Detection Using PPG Signals with LightGBM")
    parser.add_argument("--data_dir", type=str, default=".", help="Path to project directory containing data")
    args = parser.parse_args()
    
    print("="*75)
    print("ARRHYTHMIA DETECTION USING PPG SIGNALS WITH LIGHTGBM")
    print("="*75)
    
    patients = load_patient_signals(args.data_dir)
    print(f"Loaded {len(patients)} patients: {sum(p['label']==1 for p in patients)} AF, {sum(p['label']==0 for p in patients)} Non-AF")
    
    print("\nExtracting 35 features from 30s windows (15s stride)...")
    t0 = time.time()
    records = []
    window_sec, step_sec, fs = 30, 15, 125.0
    w_len, s_len = int(window_sec * fs), int(step_sec * fs)
    
    for p in patients:
        filt = bandpass_filter(p['ppg'], fs=fs, low=0.5, high=5.0)
        filt = (filt - np.mean(filt)) / (np.std(filt) + 1e-6)
        n = len(filt)
        for s in range(0, n - w_len + 1, s_len):
            feats = extract_window_features(filt[s:s + w_len], fs=fs)
            feats['patient_id'] = p['patient_id']
            feats['label'] = p['label']
            feats['window_start_sec'] = s / fs
            records.append(feats)
            
    df_feat = pd.DataFrame(records)
    print(f"Extracted {len(df_feat)} windows across {len(records[0])} dimensions in {time.time()-t0:.2f} s")
    
    feature_cols = [c for c in df_feat.columns if c not in ['patient_id', 'label', 'window_start_sec']]
    
    # 5-Fold Stratified Group CV
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = df_feat[feature_cols].values
    y = df_feat['label'].values
    groups = df_feat['patient_id'].values
    
    cv_accs, cv_recs, cv_f1s, cv_rocs, cv_paccs = [], [], [], [], []
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42 + fold, verbose=-1)
        clf.fit(X[tr_idx], y[tr_idx])
        preds = clf.predict(X[va_idx])
        probs = clf.predict_proba(X[va_idx])[:, 1]
        
        cv_accs.append(accuracy_score(y[va_idx], preds))
        cv_recs.append(recall_score(y[va_idx], preds, zero_division=0))
        cv_f1s.append(f1_score(y[va_idx], preds, zero_division=0))
        cv_rocs.append(roc_auc_score(y[va_idx], probs))
        
        val_df = pd.DataFrame({'pid': groups[va_idx], 'true': y[va_idx], 'prob': probs})
        p_agg = val_df.groupby('pid').agg({'true': 'first', 'prob': 'mean'})
        p_agg['pred'] = (p_agg['prob'] >= 0.5).astype(int)
        cv_paccs.append(accuracy_score(p_agg['true'], p_agg['pred']))
        
    print("\n" + "="*75)
    print("5-FOLD STRATIFIED GROUP CROSS-VALIDATION RESULTS")
    print("="*75)
    print(f"Window Accuracy:          {np.mean(cv_accs)*100:.2f}% ± {np.std(cv_accs)*100:.2f}%")
    print(f"Window Recall (AF Sens):  {np.mean(cv_recs)*100:.2f}% ± {np.std(cv_recs)*100:.2f}%")
    print(f"Window F1-Score:          {np.mean(cv_f1s)*100:.2f}% ± {np.std(cv_f1s)*100:.2f}%")
    print(f"Window ROC-AUC:           {np.mean(cv_rocs):.4f} ± {np.std(cv_rocs):.4f}")
    print(f"Patient-Level Accuracy:   {np.mean(cv_paccs)*100:.2f}% ± {np.std(cv_paccs)*100:.2f}%")
    print("="*75)

if __name__ == "__main__":
    main()
