# HFAC Greenhouse Control System - UTS Data-Driven

Sistem kontrol greenhouse menggunakan berbagai algoritma machine learning dan control theory untuk mengoptimalkan kondisi lingkungan tanaman.

## Struktur File

```
src/
├── genenratedata.py              # Generate synthetic dataset
├── NN_training.py                # Neural Network training
├── mpc_control.py                # Model Predictive Control
├── qlearning_control.py          # Q-Learning (RL)
├── comparison_analysis.py        # Perbandingan semua metode
├── run_all.py                    # Run all scripts
├── requirements.txt              # Dependencies
└── README.md                     # File ini
```

## Cara Menjalankan

### 1. Generate Dataset
```bash
python genenratedata.py
```
**Output:** `../data/hfac_greenhouse_dataset.csv` (10,080 samples - simulasi 7 hari)

### 2. Training Neural Network
```bash
python NN_training.py
```
**Output:** 
- `../models/hfac_model.h5` (trained model)
- `../models/scaler_X.pkl`, `../models/scaler_y.pkl` (scalers)
- `../images/training_history.png` (training curves)
- `../images/prediction_vs_actual.png` (validation results)

### 3. Model Predictive Control (MPC)
```bash
python mpc_control.py
```
**Output:**
- `../data/mpc_control_results.csv` (control results)
- `../images/mpc_control_results.png` (visualization)

### 4. Q-Learning (Reinforcement Learning)
```bash
python qlearning_control.py
```
**Output:**
- `../models/qlearning_qtable.pkl` (trained Q-table)
- `../data/qlearning_control_results.csv` (control results)
- `../data/qlearning_training_history.csv` (training metrics)
- `../images/qlearning_control_results.png` (visualization)

### 5. Comparison Analysis
```bash
python comparison_analysis.py
```
**Output:**
- `../data/algorithm_comparison.csv` (metrics comparison)
- `../images/algorithm_comparison.png` (comparison charts)

## Algoritma yang Digunakan

### 1. Neural Network (Deep Learning)
- **Library:** TensorFlow/Keras
- **Arsitektur:** Dense layers dengan dropout
- **Kelebihan:** Akurat untuk pattern recognition, cepat inference
- **Kekurangan:** Membutuhkan banyak data, black-box

### 2. Model Predictive Control (MPC)
- **Library:** scipy.optimize
- **Metode:** Optimization-based control
- **Kelebihan:** Optimal control, dapat handle constraints
- **Kekurangan:** Komputasi berat, butuh model sistem akurat

### 3. Q-Learning (Reinforcement Learning)
- **Library:** NumPy (tabular Q-learning)
- **Metode:** Model-free RL
- **Kelebihan:** Tidak perlu model sistem, adaptif
- **Kekurangan:** Training lama, curse of dimensionality

## Target Kontrol

- **Temperature:** 25 C (tolerance 1 C)
- **Humidity:** 65% (tolerance 3%)
- **Light Intensity:** 70% (tolerance 5%)

## Metrics Evaluasi

- **MAE (Mean Absolute Error):** Error rata-rata
- **RMSE (Root Mean Square Error):** Error dengan penalti untuk outlier
- **Control Effort:** Average PWM usage (energy efficiency)
- **Settling Time:** Waktu untuk mencapai tolerance

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scipy joblib
```

## Catatan Penting

1. **Urutan Eksekusi:** Jalankan file sesuai urutan di atas
2. **Dataset:** File `genenratedata.py` harus dijalankan pertama kali
3. **Training:** Neural Network perlu training dulu sebelum comparison
4. **Comparison:** Pastikan semua metode sudah dijalankan sebelum comparison_analysis.py

## Author

**Alvin** - UTS Data-Driven Exam

## Last Updated

1 Desember 2025
