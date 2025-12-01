# HFAC Greenhouse Control System - Project Structure

Proyek ini telah direorganisasi dengan struktur folder yang lebih teratur untuk memisahkan data, gambar, model, dan source code.

## Struktur Folder

```
UTS/
├── data/                          # Dataset dan hasil CSV
│   ├── hfac_greenhouse_dataset.csv
│   ├── mpc_control_results.csv
│   ├── qlearning_control_results.csv
│   ├── qlearning_training_history.csv
│   └── algorithm_comparison.csv
│
├── images/                        # Visualisasi dan grafik
│   ├── prediction_vs_actual.png
│   ├── training_history.png
│   ├── algorithm_comparison.png
│   ├── mpc_control_results.png
│   └── qlearning_control_results.png
│
├── models/                        # Model terlatih dan scaler
│   ├── hfac_model.h5
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   └── qlearning_qtable.pkl
│
├── src/                          # Source code Python
│   ├── NN_training.py
│   ├── genenratedata.py
│   ├── comparison_analysis.py
│   ├── mpc_control.py
│   ├── qlearning_control.py
│   ├── run_all.py
│   ├── requirements.txt
│   └── README.md
│
└── usemodel.py                   # GUI Application (run from UTS root)
```

## Cara Menjalankan

### 1. Training Model Neural Network

```powershell
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS\src
python NN_training.py
```

**Output:**
- Model: `../models/hfac_model.h5`
- Scalers: `../models/scaler_X.pkl`, `../models/scaler_y.pkl`
- Grafik: `../images/training_history.png`, `../images/prediction_vs_actual.png`

### 2. Generate Dataset

```powershell
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS\src
python genenratedata.py
```

**Output:**
- Dataset: `../data/hfac_greenhouse_dataset.csv`

### 3. Comparison Analysis

```powershell
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS\src
python comparison_analysis.py
```

**Output:**
- Hasil: `../data/algorithm_comparison.csv`
- Grafik: `../images/algorithm_comparison.png`

### 4. GUI Application

```powershell
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS
python usemodel.py
```

**Catatan:** GUI harus dijalankan dari root folder UTS karena menggunakan path `models/`

## Dependencies

Install semua dependencies dengan:

```powershell
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS\src
pip install -r requirements.txt
```

## Deskripsi File

### Source Code (src/)

- **NN_training.py**: Script untuk melatih Neural Network model
- **genenratedata.py**: Script untuk generate synthetic dataset
- **comparison_analysis.py**: Analisis perbandingan algoritma (NN, MPC, Q-Learning)
- **mpc_control.py**: Implementasi Model Predictive Control
- **qlearning_control.py**: Implementasi Q-Learning Reinforcement Learning
- **run_all.py**: Script untuk menjalankan semua proses sekaligus

### Data (data/)

- **hfac_greenhouse_dataset.csv**: Dataset utama untuk training
- **mpc_control_results.csv**: Hasil eksperimen MPC
- **qlearning_control_results.csv**: Hasil eksperimen Q-Learning
- **algorithm_comparison.csv**: Tabel perbandingan performa algoritma

### Models (models/)

- **hfac_model.h5**: Model Neural Network terlatih
- **scaler_X.pkl**: StandardScaler untuk input features
- **scaler_y.pkl**: StandardScaler untuk output targets
- **qlearning_qtable.pkl**: Q-Table untuk Q-Learning

### Images (images/)

Semua visualisasi hasil training dan eksperimen

## Workflow Lengkap

1. **Generate Data** - genenratedata.py
2. **Train Model** - NN_training.py
3. **Run Experiments** - mpc_control.py, qlearning_control.py
4. **Compare Results** - comparison_analysis.py
5. **Use Model** - usemodel.py (GUI)

## Important Notes

- Semua script di folder `src/` menggunakan relative paths (`../data/`, `../models/`, `../images/`)
- Jalankan script dari dalam folder `src/` kecuali `usemodel.py`
- `usemodel.py` harus dijalankan dari root folder `UTS/`

## Contact

Untuk pertanyaan atau issues, silakan hubungi maintainer proyek.
