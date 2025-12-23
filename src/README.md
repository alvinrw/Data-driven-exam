# HFAC Greenhouse Control System - Source Code

Greenhouse control system using machine learning and control theory algorithms to optimize plant environmental conditions.

## File Structure

```
src/
├── genenratedata.py          # Synthetic dataset generation
├── NN_training.py            # Neural Network training
├── mpc_control.py            # Model Predictive Control
├── qlearning_control.py      # Q-Learning reinforcement learning
├── comparison_analysis.py    # Algorithm comparison
├── run_all.py                # Automated execution script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Execution Order

### 1. Generate Dataset
```bash
python genenratedata.py
```
**Output**: `../data/hfac_greenhouse_dataset.csv` (10,080 samples - 7 days simulation)

### 2. Train Neural Network
```bash
python NN_training.py
```
**Output**:
- `../models/hfac_model.h5` (trained model)
- `../models/scaler_X.pkl`, `../models/scaler_y.pkl` (scalers)
- `../images/training_history.png` (training curves)
- `../images/prediction_vs_actual.png` (validation results)

### 3. Model Predictive Control
```bash
python mpc_control.py
```
**Output**:
- `../data/mpc_control_results.csv` (control results)
- `../images/mpc_control_results.png` (visualization)

### 4. Q-Learning Training
```bash
python qlearning_control.py
```
**Output**:
- `../models/qlearning_qtable.pkl` (trained Q-table)
- `../data/qlearning_control_results.csv` (control results)
- `../data/qlearning_training_history.csv` (training metrics)
- `../images/qlearning_control_results.png` (visualization)

### 5. Comparison Analysis
```bash
python comparison_analysis.py
```
**Output**:
- `../data/algorithm_comparison.csv` (metrics comparison)
- `../images/algorithm_comparison.png` (comparison charts)

## Algorithms

### Neural Network (Deep Learning)
- **Library**: TensorFlow/Keras
- **Architecture**: Dense layers with dropout and batch normalization
- **Advantages**: High accuracy, fast inference
- **Disadvantages**: Requires training data, black-box model

### Model Predictive Control (MPC)
- **Library**: scipy.optimize
- **Method**: Optimization-based control
- **Advantages**: Optimal control, handles constraints
- **Disadvantages**: Computationally intensive, requires system model

### Q-Learning (Reinforcement Learning)
- **Library**: NumPy (tabular Q-learning)
- **Method**: Model-free reinforcement learning
- **Advantages**: No system model needed, adaptive
- **Disadvantages**: Long training time, curse of dimensionality

## Control Targets

- **Temperature**: 25°C (tolerance ±1°C)
- **Humidity**: 65% (tolerance ±3%)
- **Light Intensity**: 70% (tolerance ±5%)

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average tracking error
- **RMSE (Root Mean Square Error)**: Error with outlier penalty
- **Control Effort**: Average PWM usage (energy efficiency)
- **Settling Time**: Steps to reach tolerance

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scipy joblib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Important Notes

1. **Execution Order**: Run files in the order listed above
2. **Dataset**: Run `genenratedata.py` first
3. **Training**: Neural Network must be trained before comparison
4. **Comparison**: Ensure all methods are executed before running `comparison_analysis.py`

## Automated Execution

To run all scripts automatically:
```bash
python run_all.py
```

**Warning**: This will take 10-20 minutes to complete all processes.

## Author

Alvin - Data-Driven Systems Exam

## Last Updated

December 23, 2025
