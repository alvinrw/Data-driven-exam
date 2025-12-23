# HFAC Greenhouse Control System

A comprehensive greenhouse environmental control system implementing and comparing three advanced control algorithms: Neural Networks, Model Predictive Control (MPC), and Q-Learning Reinforcement Learning.

## Features

- **Multi-Algorithm Support**: Compare Neural Network, MPC, and Q-Learning approaches
- **Real-time Control**: Automated control of temperature, humidity, light, and air circulation
- **Interactive GUI**: User-friendly interface for testing and visualization
- **Comprehensive Analysis**: Performance metrics and comparative visualizations
- **Synthetic Data Generation**: Physics-based dataset generation for training and testing

## Quick Start

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scipy joblib
```

### Basic Usage

1. **Generate Dataset**
```bash
cd src
python genenratedata.py
```

2. **Train Neural Network**
```bash
python NN_training.py
```

3. **Run GUI Application**
```bash
cd ..
python usemodel.py
```

## Project Structure

```
UTS/
├── data/                  # Datasets and results
├── images/                # Visualizations and plots
├── models/                # Trained models and scalers
├── src/                   # Source code
│   ├── NN_training.py
│   ├── mpc_control.py
│   ├── qlearning_control.py
│   ├── comparison_analysis.py
│   ├── genenratedata.py
│   └── run_all.py
├── usemodel.py            # GUI application
└── usemodelV2.py          # Enhanced GUI with comparison features
```

## System Concept

This project demonstrates a complete data-driven control system workflow:

1. **Data Generation** (`genenratedata.py`): Creates physics-based synthetic greenhouse data simulating 7 days of operation with realistic sensor readings and actuator responses.

2. **Model Training** (`NN_training.py`, `qlearning_control.py`): Trains intelligent control models using the generated data to learn optimal control strategies.

3. **Control Simulation** (`mpc_control.py`): Implements optimization-based control without requiring training data.

4. **Performance Comparison** (`comparison_analysis.py`): Evaluates all three algorithms to determine which performs best under different conditions.

5. **Real-time Application** (`usemodel.py`, `usemodelV2.py`): Provides interactive interfaces to test and visualize the trained models in action.

The system controls four actuators (cooling fan, circulation fan, water pump, grow light) based on four sensor inputs (temperature, humidity, light intensity, motion) to maintain optimal greenhouse conditions.

## Control Algorithms

## Control Algorithms

### Neural Network
- Deep learning model with 4 hidden layers
- Fast inference after training
- Excellent for pattern recognition
- Requires training data

### Model Predictive Control (MPC)
- Optimization-based control
- Handles constraints effectively
- Predictive horizon planning
- No training required

### Q-Learning
- Reinforcement learning approach
- Model-free learning
- Adaptive to environment changes
- Learns optimal policy through experience

## Target Control Parameters

- **Temperature**: 25°C (±1°C tolerance)
- **Humidity**: 65% (±3% tolerance)
- **Light Intensity**: 70% (±5% tolerance)

## Performance Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **Control Effort**: Average PWM usage
- **Settling Time**: Time to reach tolerance

## Running Complete Analysis

To run all algorithms and generate comparison:

```bash
cd src
python run_all.py
```

This will:
1. Generate synthetic dataset (10,080 samples)
2. Train Neural Network model
3. Run MPC control simulation
4. Train Q-Learning agent
5. Generate comparative analysis

## GUI Applications

The project includes two GUI applications with different capabilities:

### usemodel.py - Basic GUI
**Purpose**: Test a single control algorithm at a time

**Features**:
- Select one algorithm (Neural Network, MPC, or Q-Learning)
- Input current greenhouse conditions
- Get instant actuator predictions
- Simulate transition from current to target conditions
- Visualize control outputs with charts

**Best for**: Testing individual algorithms, understanding how each method works

**Usage**:
```bash
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS
python usemodel.py
```

To change the algorithm, edit line 13 in `usemodel.py`:
```python
SELECTED_MODEL = "NEURAL_NETWORK"  # or "MPC" or "QLEARNING"
```

### usemodelV2.py - Advanced GUI
**Purpose**: Compare all three algorithms simultaneously

**Features**:
- All features from basic GUI
- Side-by-side comparison of all three algorithms
- Performance metrics for each method
- Energy distribution analysis
- Detailed error tracking
- Export results to PNG

**Best for**: Algorithm comparison, research analysis, presentation materials

**Usage**:
```bash
cd c:\Users\alvin\Documents\vscode_apin\Datadriven\UTS
python usemodelV2.py
```

**Key Difference**: 
- `usemodel.py` = Single algorithm testing
- `usemodelV2.py` = Multi-algorithm comparison with advanced analytics

## Documentation

- See `src/README.md` for detailed source code documentation

## Author

Alvin - Data-Driven Systems Exam

## License

Educational project for academic purposes.
