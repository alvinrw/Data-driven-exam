"""
Model Predictive Control (MPC) for Greenhouse Control System

Algorithm: Model Predictive Control
Description: Control method that predicts future system behavior and optimizes
             control actions to achieve desired targets.

Principles:
1. Predict system state several steps ahead (prediction horizon)
2. Optimize control input to minimize error from target
3. Execute first control input, then repeat (receding horizon)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=" * 70)
print("MODEL PREDICTIVE CONTROL (MPC) - GREENHOUSE CONTROL SYSTEM")
print("=" * 70)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n[1] Loading dataset...")
df = pd.read_csv('hfac_greenhouse_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} samples")

# Ambil subset untuk testing (100 samples untuk demo)
test_samples = 500
df_test = df.head(test_samples).copy()

# ==========================================
# 2. MPC PARAMETERS
# ==========================================

print("\n[2] Setting up MPC parameters...")

# Target setpoints (nilai ideal greenhouse)
TARGET_TEMP = 25.0      # °C
TARGET_HUMIDITY = 65.0  # %
TARGET_LIGHT = 70.0     # %

# MPC Horizon
PREDICTION_HORIZON = 5  # Prediksi 5 step ke depan
CONTROL_HORIZON = 3     # Optimasi 3 control actions

# Weights untuk cost function (seberapa penting setiap variabel)
W_TEMP = 2.0      # Weight untuk temperature error
W_HUMIDITY = 1.5  # Weight untuk humidity error
W_LIGHT = 1.0     # Weight untuk light error
W_CONTROL = 0.1   # Weight untuk control effort (penalti PWM tinggi)

# System dynamics parameters (simplified linear model)
# dx/dt = A*x + B*u
# Asumsi: perubahan state proporsional dengan control input

# Time constant (seberapa cepat sistem merespon)
TAU_TEMP = 10.0      # Temperature time constant (menit)
TAU_HUMIDITY = 8.0   # Humidity time constant (menit)
TAU_LIGHT = 2.0      # Light time constant (menit)

DT = 1.0  # Time step (1 menit)

print(f"Prediction Horizon: {PREDICTION_HORIZON} steps")
print(f"Control Horizon: {CONTROL_HORIZON} steps")
print(f"Target: Temp={TARGET_TEMP}°C, Humidity={TARGET_HUMIDITY}%, Light={TARGET_LIGHT}%")

# ==========================================
# 3. SYSTEM MODEL (Simplified First-Order)
# ==========================================

def predict_next_state(current_state, control_input):
    """
    Prediksi state berikutnya berdasarkan model first-order
    
    Args:
        current_state: [temp, humidity, light]
        control_input: [fan_cooling_pwm, fan_circulation_pwm, water_pump_pwm, grow_light_pwm]
    
    Returns:
        next_state: [temp_next, humidity_next, light_next]
    """
    temp, humidity, light = current_state
    fan_cooling, fan_circulation, water_pump, grow_light = control_input
    
    # Temperature dynamics
    # Fan cooling menurunkan suhu
    cooling_effect = -fan_cooling * 0.15  # Cooling rate
    temp_next = temp + (cooling_effect - (temp - TARGET_TEMP) / TAU_TEMP) * DT
    
    # Humidity dynamics
    # Water pump menaikkan humidity, fan circulation menurunkan humidity
    humidify_effect = water_pump * 0.2
    dehumidify_effect = -fan_circulation * 0.1
    humidity_next = humidity + (humidify_effect + dehumidify_effect - (humidity - TARGET_HUMIDITY) / TAU_HUMIDITY) * DT
    
    # Light dynamics
    # Grow light menaikkan intensitas cahaya
    light_effect = grow_light * 0.3
    light_next = light + (light_effect - (light - TARGET_LIGHT) / TAU_LIGHT) * DT
    
    # Clip to physical limits
    temp_next = np.clip(temp_next, 15, 40)
    humidity_next = np.clip(humidity_next, 30, 90)
    light_next = np.clip(light_next, 0, 100)
    
    return np.array([temp_next, humidity_next, light_next])

# ==========================================
# 4. MPC COST FUNCTION
# ==========================================

def mpc_cost_function(control_sequence, current_state):
    """
    Cost function untuk MPC optimization
    
    Minimize: Sum of (state_error^2 + control_effort^2) over prediction horizon
    
    Args:
        control_sequence: Flattened array of control inputs [u0, u1, ..., u_{N-1}]
                         Each u_i = [fan_cooling, fan_circulation, water_pump, grow_light]
        current_state: Current system state [temp, humidity, light]
    
    Returns:
        total_cost: Scalar cost value
    """
    # Reshape control sequence
    control_sequence = control_sequence.reshape(CONTROL_HORIZON, 4)
    
    # Initialize
    state = current_state.copy()
    total_cost = 0.0
    
    # Predict over horizon
    for i in range(PREDICTION_HORIZON):
        # Use control input (repeat last control if beyond control horizon)
        if i < CONTROL_HORIZON:
            u = control_sequence[i]
        else:
            u = control_sequence[-1]
        
        # Predict next state
        state = predict_next_state(state, u)
        
        # State error cost
        temp_error = (state[0] - TARGET_TEMP) ** 2
        humidity_error = (state[1] - TARGET_HUMIDITY) ** 2
        light_error = (state[2] - TARGET_LIGHT) ** 2
        
        state_cost = W_TEMP * temp_error + W_HUMIDITY * humidity_error + W_LIGHT * light_error
        
        # Control effort cost (penalize high PWM usage)
        control_cost = W_CONTROL * np.sum(u ** 2)
        
        total_cost += state_cost + control_cost
    
    return total_cost

# ==========================================
# 5. MPC CONTROLLER
# ==========================================

def mpc_controller(current_state, previous_control=None):
    """
    MPC controller yang menghitung optimal control input
    
    Args:
        current_state: [temp, humidity, light]
        previous_control: Previous control input (untuk warm start)
    
    Returns:
        optimal_control: [fan_cooling_pwm, fan_circulation_pwm, water_pump_pwm, grow_light_pwm]
    """
    # Initial guess (warm start dengan previous control atau zero)
    if previous_control is not None:
        # Shift previous sequence dan tambah zero di akhir
        u0 = np.vstack([previous_control[1:], previous_control[-1:]])
    else:
        u0 = np.zeros((CONTROL_HORIZON, 4))
    
    u0 = u0.flatten()
    
    # Bounds: PWM harus antara 0-100
    bounds = [(0, 100)] * (CONTROL_HORIZON * 4)
    
    # Optimization
    result = minimize(
        mpc_cost_function,
        u0,
        args=(current_state,),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-6}
    )
    
    # Extract optimal control sequence
    optimal_sequence = result.x.reshape(CONTROL_HORIZON, 4)
    
    # Return first control input (receding horizon principle)
    return optimal_sequence[0], optimal_sequence

# ==========================================
# 6. RUN MPC SIMULATION
# ==========================================

print("\n[3] Running MPC simulation...")

# Storage untuk hasil
results = {
    'temperature': [],
    'humidity': [],
    'light_intensity': [],
    'fan_cooling_pwm': [],
    'fan_circulation_pwm': [],
    'water_pump_pwm': [],
    'grow_light_pwm': []
}

# Initial state
current_state = np.array([
    df_test.iloc[0]['temperature'],
    df_test.iloc[0]['humidity'],
    df_test.iloc[0]['light_intensity']
])

previous_control_sequence = None

# Simulate
for i in range(test_samples):
    # Get optimal control using MPC
    optimal_control, control_sequence = mpc_controller(current_state, previous_control_sequence)
    previous_control_sequence = control_sequence
    
    # Store results
    results['temperature'].append(current_state[0])
    results['humidity'].append(current_state[1])
    results['light_intensity'].append(current_state[2])
    results['fan_cooling_pwm'].append(optimal_control[0])
    results['fan_circulation_pwm'].append(optimal_control[1])
    results['water_pump_pwm'].append(optimal_control[2])
    results['grow_light_pwm'].append(optimal_control[3])
    
    # Update state (simulate real system response)
    # Dalam real system, ini akan datang dari sensor
    # Untuk simulasi, kita pakai model prediction
    current_state = predict_next_state(current_state, optimal_control)
    
    # Progress
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{test_samples} samples...")

print("MPC simulation completed!")

# ==========================================
# 7. EVALUATION
# ==========================================

print("\n[4] Evaluating MPC performance...")

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Calculate tracking errors
temp_error = np.abs(df_results['temperature'] - TARGET_TEMP)
humidity_error = np.abs(df_results['humidity'] - TARGET_HUMIDITY)
light_error = np.abs(df_results['light_intensity'] - TARGET_LIGHT)

# Metrics
mae_temp = np.mean(temp_error)
mae_humidity = np.mean(humidity_error)
mae_light = np.mean(light_error)

rmse_temp = np.sqrt(np.mean(temp_error ** 2))
rmse_humidity = np.sqrt(np.mean(humidity_error ** 2))
rmse_light = np.sqrt(np.mean(light_error ** 2))

print("\n" + "=" * 70)
print("MPC PERFORMANCE METRICS")
print("=" * 70)
print(f"\nTemperature Control:")
print(f"  MAE:  {mae_temp:.4f} °C")
print(f"  RMSE: {rmse_temp:.4f} °C")

print(f"\nHumidity Control:")
print(f"  MAE:  {mae_humidity:.4f} %")
print(f"  RMSE: {rmse_humidity:.4f} %")

print(f"\nLight Control:")
print(f"  MAE:  {mae_light:.4f} %")
print(f"  RMSE: {rmse_light:.4f} %")

# Average control effort
avg_fan_cooling = np.mean(df_results['fan_cooling_pwm'])
avg_fan_circulation = np.mean(df_results['fan_circulation_pwm'])
avg_water_pump = np.mean(df_results['water_pump_pwm'])
avg_grow_light = np.mean(df_results['grow_light_pwm'])

print(f"\nAverage Control Effort (PWM %):")
print(f"  Fan Cooling:     {avg_fan_cooling:.2f}%")
print(f"  Fan Circulation: {avg_fan_circulation:.2f}%")
print(f"  Water Pump:      {avg_water_pump:.2f}%")
print(f"  Grow Light:      {avg_grow_light:.2f}%")

# ==========================================
# 8. VISUALIZATION
# ==========================================

print("\n[5] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Temperature Control
axes[0, 0].plot(df_results['temperature'], label='Actual Temperature', linewidth=2)
axes[0, 0].axhline(y=TARGET_TEMP, color='r', linestyle='--', label='Target', linewidth=2)
axes[0, 0].fill_between(range(len(df_results)), TARGET_TEMP - 1, TARGET_TEMP + 1, 
                         alpha=0.2, color='green', label='±1°C tolerance')
axes[0, 0].set_xlabel('Time (minutes)', fontsize=11)
axes[0, 0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0, 0].set_title('MPC Temperature Control', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Humidity Control
axes[0, 1].plot(df_results['humidity'], label='Actual Humidity', linewidth=2, color='blue')
axes[0, 1].axhline(y=TARGET_HUMIDITY, color='r', linestyle='--', label='Target', linewidth=2)
axes[0, 1].fill_between(range(len(df_results)), TARGET_HUMIDITY - 3, TARGET_HUMIDITY + 3, 
                         alpha=0.2, color='green', label='±3% tolerance')
axes[0, 1].set_xlabel('Time (minutes)', fontsize=11)
axes[0, 1].set_ylabel('Humidity (%)', fontsize=11)
axes[0, 1].set_title('MPC Humidity Control', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Light Control
axes[1, 0].plot(df_results['light_intensity'], label='Actual Light', linewidth=2, color='orange')
axes[1, 0].axhline(y=TARGET_LIGHT, color='r', linestyle='--', label='Target', linewidth=2)
axes[1, 0].fill_between(range(len(df_results)), TARGET_LIGHT - 5, TARGET_LIGHT + 5, 
                         alpha=0.2, color='green', label='±5% tolerance')
axes[1, 0].set_xlabel('Time (minutes)', fontsize=11)
axes[1, 0].set_ylabel('Light Intensity (%)', fontsize=11)
axes[1, 0].set_title('MPC Light Control', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Control Inputs
axes[1, 1].plot(df_results['fan_cooling_pwm'], label='Fan Cooling', linewidth=1.5, alpha=0.8)
axes[1, 1].plot(df_results['fan_circulation_pwm'], label='Fan Circulation', linewidth=1.5, alpha=0.8)
axes[1, 1].plot(df_results['water_pump_pwm'], label='Water Pump', linewidth=1.5, alpha=0.8)
axes[1, 1].plot(df_results['grow_light_pwm'], label='Grow Light', linewidth=1.5, alpha=0.8)
axes[1, 1].set_xlabel('Time (minutes)', fontsize=11)
axes[1, 1].set_ylabel('PWM (%)', fontsize=11)
axes[1, 1].set_title('MPC Control Inputs', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mpc_control_results.png', dpi=300, bbox_inches='tight')
print("Saved: mpc_control_results.png")

# ==========================================
# 9. SAVE RESULTS
# ==========================================

print("\n[6] Saving results...")
df_results.to_csv('mpc_control_results.csv', index=False)
print("Saved: mpc_control_results.csv")

print("\n" + "=" * 70)
print("MPC CONTROL SIMULATION COMPLETED!")
print("=" * 70)
print("\nFiles generated:")
print("  - mpc_control_results.csv (control data)")
print("  - mpc_control_results.png (visualization)")
