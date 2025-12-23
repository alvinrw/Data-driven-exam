"""
Greenhouse Control Algorithm Comparison

This file compares the performance of various control methods:
1. Neural Network (Deep Learning)
2. Model Predictive Control (MPC)
3. Q-Learning (Reinforcement Learning)

Goal: Comparative analysis to determine the best method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("COMPARATIVE ANALYSIS - GREENHOUSE CONTROL ALGORITHMS")
print("=" * 70)

# ==========================================
# 1. LOAD RESULTS FROM ALL METHODS
# ==========================================

print("\n[1] Loading results from all control methods...")

# Note: Jalankan file-file berikut terlebih dahulu:
# - NN_training.py (untuk Neural Network)
# - mpc_control.py (untuk MPC)
# - qlearning_control.py (untuk Q-Learning)

try:
    # Neural Network results (dari testing phase)
    # Kita akan load dari dataset asli dan prediksi model
    import tensorflow as tf
    from tensorflow import keras
    import joblib
    
    print("  Loading Neural Network model...")
    nn_model = keras.models.load_model(
        '../models/hfac_model.h5',
        custom_objects={'mse': tf.keras.losses.MeanSquaredError}
    )
    scaler_X = joblib.load('../models/scaler_X.pkl')
    scaler_y = joblib.load('../models/scaler_y.pkl')
    
    # Load test data
    df = pd.read_csv('../data/hfac_greenhouse_dataset.csv')
    test_samples = 500
    df_test = df.head(test_samples)
    
    X_test = df_test[['temperature', 'humidity', 'light_intensity', 'motion']].values
    X_test_scaled = scaler_X.transform(X_test)
    
    y_pred_scaled = nn_model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = np.clip(y_pred, 0, 100)
    
    nn_results = pd.DataFrame({
        'temperature': df_test['temperature'].values,
        'humidity': df_test['humidity'].values,
        'light_intensity': df_test['light_intensity'].values,
        'fan_cooling_pwm': y_pred[:, 0],
        'fan_circulation_pwm': y_pred[:, 1],
        'water_pump_pwm': y_pred[:, 2],
        'grow_light_pwm': y_pred[:, 3]
    })
    nn_available = True
    print("  [OK] Neural Network results loaded")
    
except Exception as e:
    print(f"  [ERROR] Neural Network results not available: {e}")
    print(f"  Make sure to run NN_training.py first!")
    nn_available = False

try:
    print("  Loading MPC results...")
    mpc_results = pd.read_csv('../data/mpc_control_results.csv')
    mpc_available = True
    print("  [OK] MPC results loaded")
except Exception as e:
    print(f"  [ERROR] MPC results not available: {e}")
    mpc_available = False

try:
    print("  Loading Q-Learning results...")
    ql_results = pd.read_csv('../data/qlearning_control_results.csv')
    ql_available = True
    print("  [OK] Q-Learning results loaded")
except Exception as e:
    print(f"  [ERROR] Q-Learning results not available: {e}")
    ql_available = False

# ==========================================
# 2. CALCULATE PERFORMANCE METRICS
# ==========================================

print("\n[2] Calculating performance metrics...")

TARGET_TEMP = 25.0
TARGET_HUMIDITY = 65.0
TARGET_LIGHT = 70.0

def calculate_metrics(df, method_name):
    """Calculate control performance metrics"""
    
    # Tracking errors
    temp_error = np.abs(df['temperature'] - TARGET_TEMP)
    humidity_error = np.abs(df['humidity'] - TARGET_HUMIDITY)
    light_error = np.abs(df['light_intensity'] - TARGET_LIGHT)
    
    # MAE
    mae_temp = np.mean(temp_error)
    mae_humidity = np.mean(humidity_error)
    mae_light = np.mean(light_error)
    mae_overall = (mae_temp + mae_humidity + mae_light) / 3
    
    # RMSE
    rmse_temp = np.sqrt(np.mean(temp_error ** 2))
    rmse_humidity = np.sqrt(np.mean(humidity_error ** 2))
    rmse_light = np.sqrt(np.mean(light_error ** 2))
    rmse_overall = (rmse_temp + rmse_humidity + rmse_light) / 3
    
    # Control effort (average PWM usage)
    avg_pwm = (
        df['fan_cooling_pwm'].mean() +
        df['fan_circulation_pwm'].mean() +
        df['water_pump_pwm'].mean() +
        df['grow_light_pwm'].mean()
    ) / 4
    
    # Settling time (berapa lama mencapai tolerance)
    in_tolerance = (temp_error < 1.0) & (humidity_error < 3.0) & (light_error < 5.0)
    settling_idx = np.argmax(in_tolerance) if np.any(in_tolerance) else len(df)
    settling_time = settling_idx
    
    return {
        'method': method_name,
        'mae_temp': mae_temp,
        'mae_humidity': mae_humidity,
        'mae_light': mae_light,
        'mae_overall': mae_overall,
        'rmse_temp': rmse_temp,
        'rmse_humidity': rmse_humidity,
        'rmse_light': rmse_light,
        'rmse_overall': rmse_overall,
        'avg_pwm': avg_pwm,
        'settling_time': settling_time
    }

# Calculate metrics for each method
metrics_list = []

if nn_available:
    metrics_list.append(calculate_metrics(nn_results, 'Neural Network'))

if mpc_available:
    metrics_list.append(calculate_metrics(mpc_results, 'MPC'))

if ql_available:
    metrics_list.append(calculate_metrics(ql_results, 'Q-Learning'))

df_metrics = pd.DataFrame(metrics_list)

# ==========================================
# 3. DISPLAY COMPARISON TABLE
# ==========================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 70)

if len(df_metrics) > 0:
    print("\n1. TRACKING ERROR (Lower is better)")
    print("-" * 70)
    print(f"{'Method':<20} {'Temp MAE':<12} {'Humidity MAE':<15} {'Light MAE':<12}")
    print("-" * 70)
    for _, row in df_metrics.iterrows():
        print(f"{row['method']:<20} {row['mae_temp']:<12.4f} {row['mae_humidity']:<15.4f} {row['mae_light']:<12.4f}")
    
    print("\n2. OVERALL PERFORMANCE")
    print("-" * 70)
    print(f"{'Method':<20} {'Overall MAE':<15} {'Overall RMSE':<15} {'Avg PWM %':<12}")
    print("-" * 70)
    for _, row in df_metrics.iterrows():
        print(f"{row['method']:<20} {row['mae_overall']:<15.4f} {row['rmse_overall']:<15.4f} {row['avg_pwm']:<12.2f}")
    
    print("\n3. SETTLING TIME (Steps to reach tolerance)")
    print("-" * 70)
    print(f"{'Method':<20} {'Settling Time (steps)':<25}")
    print("-" * 70)
    for _, row in df_metrics.iterrows():
        print(f"{row['method']:<20} {row['settling_time']:<25}")
    
    # Determine best method
    print("\n" + "=" * 70)
    print("RANKING")
    print("=" * 70)
    
    # Rank by overall MAE (lower is better)
    df_ranked = df_metrics.sort_values('mae_overall')
    print("\nRanking by Overall MAE (Best to Worst):")
    for i, (_, row) in enumerate(df_ranked.iterrows(), 1):
        print(f"  {i}. {row['method']:<20} (MAE: {row['mae_overall']:.4f})")
    
else:
    print("No results available. Please run the control algorithms first.")

# ==========================================
# 4. VISUALIZATION
# ==========================================

if len(df_metrics) > 0:
    print("\n[3] Creating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    methods = df_metrics['method'].values
    
    # Plot 1: MAE Comparison
    x = np.arange(len(methods))
    width = 0.25
    
    axes[0, 0].bar(x - width, df_metrics['mae_temp'], width, label='Temperature', alpha=0.8)
    axes[0, 0].bar(x, df_metrics['mae_humidity'], width, label='Humidity', alpha=0.8)
    axes[0, 0].bar(x + width, df_metrics['mae_light'], width, label='Light', alpha=0.8)
    axes[0, 0].set_xlabel('Method', fontsize=11)
    axes[0, 0].set_ylabel('MAE', fontsize=11)
    axes[0, 0].set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Overall Performance
    axes[0, 1].bar(methods, df_metrics['mae_overall'], alpha=0.8, color='steelblue')
    axes[0, 1].set_xlabel('Method', fontsize=11)
    axes[0, 1].set_ylabel('Overall MAE', fontsize=11)
    axes[0, 1].set_title('Overall Performance (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Control Effort
    axes[1, 0].bar(methods, df_metrics['avg_pwm'], alpha=0.8, color='coral')
    axes[1, 0].set_xlabel('Method', fontsize=11)
    axes[1, 0].set_ylabel('Average PWM (%)', fontsize=11)
    axes[1, 0].set_title('Control Effort (Energy Usage)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Settling Time (Normalized to percentage)
    # Normalize settling time to 0-100 scale for better visualization
    max_settling = df_metrics['settling_time'].max()
    settling_normalized = (df_metrics['settling_time'] / max_settling) * 100
    
    bars = axes[1, 1].bar(methods, settling_normalized, alpha=0.8, color='mediumseagreen')
    axes[1, 1].set_xlabel('Method', fontsize=11)
    axes[1, 1].set_ylabel('Settling Time (normalized %)', fontsize=11)
    axes[1, 1].set_title('Settling Time - Normalized (Lower is Better)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add actual values as text on bars
    for i, (bar, val) in enumerate(zip(bars, df_metrics['settling_time'])):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{int(val)} steps',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../images/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: ../images/algorithm_comparison.png")
    
    # ==========================================
    # 5. SAVE COMPARISON RESULTS
    # ==========================================
    
    print("\n[4] Saving comparison results...")
    df_metrics.to_csv('../data/algorithm_comparison.csv', index=False)
    print("Saved: ../data/algorithm_comparison.csv")

print("\n" + "=" * 70)
print("COMPARISON ANALYSIS COMPLETED!")
print("=" * 70)

# ==========================================
# 6. ANALYSIS SUMMARY
# ==========================================

if len(df_metrics) > 0:
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("""
    Algorithm Comparison Summary:
    
    1. Neural Network: Best for accuracy and pattern recognition
    2. Model Predictive Control: Best for optimal control with constraints
    3. Q-Learning: Best for adaptability and model-free learning
    
    Each method has trade-offs between accuracy, computational cost,
    and implementation complexity. Choose based on your specific requirements.
    """)

print("\nFiles generated:")
print("  - ../data/algorithm_comparison.csv (metrics table)")
print("  - ../images/algorithm_comparison.png (visualization)")
