import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

print("HFAC Neural Network Training")
print("="*60)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\nLoading dataset...")
df = pd.read_csv('../data/hfac_greenhouse_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# ==========================================
# 2. PREPARE DATA
# ==========================================

print("\nPreparing data...")

# Features (Input sensors)
X = df[['temperature', 'humidity', 'light_intensity', 'motion']].values

# Targets (Output PWM actuators)
y = df[['fan_cooling_pwm', 'fan_circulation_pwm', 'water_pump_pwm', 'grow_light_pwm']].values

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ==========================================
# 3. BUILD NEURAL NETWORK MODEL
# ==========================================

print("\nBuilding Neural Network...")

model = keras.Sequential([
    layers.Input(shape=(4,)),  # 4 input features
    
    # Hidden layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    
    # Output layer: 4 PWM values (0-100%)
    layers.Dense(4, activation='linear')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model.summary())

# ==========================================
# 4. TRAINING
# ==========================================

print("\nTraining model...")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7
)

# Train
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ==========================================
# 5. EVALUATION
# ==========================================

print("\nEvaluating model...")

# Predictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Clip predictions to valid PWM range (0-100)
y_pred = np.clip(y_pred, 0, 100)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy (R² * 100): {r2*100:.2f}%")

# Per-actuator metrics
actuator_names = ['Fan Cooling', 'Fan Circulation', 'Water Pump', 'Grow Light']
print("\nPer-Actuator Performance:")
print("-"*60)
for i, name in enumerate(actuator_names):
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{name:20s} | R²: {r2_i:.4f} | MAE: {mae_i:.4f}")

# ==========================================
# 6. ADVANCED VISUALIZATIONS
# ==========================================

print("\nCreating advanced visualizations...")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training History (Loss)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontsize=11)
ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training History (MAE)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('MAE', fontsize=11)
ax2.set_title('Training & Validation MAE', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: R² Score per Actuator
ax3 = fig.add_subplot(gs[0, 2])
r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(4)]
colors = ['#e74c3c', '#3498db', '#1abc9c', '#f39c12']
bars = ax3.bar(actuator_names, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('R² Score', fontsize=11)
ax3.set_title('R² Score per Actuator', fontsize=12, fontweight='bold')
ax3.set_xticklabels(actuator_names, rotation=15, ha='right', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 1.1)

# Add value labels on bars
for bar, val in zip(bars, r2_scores):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4-7: Prediction vs Actual for each actuator
for i, name in enumerate(actuator_names):
    row = 1 + i // 2
    col = i % 2
    ax = fig.add_subplot(gs[row, col])
    
    # Sample 300 points for clarity
    sample_size = min(300, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    ax.scatter(y_test[indices, i], y_pred[indices, i], alpha=0.5, s=30, color=colors[i])
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R² for this actuator
    r2_act = r2_score(y_test[:, i], y_pred[:, i])
    
    ax.set_xlabel('Actual PWM (%)', fontsize=10)
    ax.set_ylabel('Predicted PWM (%)', fontsize=10)
    ax.set_title(f'{name} (R²={r2_act:.3f})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

# Plot 8: Residual Plot (Error Distribution)
ax8 = fig.add_subplot(gs[2, 0])
residuals = (y_test - y_pred).flatten()
ax8.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax8.set_xlabel('Residual (Actual - Predicted)', fontsize=10)
ax8.set_ylabel('Frequency', fontsize=10)
ax8.set_title('Residual Distribution', fontsize=11, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Error by Actuator (Box Plot)
ax9 = fig.add_subplot(gs[2, 1])
errors_by_actuator = [np.abs(y_test[:, i] - y_pred[:, i]) for i in range(4)]
bp = ax9.boxplot(errors_by_actuator, labels=actuator_names, patch_artist=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax9.set_ylabel('Absolute Error', fontsize=10)
ax9.set_title('Error Distribution by Actuator', fontsize=11, fontweight='bold')
ax9.set_xticklabels(actuator_names, rotation=15, ha='right', fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')

# Plot 10: Performance Summary Table
ax10 = fig.add_subplot(gs[2, 2])
ax10.axis('off')

summary_data = []
for i, name in enumerate(actuator_names):
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    summary_data.append([name, f'{r2_i:.3f}', f'{mae_i:.2f}', f'{rmse_i:.2f}'])

table = ax10.table(cellText=summary_data,
                   colLabels=['Actuator', 'R²', 'MAE', 'RMSE'],
                   cellLoc='center',
                   loc='center',
                   bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, 5):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

ax10.set_title('Performance Summary', fontsize=11, fontweight='bold', pad=20)

plt.suptitle('Neural Network Training Results - Comprehensive Analysis', 
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('../images/training_history.png', dpi=300, bbox_inches='tight')
print("Saved: ../images/training_history.png")

# ==========================================
# 7. PREDICTION VS ACTUAL (Separate Plot)
# ==========================================

plt.figure(figsize=(16, 10))

for i, name in enumerate(actuator_names):
    plt.subplot(2, 2, i+1)
    
    # Sample 200 points for clarity
    sample_size = min(200, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(y_test[indices, i], y_pred[indices, i], alpha=0.5, s=30, color=colors[i])
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual PWM (%)', fontsize=11)
    plt.ylabel('Predicted PWM (%)', fontsize=11)
    plt.title(f'{name} - Prediction vs Actual', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)

plt.tight_layout()
plt.savefig('../images/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("Saved: ../images/prediction_vs_actual.png")

# ==========================================
# 8. SAVE MODEL & SCALERS
# ==========================================

print("\nSaving model and scalers...")

model.save('../models/hfac_model.h5')
joblib.dump(scaler_X, '../models/scaler_X.pkl')
joblib.dump(scaler_y, '../models/scaler_y.pkl')

print("Model saved: ../models/hfac_model.h5")
print("Scaler X saved: ../models/scaler_X.pkl")
print("Scaler y saved: ../models/scaler_y.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  - ../models/hfac_model.h5 (trained model)")
print("  - ../models/scaler_X.pkl (input scaler)")
print("  - ../models/scaler_y.pkl (output scaler)")
print("  - ../images/training_history.png (comprehensive analysis)")
print("  - ../images/prediction_vs_actual.png (prediction plots)")
print("\nNext step: Run comparison_analysis.py or use the GUI!")