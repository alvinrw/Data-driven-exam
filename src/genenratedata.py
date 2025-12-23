import numpy as np
import pandas as pd
import math

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples: 7 days simulation, 1 minute intervals
# 7 days * 24 hours * 60 minutes = 10080 data points
n_samples = 10080
time_steps = np.arange(n_samples)

print("Generating HFAC Realistic Synthetic Dataset...")

# ==========================================
# GENERATE SENSOR DATA (PHYSICS BASED)
# ==========================================

# Time simulation
# 1 day = 1440 minutes. Use sine function for daily cycle.
day_cycle = 1440
time_factor = (2 * np.pi * time_steps) / day_cycle

# Light intensity (0-100%)
# Sunrise at 6AM, sunset at 6PM. Peak at noon.
# Logic: Maximum at noon, minimum at night.
light_base = -np.cos(time_factor)
light_intensity = (light_base + 0.2) * 60
light_intensity = np.clip(light_intensity, 0, 100)
# Add cloud cover variation
cloud_cover = np.random.uniform(0.8, 1.0, n_samples)
light_intensity = light_intensity * cloud_cover

# Temperature (15-40 degrees C)
# Temperature follows light but with lag (delayed heating/cooling)
# Shift phase slightly from light
temp_base = -np.cos(time_factor - 0.5) 
temperature = 25 + (temp_base * 7)  # Mean 25, variation +/- 7 (18-32)
# Add random noise
temperature += np.random.normal(0, 0.5, n_samples)
temperature = np.clip(temperature, 15, 40)

# Humidity (30-90% RH)
# Humidity inversely proportional to temperature (temp up, RH down)
humidity = 85 - (temp_base * 20)  # Mean 85, drops to 65 when hot
# Add random noise
humidity += np.random.normal(0, 2, n_samples)
humidity = np.clip(humidity, 30, 90)

# Motion detected (0 or 1)
# More activity during daytime (8AM - 5PM)
# Use probability based on hour of day
hour_of_day = (time_steps % 1440) / 60
motion_prob = np.where((hour_of_day > 7) & (hour_of_day < 18), 0.6, 0.1)
motion = np.random.binomial(1, motion_prob)

# ==========================================
# GENERATE ACTUATOR PWM (0-100%)
# Berdasarkan logika greenhouse control
# ==========================================

# Inisialisasi arrays
fan_cooling_pwm = np.zeros(n_samples)
fan_circulation_pwm = np.zeros(n_samples)
water_pump_pwm = np.zeros(n_samples)
grow_light_pwm = np.zeros(n_samples)

# Target ideal greenhouse conditions:
TARGET_TEMP = 25  # degrees C
TARGET_HUMIDITY = 65  # %
TARGET_LIGHT = 70  # %

for i in range(n_samples):
    temp = temperature[i]
    hum = humidity[i]
    light = light_intensity[i]
    has_motion = motion[i]
    
    # Fan cooling PWM
    # Activates when temp > target, higher temp = higher PWM
    if temp > TARGET_TEMP:
        temp_diff = temp - TARGET_TEMP
        fan_cooling_pwm[i] = min(100, (temp_diff / 10) * 100)
    else:
        fan_cooling_pwm[i] = 0
    
    # Fan circulation PWM
    # Always runs at base level for circulation, higher with motion
    base_circulation = 20  # Base 20% for minimal circulation
    if has_motion:
        fan_circulation_pwm[i] = min(100, base_circulation + 30)
    else:
        fan_circulation_pwm[i] = base_circulation
    
    # Increase circulation if humidity is high
    if hum > 75:
        fan_circulation_pwm[i] = min(100, fan_circulation_pwm[i] + 20)
    
    # Water pump PWM
    # Activates when humidity < target
    if hum < TARGET_HUMIDITY:
        hum_diff = TARGET_HUMIDITY - hum
        water_pump_pwm[i] = min(100, (hum_diff / 20) * 100)
    else:
        water_pump_pwm[i] = 0
    
    # Grow light PWM
    # Activates when light intensity < target
    if light < TARGET_LIGHT:
        light_diff = TARGET_LIGHT - light
        grow_light_pwm[i] = min(100, (light_diff / 70) * 100)
    else:
        grow_light_pwm[i] = 0

# Add noise and realistic variation
# Add small noise to PWM (+/- 2%) to simulate real conditions
fan_cooling_pwm += np.random.uniform(-2, 2, n_samples)
fan_circulation_pwm += np.random.uniform(-2, 2, n_samples)
water_pump_pwm += np.random.uniform(-2, 2, n_samples)
grow_light_pwm += np.random.uniform(-2, 2, n_samples)

# Clip to range 0-100
fan_cooling_pwm = np.clip(fan_cooling_pwm, 0, 100)
fan_circulation_pwm = np.clip(fan_circulation_pwm, 0, 100)
water_pump_pwm = np.clip(water_pump_pwm, 0, 100)
grow_light_pwm = np.clip(grow_light_pwm, 0, 100)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'light_intensity': light_intensity,
    'motion': motion,
    'fan_cooling_pwm': fan_cooling_pwm,
    'fan_circulation_pwm': fan_circulation_pwm,
    'water_pump_pwm': water_pump_pwm,
    'grow_light_pwm': grow_light_pwm
})

# Shuffle dataset for training (important to avoid bias)
# Shuffling ensures training is not biased by time sequence
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# SAVE TO CSV
# ==========================================

csv_filename = 'hfac_greenhouse_dataset.csv'
df_shuffled.to_csv(csv_filename, index=False)

print(f"Dataset generated successfully!")
print(f"Total samples: {n_samples} (Simulated 7 Days)")
print(f"Saved to: {csv_filename}")
print("\nDataset Statistics:")
print("="*60)
print(df.describe())
print("\nSample data (first 5 rows):")
print("="*60)
print(df.head())
print("\nDataset ready for training!")