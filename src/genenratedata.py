import numpy as np
import pandas as pd
import math

# Set random seed untuk reproducibility
np.random.seed(42)

# Jumlah data (Simulasi 7 hari, data per 1 menit)
# 7 hari * 24 jam * 60 menit = 10080 data points
n_samples = 10080
time_steps = np.arange(n_samples)

print("Generating HFAC Realistic Synthetic Dataset...")

# ==========================================
# GENERATE SENSOR DATA (PHYSICS BASED)
# ==========================================

# 1. TIME SIMULATION
# 1 hari = 1440 menit. Kita pakai fungsi Sinus untuk siklus harian.
day_cycle = 1440
time_factor = (2 * np.pi * time_steps) / day_cycle

# 2. LIGHT INTENSITY (0-100%)
# Matahari terbit jam 6, terbenam jam 18. Puncak jam 12.
# Sinus -1 sampai 1. Kita geser biar pas siang positif.
# Logic: Max di siang, 0 di malam.
light_base = -np.cos(time_factor) # Mulai dari malam (rendah), naik ke siang
light_intensity = (light_base + 0.2) * 60 # Scale up
light_intensity = np.clip(light_intensity, 0, 100) # Clip 0-100
# Tambah variasi awan (random noise reduction)
cloud_cover = np.random.uniform(0.8, 1.0, n_samples)
light_intensity = light_intensity * cloud_cover

# 3. TEMPERATURE (15-40°C)
# Suhu mengikuti cahaya tapi ada lag (terlambat panas, terlambat dingin)
# Kita geser phase sedikit dari cahaya
temp_base = -np.cos(time_factor - 0.5) 
temperature = 25 + (temp_base * 7) # Mean 25, variasi +/- 7 (18-32)
# Tambah noise random
temperature += np.random.normal(0, 0.5, n_samples)
temperature = np.clip(temperature, 15, 40)

# 4. HUMIDITY (30-90% RH)
# Kelembaban berbanding terbalik dengan suhu (Suhu naik, RH turun)
humidity = 85 - (temp_base * 20) # Mean 85, turun sampai 65 saat panas
# Tambah noise random
humidity += np.random.normal(0, 2, n_samples)
humidity = np.clip(humidity, 30, 90)

# 5. MOTION DETECTED (0 atau 1)
# Lebih banyak aktivitas di siang hari (jam 08.00 - 17.00)
# Kita pakai probabilitas berdasarkan jam
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

# Target ideal greenhouse:
TARGET_TEMP = 25  # °C
TARGET_HUMIDITY = 65  # %
TARGET_LIGHT = 70  # %

for i in range(n_samples):
    temp = temperature[i]
    hum = humidity[i]
    light = light_intensity[i]
    has_motion = motion[i]
    
    # ========== FAN COOLING PWM ==========
    # Nyala kalau suhu > target, makin panas makin kencang
    if temp > TARGET_TEMP:
        temp_diff = temp - TARGET_TEMP
        fan_cooling_pwm[i] = min(100, (temp_diff / 10) * 100)  # Lebih sensitif
    else:
        fan_cooling_pwm[i] = 0
    
    # ========== FAN CIRCULATION PWM ==========
    # Selalu jalan sedikit untuk sirkulasi, lebih kencang kalau ada motion
    base_circulation = 20  # Base 20% untuk sirkulasi minimal
    if has_motion:
        fan_circulation_pwm[i] = min(100, base_circulation + 30)
    else:
        fan_circulation_pwm[i] = base_circulation
    
    # Tambah sirkulasi kalau humidity tinggi
    if hum > 75:
        fan_circulation_pwm[i] = min(100, fan_circulation_pwm[i] + 20)
    
    # ========== WATER PUMP PWM ==========
    # Nyala kalau humidity < target
    if hum < TARGET_HUMIDITY:
        hum_diff = TARGET_HUMIDITY - hum
        water_pump_pwm[i] = min(100, (hum_diff / 20) * 100)
    else:
        water_pump_pwm[i] = 0
    
    # ========== GROW LIGHT PWM ==========
    # Nyala kalau intensitas cahaya < target
    if light < TARGET_LIGHT:
        light_diff = TARGET_LIGHT - light
        grow_light_pwm[i] = min(100, (light_diff / 70) * 100)
    else:
        grow_light_pwm[i] = 0

# ==========================================
# TAMBAH NOISE & VARIASI REALISTIS
# ==========================================

# Tambah noise kecil ke PWM (±2%) untuk simulasi kondisi real
fan_cooling_pwm += np.random.uniform(-2, 2, n_samples)
fan_circulation_pwm += np.random.uniform(-2, 2, n_samples)
water_pump_pwm += np.random.uniform(-2, 2, n_samples)
grow_light_pwm += np.random.uniform(-2, 2, n_samples)

# Clip ke range 0-100
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

# Shuffle dataset (PENTING untuk training, tapi kalau mau lihat time series jangan di shuffle dulu)
# Kita shuffle agar training tidak bias urutan
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