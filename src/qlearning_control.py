"""
Q-LEARNING (REINFORCEMENT LEARNING) untuk Greenhouse Control System
====================================================================

Algoritma: Q-Learning (Tabular Reinforcement Learning)
Deskripsi: Agent belajar policy optimal melalui trial-and-error interaction
           dengan environment, tanpa memerlukan model sistem.

Prinsip Kerja:
1. Agent mengamati state environment (temperature, humidity, light)
2. Memilih action (PWM values) berdasarkan Q-table
3. Menerima reward berdasarkan seberapa dekat dengan target
4. Update Q-table menggunakan Bellman equation
5. Ulangi hingga konvergen ke policy optimal

Kelebihan:
- Model-free: tidak perlu tahu dinamika sistem
- Dapat belajar policy optimal dari experience
- Robust terhadap perubahan sistem

Kekurangan:
- Membutuhkan banyak iterasi untuk konvergen
- Curse of dimensionality untuk continuous state space
- Exploration vs exploitation trade-off
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pickle

print("=" * 70)
print("Q-LEARNING (REINFORCEMENT LEARNING) - GREENHOUSE CONTROL SYSTEM")
print("=" * 70)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n[1] Loading dataset...")
df = pd.read_csv('hfac_greenhouse_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} samples")

# ==========================================
# 2. Q-LEARNING PARAMETERS
# ==========================================

print("\n[2] Setting up Q-Learning parameters...")

# Target setpoints
TARGET_TEMP = 25.0
TARGET_HUMIDITY = 65.0
TARGET_LIGHT = 70.0

# Q-Learning hyperparameters
LEARNING_RATE = 0.1      # Alpha: seberapa cepat belajar
DISCOUNT_FACTOR = 0.95   # Gamma: seberapa penting future rewards
EPSILON = 1.0            # Exploration rate (start high)
EPSILON_DECAY = 0.995    # Decay rate per episode
EPSILON_MIN = 0.01       # Minimum exploration

NUM_EPISODES = 1000      # Jumlah episode training
MAX_STEPS = 100          # Max steps per episode

# State discretization (untuk membuat Q-table)
# Kita bagi continuous state space menjadi discrete bins
TEMP_BINS = np.linspace(15, 40, 6)      # 5 bins untuk temperature
HUMIDITY_BINS = np.linspace(30, 90, 6)  # 5 bins untuk humidity
LIGHT_BINS = np.linspace(0, 100, 6)     # 5 bins untuk light

# Action discretization (PWM levels)
# Untuk simplifikasi, kita pakai discrete PWM levels
PWM_LEVELS = [0, 25, 50, 75, 100]

# Jumlah actions = kombinasi dari 4 actuators dengan 5 levels each
# Tapi ini akan terlalu besar (5^4 = 625 actions)
# Kita simplifikasi: setiap actuator punya 3 levels (Low, Medium, High)
ACTION_LEVELS = [0, 50, 100]

print(f"Learning Rate: {LEARNING_RATE}")
print(f"Discount Factor: {DISCOUNT_FACTOR}")
print(f"Episodes: {NUM_EPISODES}")
print(f"State Space: {len(TEMP_BINS)-1} x {len(HUMIDITY_BINS)-1} x {len(LIGHT_BINS)-1} = {(len(TEMP_BINS)-1) * (len(HUMIDITY_BINS)-1) * (len(LIGHT_BINS)-1)} states")
print(f"Action Space: {len(ACTION_LEVELS)}^4 = {len(ACTION_LEVELS)**4} actions")

# ==========================================
# 3. STATE DISCRETIZATION
# ==========================================

def discretize_state(temp, humidity, light):
    """
    Convert continuous state to discrete state index
    
    Args:
        temp: Temperature (continuous)
        humidity: Humidity (continuous)
        light: Light intensity (continuous)
    
    Returns:
        state_index: Tuple (temp_bin, humidity_bin, light_bin)
    """
    temp_bin = np.digitize(temp, TEMP_BINS) - 1
    humidity_bin = np.digitize(humidity, HUMIDITY_BINS) - 1
    light_bin = np.digitize(light, LIGHT_BINS) - 1
    
    # Clip to valid range
    temp_bin = np.clip(temp_bin, 0, len(TEMP_BINS) - 2)
    humidity_bin = np.clip(humidity_bin, 0, len(HUMIDITY_BINS) - 2)
    light_bin = np.clip(light_bin, 0, len(LIGHT_BINS) - 2)
    
    return (temp_bin, humidity_bin, light_bin)

# ==========================================
# 4. ACTION ENCODING/DECODING
# ==========================================

def encode_action(fan_cooling, fan_circulation, water_pump, grow_light):
    """
    Encode 4 PWM values into single action index
    
    Args:
        fan_cooling, fan_circulation, water_pump, grow_light: PWM values (0, 50, or 100)
    
    Returns:
        action_index: Integer index
    """
    # Map PWM to level index (0, 50, 100 -> 0, 1, 2)
    fc_idx = ACTION_LEVELS.index(fan_cooling)
    fcir_idx = ACTION_LEVELS.index(fan_circulation)
    wp_idx = ACTION_LEVELS.index(water_pump)
    gl_idx = ACTION_LEVELS.index(grow_light)
    
    # Encode as single integer
    action_index = fc_idx * (len(ACTION_LEVELS)**3) + fcir_idx * (len(ACTION_LEVELS)**2) + wp_idx * len(ACTION_LEVELS) + gl_idx
    
    return action_index

def decode_action(action_index):
    """
    Decode action index to 4 PWM values
    
    Args:
        action_index: Integer index
    
    Returns:
        (fan_cooling, fan_circulation, water_pump, grow_light): PWM values
    """
    gl_idx = action_index % len(ACTION_LEVELS)
    action_index //= len(ACTION_LEVELS)
    
    wp_idx = action_index % len(ACTION_LEVELS)
    action_index //= len(ACTION_LEVELS)
    
    fcir_idx = action_index % len(ACTION_LEVELS)
    action_index //= len(ACTION_LEVELS)
    
    fc_idx = action_index
    
    return (ACTION_LEVELS[fc_idx], ACTION_LEVELS[fcir_idx], ACTION_LEVELS[wp_idx], ACTION_LEVELS[gl_idx])

# ==========================================
# 5. REWARD FUNCTION
# ==========================================

def calculate_reward(temp, humidity, light):
    """
    Calculate reward based on how close to target
    
    Reward design:
    - Positive reward jika dekat dengan target
    - Negative reward jika jauh dari target
    - Bonus jika semua dalam tolerance
    
    Args:
        temp, humidity, light: Current state values
    
    Returns:
        reward: Scalar reward value
    """
    # Calculate errors
    temp_error = abs(temp - TARGET_TEMP)
    humidity_error = abs(humidity - TARGET_HUMIDITY)
    light_error = abs(light - TARGET_LIGHT)
    
    # Weighted error (temperature lebih penting)
    weighted_error = 2.0 * temp_error + 1.5 * humidity_error + 1.0 * light_error
    
    # Reward = negative error (minimize error = maximize reward)
    reward = -weighted_error
    
    # Bonus jika dalam tolerance
    if temp_error < 1.0 and humidity_error < 3.0 and light_error < 5.0:
        reward += 50  # Big bonus for being in tolerance
    
    return reward

# ==========================================
# 6. ENVIRONMENT SIMULATOR
# ==========================================

class GreenhouseEnv:
    """
    Simplified greenhouse environment untuk Q-Learning
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset environment to random initial state"""
        # Random initial state (realistic ranges)
        self.temp = np.random.uniform(20, 30)
        self.humidity = np.random.uniform(50, 80)
        self.light = np.random.uniform(30, 90)
        
        return self.get_state()
    
    def get_state(self):
        """Get current state"""
        return (self.temp, self.humidity, self.light)
    
    def step(self, action):
        """
        Execute action and return next state, reward, done
        
        Args:
            action: (fan_cooling, fan_circulation, water_pump, grow_light)
        
        Returns:
            next_state, reward, done
        """
        fan_cooling, fan_circulation, water_pump, grow_light = action
        
        # Simulate system dynamics (simplified)
        # Temperature
        cooling_effect = -fan_cooling * 0.05
        self.temp += cooling_effect + np.random.normal(0, 0.2)
        self.temp = np.clip(self.temp, 15, 40)
        
        # Humidity
        humidify_effect = water_pump * 0.08
        dehumidify_effect = -fan_circulation * 0.04
        self.humidity += humidify_effect + dehumidify_effect + np.random.normal(0, 0.5)
        self.humidity = np.clip(self.humidity, 30, 90)
        
        # Light
        light_effect = grow_light * 0.1
        self.light += light_effect + np.random.normal(0, 1.0)
        self.light = np.clip(self.light, 0, 100)
        
        # Calculate reward
        reward = calculate_reward(self.temp, self.humidity, self.light)
        
        # Episode terminates if in tolerance for stability
        temp_error = abs(self.temp - TARGET_TEMP)
        humidity_error = abs(self.humidity - TARGET_HUMIDITY)
        light_error = abs(self.light - TARGET_LIGHT)
        
        done = (temp_error < 0.5 and humidity_error < 2.0 and light_error < 3.0)
        
        return self.get_state(), reward, done

# ==========================================
# 7. Q-TABLE INITIALIZATION
# ==========================================

print("\n[3] Initializing Q-table...")

# Q-table dimensions
n_states = (len(TEMP_BINS) - 1) * (len(HUMIDITY_BINS) - 1) * (len(LIGHT_BINS) - 1)
n_actions = len(ACTION_LEVELS) ** 4

# Initialize Q-table (dictionary untuk sparse representation)
Q_table = {}

def get_Q_value(state, action):
    """Get Q-value from table, return 0 if not exists"""
    return Q_table.get((state, action), 0.0)

def set_Q_value(state, action, value):
    """Set Q-value in table"""
    Q_table[(state, action)] = value

print(f"Q-table initialized (sparse representation)")

# ==========================================
# 8. Q-LEARNING TRAINING
# ==========================================

print("\n[4] Training Q-Learning agent...")

env = GreenhouseEnv()

# Training metrics
episode_rewards = []
episode_lengths = []

epsilon = EPSILON

for episode in range(NUM_EPISODES):
    # Reset environment
    state = env.reset()
    state_discrete = discretize_state(*state)
    
    total_reward = 0
    
    for step in range(MAX_STEPS):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore: random action
            action_index = np.random.randint(0, n_actions)
        else:
            # Exploit: best action from Q-table
            q_values = [get_Q_value(state_discrete, a) for a in range(n_actions)]
            action_index = np.argmax(q_values)
        
        # Decode action
        action = decode_action(action_index)
        
        # Execute action
        next_state, reward, done = env.step(action)
        next_state_discrete = discretize_state(*next_state)
        
        # Q-Learning update (Bellman equation)
        current_Q = get_Q_value(state_discrete, action_index)
        
        # Max Q-value for next state
        next_q_values = [get_Q_value(next_state_discrete, a) for a in range(n_actions)]
        max_next_Q = max(next_q_values)
        
        # Update rule: Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_Q(s',a') - Q(s,a)]
        new_Q = current_Q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_Q - current_Q)
        set_Q_value(state_discrete, action_index, new_Q)
        
        # Update state
        state_discrete = next_state_discrete
        total_reward += reward
        
        if done:
            break
    
    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # Store metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(step + 1)
    
    # Progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        print(f"  Episode {episode + 1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Epsilon: {epsilon:.3f}")

print(f"\nTraining completed! Q-table size: {len(Q_table)} entries")

# ==========================================
# 9. TESTING Q-LEARNING AGENT
# ==========================================

print("\n[5] Testing Q-Learning agent...")

# Test on multiple episodes
test_episodes = 50
test_results = {
    'temperature': [],
    'humidity': [],
    'light_intensity': [],
    'fan_cooling_pwm': [],
    'fan_circulation_pwm': [],
    'water_pump_pwm': [],
    'grow_light_pwm': []
}

for ep in range(test_episodes):
    state = env.reset()
    
    for step in range(MAX_STEPS):
        state_discrete = discretize_state(*state)
        
        # Use greedy policy (no exploration)
        q_values = [get_Q_value(state_discrete, a) for a in range(n_actions)]
        action_index = np.argmax(q_values)
        action = decode_action(action_index)
        
        # Store results
        test_results['temperature'].append(state[0])
        test_results['humidity'].append(state[1])
        test_results['light_intensity'].append(state[2])
        test_results['fan_cooling_pwm'].append(action[0])
        test_results['fan_circulation_pwm'].append(action[1])
        test_results['water_pump_pwm'].append(action[2])
        test_results['grow_light_pwm'].append(action[3])
        
        # Next state
        state, reward, done = env.step(action)
        
        if done:
            break

df_results = pd.DataFrame(test_results)

# ==========================================
# 10. EVALUATION
# ==========================================

print("\n[6] Evaluating Q-Learning performance...")

# Calculate errors
temp_error = np.abs(df_results['temperature'] - TARGET_TEMP)
humidity_error = np.abs(df_results['humidity'] - TARGET_HUMIDITY)
light_error = np.abs(df_results['light_intensity'] - TARGET_LIGHT)

mae_temp = np.mean(temp_error)
mae_humidity = np.mean(humidity_error)
mae_light = np.mean(light_error)

rmse_temp = np.sqrt(np.mean(temp_error ** 2))
rmse_humidity = np.sqrt(np.mean(humidity_error ** 2))
rmse_light = np.sqrt(np.mean(light_error ** 2))

print("\n" + "=" * 70)
print("Q-LEARNING PERFORMANCE METRICS")
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

# ==========================================
# 11. VISUALIZATION
# ==========================================

print("\n[7] Creating visualizations...")

# Plot 1: Training Progress
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Training rewards
axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
# Moving average
window = 50
moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
axes[0, 0].plot(moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg', color='red')
axes[0, 0].set_xlabel('Episode', fontsize=11)
axes[0, 0].set_ylabel('Total Reward', fontsize=11)
axes[0, 0].set_title('Q-Learning Training Progress', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Temperature control
axes[0, 1].plot(df_results['temperature'], label='Actual', linewidth=2)
axes[0, 1].axhline(y=TARGET_TEMP, color='r', linestyle='--', label='Target', linewidth=2)
axes[0, 1].fill_between(range(len(df_results)), TARGET_TEMP - 1, TARGET_TEMP + 1, 
                         alpha=0.2, color='green', label='±1°C tolerance')
axes[0, 1].set_xlabel('Time Step', fontsize=11)
axes[0, 1].set_ylabel('Temperature (°C)', fontsize=11)
axes[0, 1].set_title('Q-Learning Temperature Control', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Humidity control
axes[1, 0].plot(df_results['humidity'], label='Actual', linewidth=2, color='blue')
axes[1, 0].axhline(y=TARGET_HUMIDITY, color='r', linestyle='--', label='Target', linewidth=2)
axes[1, 0].fill_between(range(len(df_results)), TARGET_HUMIDITY - 3, TARGET_HUMIDITY + 3, 
                         alpha=0.2, color='green', label='±3% tolerance')
axes[1, 0].set_xlabel('Time Step', fontsize=11)
axes[1, 0].set_ylabel('Humidity (%)', fontsize=11)
axes[1, 0].set_title('Q-Learning Humidity Control', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Light control
axes[1, 1].plot(df_results['light_intensity'], label='Actual', linewidth=2, color='orange')
axes[1, 1].axhline(y=TARGET_LIGHT, color='r', linestyle='--', label='Target', linewidth=2)
axes[1, 1].fill_between(range(len(df_results)), TARGET_LIGHT - 5, TARGET_LIGHT + 5, 
                         alpha=0.2, color='green', label='±5% tolerance')
axes[1, 1].set_xlabel('Time Step', fontsize=11)
axes[1, 1].set_ylabel('Light Intensity (%)', fontsize=11)
axes[1, 1].set_title('Q-Learning Light Control', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qlearning_control_results.png', dpi=300, bbox_inches='tight')
print("Saved: qlearning_control_results.png")

# ==========================================
# 12. SAVE RESULTS
# ==========================================

print("\n[8] Saving results...")

# Save Q-table
with open('qlearning_qtable.pkl', 'wb') as f:
    pickle.dump(Q_table, f)
print("Saved: qlearning_qtable.pkl")

# Save results
df_results.to_csv('qlearning_control_results.csv', index=False)
print("Saved: qlearning_control_results.csv")

# Save training history
training_history = pd.DataFrame({
    'episode': range(NUM_EPISODES),
    'reward': episode_rewards,
    'length': episode_lengths
})
training_history.to_csv('qlearning_training_history.csv', index=False)
print("Saved: qlearning_training_history.csv")

print("\n" + "=" * 70)
print("Q-LEARNING TRAINING & TESTING COMPLETED!")
print("=" * 70)
print("\nFiles generated:")
print("  - qlearning_qtable.pkl (trained Q-table)")
print("  - qlearning_control_results.csv (test results)")
print("  - qlearning_training_history.csv (training metrics)")
print("  - qlearning_control_results.png (visualization)")
