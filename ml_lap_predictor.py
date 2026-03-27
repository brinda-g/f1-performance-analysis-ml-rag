# F1 Lap Time Predictor - Machine Learning Model
# Predicts lap times based on tire compound, tire age, and track position

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

print("🏎️ Loading Monaco GP 2024 for ML training...")

# Load Monaco 2024
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

print("✅ Data loaded! Now preparing ML training data...\n")

# Get all laps
laps = session.laps

# Filter valid laps
laps = laps[laps['LapNumber'] > 0]
laps = laps[laps['LapTime'].notna()]
laps = laps[laps['Compound'].notna()]

# Remove extremely slow laps (pit laps, safety car, etc.)
lap_times_seconds = laps['LapTime'].dt.total_seconds()
laps = laps[lap_times_seconds < 120]  # Remove laps slower than 2 minutes

print(f"📊 Training dataset: {len(laps)} laps\n")

# ============ FEATURE ENGINEERING ============

# 1. Encode tire compound as numbers
compound_mapping = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
laps = laps.copy()  # Create explicit copy to avoid warnings
laps['CompoundEncoded'] = laps['Compound'].map(compound_mapping)

# 2. Calculate tire age (laps on current tire set)
laps['TireAge'] = laps['TyreLife']

# 3. Get track position
laps['Position'] = laps['Position']

# 4. Get lap number
laps['LapNum'] = laps['LapNumber']

# 5. Target variable: lap time in seconds
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

# Create feature matrix (X) and target (y)
features = ['CompoundEncoded', 'TireAge', 'Position', 'LapNum']
X = laps[features].copy()
y = laps['LapTimeSeconds'].copy()

# Handle missing values
X = X.fillna(X.mean())

print("🔧 Features prepared:")
print(f"   - Tire compound (0=Soft, 1=Medium, 2=Hard)")
print(f"   - Tire age (laps on current tires)")
print(f"   - Track position (1st, 2nd, 3rd...)")
print(f"   - Lap number\n")

# ============ TRAIN ML MODEL ============

print("🤖 Training Random Forest model...")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Max tree depth
    min_samples_split=10,  # Min samples to split
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

model.fit(X_train, y_train)

print("✅ Model trained!\n")

# ============ EVALUATE MODEL ============

print("📈 Evaluating model performance...\n")

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   Mean Absolute Error: {mae:.2f} seconds")
print(f"   R² Score: {r2:.3f} ({r2*100:.1f}% accuracy)")
print(f"\n   💡 The model predicts lap times within ±{mae:.2f} seconds on average!\n")

# ============ SAVE MODEL FIRST (BEFORE PLOTTING) ============

print("💾 Saving trained model...")

with open('lap_time_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'lap_time_model.pkl'\n")

# ============ VISUALIZE PREDICTIONS ============

print("📊 Creating prediction visualization...")

plt.figure(figsize=(12, 6))
plt.style.use('dark_background')

# Scatter plot: actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.5, color='#00D2BE', s=50, edgecolors='white', linewidths=0.5)

# Perfect prediction line (if predictions were perfect)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

# Styling
plt.title('F1 Lap Time Prediction - Model Performance', 
          fontsize=18, fontweight='bold', color='white', pad=20)
plt.xlabel('Actual Lap Time (seconds)', fontsize=14, fontweight='bold', color='#C0C0C0')
plt.ylabel('Predicted Lap Time (seconds)', fontsize=14, fontweight='bold', color='#C0C0C0')

plt.grid(True, alpha=0.2, color='#38383F', linestyle='--')

# Add performance text
textstr = f'MAE: {mae:.2f}s\nR²: {r2:.3f}'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#1E1E2E', edgecolor='#E10600', linewidth=2))

plt.legend(loc='lower right', fontsize=11, framealpha=0.9, 
           facecolor='#1E1E2E', edgecolor='#E10600')

# Background
ax = plt.gca()
ax.set_facecolor('#15151E')
plt.gcf().patch.set_facecolor('#1E1E2E')

plt.tight_layout()
plt.savefig('ml_model_performance.png', dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
print("✅ Visualization saved as 'ml_model_performance.png'\n")

# Don't show plot - just save it
# plt.show()  # Commented out so it doesn't block
plt.close()  # Close the plot instead

# ============ TEST PREDICTION ============

print("🔮 Testing prediction with example inputs:\n")

# Example 1: Fresh soft tires, 1st position, lap 10
example1 = pd.DataFrame({
    'CompoundEncoded': [0],  # Soft
    'TireAge': [3],          # 3 laps old
    'Position': [1],         # 1st place
    'LapNum': [10]           # Lap 10
})

pred1 = model.predict(example1)[0]
print(f"   Scenario 1: Soft tires (3 laps old), P1, Lap 10")
print(f"   Predicted lap time: {pred1:.2f} seconds\n")

# Example 2: Old medium tires, 5th position, lap 40
example2 = pd.DataFrame({
    'CompoundEncoded': [1],  # Medium
    'TireAge': [25],         # 25 laps old
    'Position': [5],         # 5th place
    'LapNum': [40]           # Lap 40
})

pred2 = model.predict(example2)[0]
print(f"   Scenario 2: Medium tires (25 laps old), P5, Lap 40")
print(f"   Predicted lap time: {pred2:.2f} seconds\n")

print("🏁 ML model training complete!")
print("🤖 You now have a working lap time predictor!")
print("📊 Model accuracy: {:.1f}%".format(r2 * 100))