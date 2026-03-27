# F1 Tire Degradation Heatmap - Visual tire performance analysis
# Shows which tire compound performs best at each lap

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

print("🏎️ Loading Monaco GP 2024 for heatmap analysis...")

# Load Monaco 2024
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

print("✅ Data loaded!\n")

# Get ALL laps from ALL drivers
all_laps = session.laps

# Filter out invalid laps
all_laps = all_laps[all_laps['LapNumber'] > 0]
all_laps = all_laps[all_laps['LapTime'].notna()]  # Remove missing lap times

print(f"📊 Total laps to analyze: {len(all_laps)}")
print(f"   Compounds used: {all_laps['Compound'].unique()}\n")

# Prepare data for heatmap
# We'll create a matrix: rows = compounds, columns = laps
compounds = ['SOFT', 'MEDIUM', 'HARD']
max_lap = int(all_laps['LapNumber'].max())

# Create matrix to store average lap times
# Initialize with NaN (no data)
heatmap_data = np.full((len(compounds), max_lap), np.nan)

# Fill in the data
for idx, compound in enumerate(compounds):
    print(f"Processing {compound} tire data...")
    
    # Get all laps with this compound
    compound_laps = all_laps[all_laps['Compound'] == compound]
    
    # For each lap number, calculate average lap time
    for lap_num in range(1, max_lap + 1):
        lap_data = compound_laps[compound_laps['LapNumber'] == lap_num]
        
        if len(lap_data) > 0:
            # Get lap times in seconds
            times = lap_data['LapTime'].dt.total_seconds()
            # Calculate average (excluding outliers)
            # Filter out times that are way too slow (>120s = pit lap or error)
            valid_times = times[times < 120]
            
            if len(valid_times) > 0:
                avg_time = valid_times.mean()
                heatmap_data[idx, lap_num - 1] = avg_time

print("\n✅ Heatmap data prepared!\n")

# Create the heatmap
fig, ax = plt.subplots(figsize=(16, 6))

# Use a color scheme: darker = faster (lower time), lighter = slower (higher time)
# Reverse the colormap so dark = fast
sns.heatmap(
    heatmap_data,
    cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red = slow, green = fast)
    cbar_kws={'label': 'Average Lap Time (seconds)'},
    yticklabels=compounds,
    xticklabels=range(1, max_lap + 1, 5),  # Show every 5th lap number
    linewidths=0.5,
    linecolor='#15151E',
    ax=ax,
    vmin=75,  # Minimum lap time for color scale
    vmax=95   # Maximum lap time for color scale
)

# Styling
plt.title('Monaco GP 2024 - Tire Degradation Heatmap\nAverage Lap Times by Compound', 
          fontsize=18, 
          fontweight='bold',
          pad=20)

plt.xlabel('Lap Number', fontsize=14, fontweight='bold')
plt.ylabel('Tire Compound', fontsize=14, fontweight='bold')

# Rotate x labels for readability
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Set background
fig.patch.set_facecolor('#1E1E2E')
ax.set_facecolor('#15151E')

# Tight layout
plt.tight_layout()

# Save
plt.savefig('tire_degradation_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
print("✅ Heatmap saved as 'tire_degradation_heatmap.png'\n")

plt.show()

print("🏁 Tire degradation heatmap complete!")
print("📊 This shows you exactly when each tire compound performs best!")
print("\n💡 INSIGHT: Look for the darkest (greenest) cells - those are the fastest laps!")
print("   Red cells = slower laps (tire degradation or traffic)")