# F1 Tire Strategy Analysis - Show lap times with tire compound data
# This reveals strategic decisions during the race

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

print("🏎️ Loading Monaco GP 2024 with tire data...")

# Load Monaco 2024
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

print("✅ Data loaded!\n")

# Driver to analyze in detail
driver = 'VER'
driver_name = 'Max Verstappen'

# Get laps
laps = session.laps.pick_driver(driver)
laps = laps[laps['LapNumber'] > 0]  # Filter out lap 0

# Extract data
lap_numbers = laps['LapNumber'].values
lap_times = laps['LapTime'].dt.total_seconds().values
compounds = laps['Compound'].values  # Tire compound: SOFT, MEDIUM, HARD

print(f"📊 Analyzing {driver_name}:")
print(f"   Total laps: {len(lap_numbers)}")
print(f"   Tire compounds used: {set(compounds)}\n")

# Tire colors (F1 official colors!)
compound_colors = {
    'SOFT': '#DC0000',      # Red
    'MEDIUM': '#FFD700',    # Yellow
    'HARD': '#FFFFFF'       # White
}

# Create the chart
plt.figure(figsize=(14, 8))
plt.style.use('dark_background')

# Plot lap times with tire compound colors
for compound in set(compounds):
    # Get laps with this compound
    mask = compounds == compound
    
    plt.scatter(
        lap_numbers[mask],
        lap_times[mask],
        c=compound_colors.get(compound, '#C0C0C0'),
        label=f'{compound} tire',
        s=80,  # Size of markers
        alpha=0.9,
        edgecolors='white',
        linewidths=0.5,
        zorder=3  # Draw on top
    )

# Connect points with a line
plt.plot(lap_numbers, lap_times, 
         color='#1E41FF', 
         linewidth=1.5, 
         alpha=0.4,
         zorder=2)

# Find pit stops (where compound changes)
pit_laps = []
for i in range(1, len(compounds)):
    if compounds[i] != compounds[i-1]:
        pit_laps.append(lap_numbers[i])
        
print(f"🔧 Pit stops detected at laps: {pit_laps}")

# Mark pit stops with vertical lines
for pit_lap in pit_laps:
    plt.axvline(
        x=pit_lap, 
        color='#00D2BE', 
        linestyle='--', 
        linewidth=2,
        alpha=0.7,
        label='Pit stop' if pit_lap == pit_laps[0] else None
    )
    
    # Add pit stop label
    plt.text(
        pit_lap, 
        plt.ylim()[1] * 0.95, 
        f'PIT\nLap {pit_lap}',
        color='#00D2BE',
        fontsize=10,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E1E2E', edgecolor='#00D2BE', linewidth=2)
    )

# Styling
plt.title(f'{driver_name} - Monaco GP 2024 - Tire Strategy Analysis', 
          fontsize=18, 
          fontweight='bold',
          color='white',
          pad=20)

plt.xlabel('Lap Number', fontsize=14, fontweight='bold', color='#C0C0C0')
plt.ylabel('Lap Time (seconds)', fontsize=14, fontweight='bold', color='#C0C0C0')

plt.grid(True, alpha=0.2, color='#38383F', linestyle='--', zorder=1)

# Legend
plt.legend(
    loc='upper left',
    fontsize=11,
    framealpha=0.95,
    facecolor='#1E1E2E',
    edgecolor='#E10600',
    labelcolor='white'
)

# Red accent line
plt.axhline(y=plt.ylim()[0], color='#E10600', linewidth=4)

# Background
ax = plt.gca()
ax.set_facecolor('#15151E')
plt.gcf().patch.set_facecolor('#1E1E2E')

plt.tight_layout()

# Save
plt.savefig('tire_strategy_analysis.png', dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
print("\n✅ Chart saved as 'tire_strategy_analysis.png'")

plt.show()

print("\n🏁 Tire strategy analysis complete!")
print("📊 You can now see exactly when pit stops happened and which tires were used!")