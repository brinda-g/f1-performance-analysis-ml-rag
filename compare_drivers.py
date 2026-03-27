# F1 Driver Comparison - Compare multiple drivers in the same race
# This shows how different strategies play out

import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

print("🏎️ Loading Monaco GP 2024 data...")

# Load Monaco 2024
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

print("✅ Data loaded! Now extracting driver data...\n")

# Select 3 drivers to compare
# VER = Verstappen, LEC = Leclerc, NOR = Norris
drivers = ['VER', 'LEC', 'NOR']
driver_names = {
    'VER': 'Max Verstappen',
    'LEC': 'Charles Leclerc', 
    'NOR': 'Lando Norris'
}

# Colors for each driver (their team colors!)
driver_colors = {
    'VER': '#1E41FF',  # Red Bull blue
    'LEC': '#DC0000',  # Ferrari red
    'NOR': '#FF8700'   # McLaren orange
}

# Create the comparison chart
plt.figure(figsize=(14, 7))
plt.style.use('dark_background')  # F1-style dark theme!

# Plot each driver
for driver in drivers:
    # Get this driver's laps
    laps = session.laps.pick_driver(driver)
    
    # Filter out lap 0 and any invalid laps (the fix for that weird point!)
    laps = laps[laps['LapNumber'] > 0]
    
    # Get lap numbers and times
    lap_numbers = laps['LapNumber']
    lap_times = laps['LapTime'].dt.total_seconds()
    
    # Plot this driver's data
    plt.plot(
        lap_numbers, 
        lap_times, 
        color=driver_colors[driver],
        linewidth=2.5,
        marker='o',
        markersize=3,
        label=driver_names[driver],
        alpha=0.8
    )
    
    print(f"✅ Plotted {driver_names[driver]} - {len(lap_numbers)} laps")

# Style the chart (F1 broadcast theme!)
plt.title('Monaco GP 2024 - Driver Comparison', 
          fontsize=18, 
          fontweight='bold',
          color='white',
          pad=20)

plt.xlabel('Lap Number', fontsize=14, fontweight='bold', color='#C0C0C0')
plt.ylabel('Lap Time (seconds)', fontsize=14, fontweight='bold', color='#C0C0C0')

# Grid styling
plt.grid(True, alpha=0.2, color='#38383F', linestyle='--')

# Legend styling
plt.legend(
    loc='upper right',
    fontsize=12,
    framealpha=0.9,
    facecolor='#1E1E2E',
    edgecolor='#E10600',
    labelcolor='white'
)

# Add red accent line at top (like F1 graphics!)
plt.axhline(y=plt.ylim()[0], color='#E10600', linewidth=4)

# Set background color
ax = plt.gca()
ax.set_facecolor('#15151E')
plt.gcf().patch.set_facecolor('#1E1E2E')

# Tight layout
plt.tight_layout()

# Save the chart
plt.savefig('driver_comparison.png', dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
print("\n✅ Chart saved as 'driver_comparison.png'")

# Show it
plt.show()

print("\n🏁 Driver comparison complete!")
print("📊 You can now see how different strategies played out!")