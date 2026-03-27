# F1 Race Data Loader - Your First Script!
# This loads lap-by-lap data from a Formula 1 race

import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# Enable FastF1 cache (speeds up data loading on repeat runs)
# This saves downloaded data so you don't re-download every time
fastf1.Cache.enable_cache('f1_cache')

print("🏎️ Loading F1 race data... This takes 30-60 seconds first time!")

# Load the 2024 Monaco Grand Prix
# Year=2024, Round=8 is Monaco, 'R' means Race (not qualifying)
session = fastf1.get_session(2024, 'Monaco', 'R')

# Download all the data from F1's servers
session.load()

print("✅ Data loaded successfully!")

# Get lap data for one driver - let's pick Max Verstappen
driver = 'VER'  # VER = Verstappen's code
laps = session.laps.pick_driver(driver)

# Extract just the lap numbers and lap times
lap_numbers = laps['LapNumber']
lap_times = laps['LapTime'].dt.total_seconds()  # Convert to seconds

# Create a simple chart
plt.figure(figsize=(12, 6))
plt.plot(lap_numbers, lap_times, color='#1E41FF', linewidth=2, marker='o', markersize=4)
plt.title(f'Max Verstappen - Monaco GP 2024 - Lap Times', fontsize=16, fontweight='bold')
plt.xlabel('Lap Number', fontsize=12)
plt.ylabel('Lap Time (seconds)', fontsize=12)
plt.grid(True, alpha=0.3)

# Save the chart as an image file
plt.savefig('verstappen_lap_times.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as 'verstappen_lap_times.png'")

# Show the chart
plt.show()

print("\n🏁 Done! Check your F1_Project folder for the chart image!")