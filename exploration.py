import fastf1
import matplotlib.pyplot as plt
#Avoid overinfo
fastf1.set_log_level('WARNING')

#Get 2024 Bahrain Race
session = fastf1.get_session(2024,'Bahrain','R')
session.load()

#Get Verstappen's laps data
laps = session.laps
driver_laps = laps.pick_driver('VER')

#Convert LapTime from timedelta to seconds
driver_laps['LapTime_sec'] = driver_laps['LapTime'].dt.total_seconds()

#Shows 7 important columns
print(driver_laps[['LapNumber','LapTime_sec','Compound','TyreLife','Stint','PitInTime','PitOutTime']])

#Graphic
plt.plot(driver_laps['LapNumber'],driver_laps['LapTime_sec'])
plt.title('Verstappen - Bahrain 2024')
plt.xlabel('Lap number')
plt.ylabel('Lap time [s]')
plt.show()