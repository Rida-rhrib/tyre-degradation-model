import fastf1
import matplotlib.pyplot as plt
#Avoid overinfo
fastf1.set_log_level('WARNING')

#Get 2024 Bahrain Race
session = fastf1.get_session(2024,'Bahrain','R')
session.load()

#Get Verstappen's laps data
laps = session.laps
driver_laps = laps.pick_drivers('VER').copy()

#Convert LapTime from timedelta to seconds
driver_laps['LapTime_sec'] = driver_laps['LapTime'].dt.total_seconds()

#Shows 7 important columns
TabEx1 = driver_laps[['LapNumber','LapTime_sec','Compound','TyreLife','Stint','PitInTime','PitOutTime']]
#print(TabEx1)

#Plot 1 : x=Lap Number; y=Lap Time
fig1 = plt.figure()
plt.plot(driver_laps['LapNumber'],driver_laps['LapTime_sec'])
plt.title('Verstappen - Bahrain 2024')
plt.xlabel('Lap number')
plt.ylabel('Lap time [s]')

#Stint isolation
TabEx2 = driver_laps.groupby('Stint').agg(
    Compound = ('Compound','first'),
    FirstLap = ('LapNumber','min'),
    LastLap = ('LapNumber','max'),
    MinTyreLife = ('TyreLife','min'),
    MaxTyreLife = ('TyreLife','max'),
    NumberOfLaps = ('LapNumber','count')
    )
TabEx2 = TabEx2.reset_index()

#Plot 2 : x=Tyre Age; y=Lap Time for each stint
markers = {
    'SOFT':'o',
    'MEDIUM':'.',
    'HARD':'+'
}

fig2 = plt.figure()

for stint, data in TabEx1.groupby('Stint'):  
    compound = data['Compound'].iloc[0]
    plt.plot(
        data['TyreLife'], 
        data['LapTime_sec'], 
        markers[compound], 
        label=f"Stint {int(stint)} : {compound}"
        )

plt.title('Stints')
plt.xlabel('Tyre age')
plt.ylabel('Lap time [s]')

plt.close(fig1)
plt.legend()
plt.show()