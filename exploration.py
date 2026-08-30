import fastf1
import matplotlib.pyplot as plt
#Force graduations
from matplotlib.ticker import MultipleLocator
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

#First filter : Pit laps
clean_laps = driver_laps.loc[
    driver_laps['PitInTime'].isna() & 
    driver_laps['PitOutTime'].isna()
].copy()
clean_laps['LapTime_sec'] = clean_laps['LapTime'].dt.total_seconds()

fig3 = plt.figure()
for stint, data in clean_laps.groupby('Stint'):
    compound = data['Compound'].iloc[0]
    plt.plot(
        data['TyreLife'],
        data['LapTime_sec'],
        markers[compound],
        label = f"Stint {int(stint)} : {compound}"
    )
plt.title('Stints with clean laps')
plt.xlabel('Tyre age')
plt.ylabel('Lap time [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

TabEx3 = clean_laps[[
    'LapNumber', 'LapTime_sec', 'Stint', 'Compound', 'TyreLife', 'TrackStatus', 'IsAccurate', 'Position'
]].copy()

#print(TabEx3.loc[(TabEx3['LapNumber']==39) | (TabEx3['LapNumber']==40)])
#print(TabEx3)
#print(clean_laps['TrackStatus'].value_counts())
#print(TabEx3[['LapNumber','LapTime_sec','Stint','TyreLife','TrackStatus']].loc[TabEx3['TrackStatus']!='1'])

#2nd filter : Green flag & Accurate laps (FastF1 library criteria)
valid_laps_mask = (
    (driver_laps['TrackStatus'] == '1') &
    (driver_laps['IsAccurate'])
)

model_laps = clean_laps.loc[valid_laps_mask].copy()

#print(len(model_laps))
#print(model_laps[['LapNumber']])

fig4 = plt.figure()
for stint, data in model_laps.groupby('Stint'):
    compound = data['Compound'].iloc[0]
    plt.plot(
        data['TyreLife'],
        data['LapTime_sec'],
        markers[compound],
        label = f"Stint {int(stint)} : {compound}"
    )

plt.title('Stints with model laps')
plt.xlabel('Tyre age')
plt.ylabel('Lap time [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)


plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.legend()
plt.show()