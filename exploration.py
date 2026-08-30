import fastf1
import numpy as np
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

#Add Lap time delta column in model_laps (difference between two consecutive laps of the same stint)
model_laps['LapTimeDelta'] = model_laps.groupby('Stint')['LapTime_sec'].diff()
#print(model_laps[['LapNumber','TyreLife','LapTime_sec','LapTimeDelta']].loc[model_laps['Stint'] == 3])

#Correcting so that Lap time delta is calculated only if the laps are consecutive
model_laps['LapNumberDelta'] = model_laps.groupby('Stint')['LapNumber'].diff()
model_laps['LapTimeDelta'] = model_laps['LapTimeDelta'].where(model_laps['LapNumberDelta']==1)
#print(model_laps[['LapNumber','LapNumberDelta','TyreLife','LapTime_sec','LapTimeDelta']])

#Sorting lap time deltas to flag possible outliers
model_laps['AbsLapTimeDelta'] = model_laps['LapTimeDelta'].abs()
#print(model_laps[['LapNumber','Stint','TyreLife','LapTime_sec','LapTimeDelta','AbsLapTimeDelta']].sort_values(by=['AbsLapTimeDelta'], ascending=False).head(10))

model_laps['Sector1Time_sec'] = model_laps['Sector1Time'].dt.total_seconds()
model_laps['Sector2Time_sec'] = model_laps['Sector2Time'].dt.total_seconds()
model_laps['Sector3Time_sec'] = model_laps['Sector3Time'].dt.total_seconds()

#print(model_laps[['LapNumber','TyreLife','LapTime_sec','Sector1Time_sec','Sector2Time_sec','Sector3Time_sec']].loc[(model_laps['LapNumber']>=39) & (model_laps['LapNumber']<=43)])

#Isolation of stint 2 for first linear model
hard_stint = model_laps.loc[model_laps['Stint']==2].copy()
#print(hard_stint[['LapNumber','TyreLife','LapTime_sec']])

x = hard_stint['TyreLife']
y = hard_stint['LapTime_sec']

#Coefficients of the linear regression
slope, intercept = np.polyfit(x,y,1)
#print(round(intercept,3),round(slope,3))

hard_stint['PredictedLapTime'] = intercept+slope*hard_stint['TyreLife']

fig5 = plt.figure()
compound = hard_stint['Compound'].iloc[0]
plt.plot(x,y,markers[compound],label='Lap time')
plt.plot(x,hard_stint['PredictedLapTime'],label='Predicted time')

plt.title('Linear regression of 2nd stint (hard)')
plt.xlabel('Tyre age')
plt.ylabel('Lap time [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

#Introduction of residual
hard_stint['Residual'] = hard_stint['LapTime_sec'] - hard_stint['PredictedLapTime']

fig6 = plt.figure()
plt.plot(hard_stint['TyreLife'],hard_stint['Residual'],'+')
plt.axhline(0,color='r')

plt.title('Residual of linear regression 2nd stint')
plt.xlabel('Tyre age [laps]')
plt.ylabel('Residual [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

mean_residual = hard_stint['Residual'].mean()
RMSE = np.sqrt(1/len(hard_stint['Residual']) * (hard_stint['Residual']**2).sum())
max_abs_residual = hard_stint['Residual'].abs().max()
print(
    f'Mean residual = {round(mean_residual,3)}s \nRMSE = {round(RMSE,3)}s \nMaximum absolute residual = {round(max_abs_residual,3)}s'
)

SSE = (hard_stint['Residual']**2).sum()
SST = ((hard_stint['LapTime_sec']-hard_stint['LapTime_sec'].mean())**2).sum()
R2 = 1 - SSE/SST
print(f'SSE = {round(SSE,3)}\nSST = {round(SST,3)}\nR2 = {round(R2,3)}')

plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)

#plt.legend()
#plt.show()