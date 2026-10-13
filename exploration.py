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
#print(f'Mean residual = {round(mean_residual,3)}s \nRMSE = {round(RMSE,3)}s \nMaximum absolute residual = {round(max_abs_residual,3)}s')

SSE = (hard_stint['Residual']**2).sum()
SST = ((hard_stint['LapTime_sec']-hard_stint['LapTime_sec'].mean())**2).sum()
R2 = 1 - SSE/SST
#print(f'SSE = {round(SSE,3)}\nSST = {round(SST,3)}\nR2 = {round(R2,3)}')

#Comparing lap times with same Tyre age & compound to isolate track's and fuel's (+e) evolution
TabEx5 = model_laps.loc[model_laps['Stint']!=2].pivot(columns='Stint',index='TyreLife',values='LapTime_sec')
TabEx5 = TabEx5.rename(columns={1:'LapTime_stint1',3:'LapTime_stint3'})
TabEx5 = TabEx5.reset_index()
TabEx5.columns.name = None
TabEx5['DeltaTime_stints'] = TabEx5['LapTime_stint3']-TabEx5['LapTime_stint1']
TabEx5 = TabEx5.loc[TabEx5['DeltaTime_stints'].notna()]
#print(TabEx5[['TyreLife','LapTime_stint1','LapTime_stint3','DeltaTime_stints']])

#print(f'Delta time between stint 3 and 1 data : \nMean = {round(TabEx5['DeltaTime_stints'].mean(),3)}s \nStandard deviation = {round(TabEx5['DeltaTime_stints'].std(),3)}s \nMin = {round(TabEx5['DeltaTime_stints'].min(),3)}s and Max = {round(TabEx5['DeltaTime_stints'].max(),3)}s')
#print(f'Mean gain per lap = {round(TabEx5['DeltaTime_stints'].mean()/40,3)}s')

#First model of fuel effect
#bf (fuel effect s/lap) = - qf (fuel consuption kg/lap) * kf (mass effect s/kg)
#Hypothesis : 20% precision on qf & kf
qf = 1.8
kf = 0.020
#Extreme cases
#print(f'qf_low kf_low : bfuel = {round(-qf*kf*0.8**2, 3)} s/lap; bremaining = {round(0.061-(-qf*kf*0.8**2), 3)} s/lap \nqf_low kf_high (or vice versa) : bfuel = {round(-qf*kf*0.8*1.2, 3)} s/lap; bremaining = {round(0.061-(-qf*kf*0.8*1.2), 3)} s/lap \nqf_high kf_hig : bfuel = {round(-qf*kf*1.2**2, 3)} s/lap; bremaining = {round(0.061-(-qf*kf*1.2**2), 3)} s/lap')

#After bibliography :
qf = 1.775
qfmin = 1.7
qfmax = 1.85

kf = 0.031
kfmin = 0.025
kfmax = 0.037

FuelEffectPerLap = qf*kf

hard_stint['FuelCorrectedLapTime'] = hard_stint['LapTime_sec'] + (hard_stint['LapNumber'] - hard_stint['LapNumber'].min())*FuelEffectPerLap
#print(hard_stint[['LapNumber', 'TyreLife', 'LapTime_sec', 'FuelCorrectedLapTime']])

bf, af = np.polyfit(x, hard_stint['FuelCorrectedLapTime'], 1)
#print(round(af,3),round(bf,3))

hard_stint['PredictedFuelCorrectedLapTime'] = af + bf * hard_stint['TyreLife']
hard_stint['FuelCorrectedResidual'] = hard_stint['FuelCorrectedLapTime'] - hard_stint['PredictedFuelCorrectedLapTime']

FuelCorrectedSSE = (hard_stint['FuelCorrectedResidual']**2).sum()
FuelCorrectedRMSE = np.sqrt(1/len(hard_stint['FuelCorrectedResidual'])*FuelCorrectedSSE)
FuelCorrectedSST = ((hard_stint['FuelCorrectedLapTime'] - hard_stint['FuelCorrectedLapTime'].mean())**2).sum()
FuelCorrectedR2 = 1 - FuelCorrectedSSE/FuelCorrectedSST

#print(f'SSE = {round(FuelCorrectedSSE,3)}\nRMSE = {round(FuelCorrectedRMSE,3)}s\nSST = {round(FuelCorrectedSST,3)}\nR2 = {round(FuelCorrectedR2,3)}')


MinFuelEffectPerLap = qfmin*kfmin

hard_stint['MinFuelCorrectedLapTime'] = hard_stint['LapTime_sec'] + (hard_stint['LapNumber'] - hard_stint['LapNumber'].min())*MinFuelEffectPerLap
#print(hard_stint[['LapNumber', 'TyreLife', 'LapTime_sec', 'FuelCorrectedLapTime']])

bfmin, afmin = np.polyfit(x, hard_stint['MinFuelCorrectedLapTime'], 1)
#print(round(afmin,3),round(bfmin,3))

hard_stint['PredictedMinFuelCorrectedLapTime'] = afmin + bfmin * hard_stint['TyreLife']
hard_stint['MinFuelCorrectedResidual'] = hard_stint['MinFuelCorrectedLapTime'] - hard_stint['PredictedMinFuelCorrectedLapTime']

MinFuelCorrectedSSE = (hard_stint['MinFuelCorrectedResidual']**2).sum()
MinFuelCorrectedRMSE = np.sqrt(1/len(hard_stint['MinFuelCorrectedResidual'])*MinFuelCorrectedSSE)
MinFuelCorrectedSST = ((hard_stint['MinFuelCorrectedLapTime'] - hard_stint['MinFuelCorrectedLapTime'].mean())**2).sum()
MinFuelCorrectedR2 = 1 - MinFuelCorrectedSSE/MinFuelCorrectedSST

#print(f'SSE = {round(MinFuelCorrectedSSE,3)}\nRMSE = {round(MinFuelCorrectedRMSE,3)}s\nSST = {round(MinFuelCorrectedSST,3)}\nR2 = {round(MinFuelCorrectedR2,3)}')



MaxFuelEffectPerLap = qfmax*kfmax

hard_stint['MaxFuelCorrectedLapTime'] = hard_stint['LapTime_sec'] + (hard_stint['LapNumber'] - hard_stint['LapNumber'].min())*MaxFuelEffectPerLap
#print(hard_stint[['LapNumber', 'TyreLife', 'LapTime_sec', 'FuelCorrectedLapTime']])

bfmax, afmax = np.polyfit(x, hard_stint['MaxFuelCorrectedLapTime'], 1)
#print(round(afmax,3),round(bfmax,3))

hard_stint['PredictedMaxFuelCorrectedLapTime'] = afmax + bfmax * hard_stint['TyreLife']
hard_stint['MaxFuelCorrectedResidual'] = hard_stint['MaxFuelCorrectedLapTime'] - hard_stint['PredictedMaxFuelCorrectedLapTime']

MaxFuelCorrectedSSE = (hard_stint['MaxFuelCorrectedResidual']**2).sum()
MaxFuelCorrectedRMSE = np.sqrt(1/len(hard_stint['MaxFuelCorrectedResidual'])*MaxFuelCorrectedSSE)
MaxFuelCorrectedSST = ((hard_stint['MaxFuelCorrectedLapTime'] - hard_stint['MaxFuelCorrectedLapTime'].mean())**2).sum()
MaxFuelCorrectedR2 = 1 - MaxFuelCorrectedSSE/MaxFuelCorrectedSST

#print(f'SSE = {round(MaxFuelCorrectedSSE,3)}\nRMSE = {round(MaxFuelCorrectedRMSE,3)}s\nSST = {round(MaxFuelCorrectedSST,3)}\nR2 = {round(MaxFuelCorrectedR2,3)}')

#print('\n\n Stint 1 : Soft')

soft_stint1 = model_laps.loc[model_laps['Stint']==1].copy()

#Coefficients of the linear regression
soft_slope, soft_intercept = np.polyfit(soft_stint1['TyreLife'],soft_stint1['LapTime_sec'],1)
#print(f'intercept = {round(soft_intercept,3)},slope = {round(soft_slope,3)}')

soft_stint1['PredictedLapTime'] = soft_intercept+soft_slope*soft_stint1['TyreLife']

fig7 = plt.figure()
compound = soft_stint1['Compound'].iloc[0]
plt.plot(soft_stint1['TyreLife'],soft_stint1['LapTime_sec'],markers[compound],label='Lap time')
plt.plot(soft_stint1['TyreLife'],soft_stint1['PredictedLapTime'],label='Predicted time')

plt.title('Linear regression of 1st stint (soft)')
plt.xlabel('Tyre age [lap]')
plt.ylabel('Lap time [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

#Introduction of residual
soft_stint1['Residual'] = soft_stint1['LapTime_sec'] - soft_stint1['PredictedLapTime']

fig8 = plt.figure()
plt.plot(soft_stint1['TyreLife'],soft_stint1['Residual'],'+')
plt.axhline(0,color='r')

plt.title('Residual of linear regression 1st stint')
plt.xlabel('Tyre age [laps]')
plt.ylabel('Residual [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

mean_residual = soft_stint1['Residual'].mean()
RMSE = np.sqrt(1/len(soft_stint1['Residual']) * (soft_stint1['Residual']**2).sum())
max_abs_residual = soft_stint1['Residual'].abs().max()
#print(f'Mean residual = {round(mean_residual,3)}s \nRMSE = {round(RMSE,3)}s \nMaximum absolute residual = {round(max_abs_residual,3)}s')

SSE = (soft_stint1['Residual']**2).sum()
SST = ((soft_stint1['LapTime_sec']-soft_stint1['LapTime_sec'].mean())**2).sum()
R2 = 1 - SSE/SST
#print(f'SSE = {round(SSE,3)}\nSST = {round(SST,3)}\nR2 = {round(R2,3)}')

soft_stint1['FuelCorrectedLapTime'] = soft_stint1['LapTime_sec'] + (soft_stint1['LapNumber'] - soft_stint1['LapNumber'].min())*FuelEffectPerLap

#print('\n\n Nominal')

soft_stint1['FuelCorrectedLapTime'] = soft_stint1['LapTime_sec'] + (soft_stint1['LapNumber'] - soft_stint1['LapNumber'].min())*FuelEffectPerLap

bf, af = np.polyfit(soft_stint1['TyreLife'], soft_stint1['FuelCorrectedLapTime'], 1)
#print(f'af = {round(af,3)},bf = {round(bf,3)}')

soft_stint1['PredictedFuelCorrectedLapTime'] = af + bf * soft_stint1['TyreLife']
soft_stint1['FuelCorrectedResidual'] = soft_stint1['FuelCorrectedLapTime'] - soft_stint1['PredictedFuelCorrectedLapTime']

FuelCorrectedSSE = (soft_stint1['FuelCorrectedResidual']**2).sum()
FuelCorrectedRMSE = np.sqrt(1/len(soft_stint1['FuelCorrectedResidual'])*FuelCorrectedSSE)
FuelCorrectedSST = ((soft_stint1['FuelCorrectedLapTime'] - soft_stint1['FuelCorrectedLapTime'].mean())**2).sum()
FuelCorrectedR2 = 1 - FuelCorrectedSSE/FuelCorrectedSST

#print(f'SSE = {round(FuelCorrectedSSE,3)}\nRMSE = {round(FuelCorrectedRMSE,3)}s\nSST = {round(FuelCorrectedSST,3)}\nR2 = {round(FuelCorrectedR2,3)}')

#print('\n\n MINIMUM Case')

soft_stint1['MinFuelCorrectedLapTime'] = soft_stint1['LapTime_sec'] + (soft_stint1['LapNumber'] - soft_stint1['LapNumber'].min())*MinFuelEffectPerLap

bfmin, afmin = np.polyfit(soft_stint1['TyreLife'], soft_stint1['MinFuelCorrectedLapTime'], 1)
#print(f'afmin = {round(afmin,3)},bfmin = {round(bfmin,3)}')

soft_stint1['PredictedMinFuelCorrectedLapTime'] = afmin + bfmin * soft_stint1['TyreLife']
soft_stint1['MinFuelCorrectedResidual'] = soft_stint1['MinFuelCorrectedLapTime'] - soft_stint1['PredictedMinFuelCorrectedLapTime']

MinFuelCorrectedSSE = (soft_stint1['MinFuelCorrectedResidual']**2).sum()
MinFuelCorrectedRMSE = np.sqrt(1/len(soft_stint1['MinFuelCorrectedResidual'])*MinFuelCorrectedSSE)
MinFuelCorrectedSST = ((soft_stint1['MinFuelCorrectedLapTime'] - soft_stint1['MinFuelCorrectedLapTime'].mean())**2).sum()
MinFuelCorrectedR2 = 1 - MinFuelCorrectedSSE/MinFuelCorrectedSST

#print(f'SSE = {round(MinFuelCorrectedSSE,3)}\nRMSE = {round(MinFuelCorrectedRMSE,3)}s\nSST = {round(MinFuelCorrectedSST,3)}\nR2 = {round(MinFuelCorrectedR2,3)}')

#print('\n\n MAXIMUM Case')

soft_stint1['MaxFuelCorrectedLapTime'] = soft_stint1['LapTime_sec'] + (soft_stint1['LapNumber'] - soft_stint1['LapNumber'].min())*MaxFuelEffectPerLap

bfmax, afmax = np.polyfit(soft_stint1['TyreLife'], soft_stint1['MaxFuelCorrectedLapTime'], 1)
#print(f'afmax = {round(afmax,3)},bfmax = {round(bfmax,3)}')

soft_stint1['PredictedMaxFuelCorrectedLapTime'] = afmax + bfmax * soft_stint1['TyreLife']
soft_stint1['MaxFuelCorrectedResidual'] = soft_stint1['MaxFuelCorrectedLapTime'] - soft_stint1['PredictedMaxFuelCorrectedLapTime']

MaxFuelCorrectedSSE = (soft_stint1['MaxFuelCorrectedResidual']**2).sum()
MaxFuelCorrectedRMSE = np.sqrt(1/len(soft_stint1['MaxFuelCorrectedResidual'])*MaxFuelCorrectedSSE)
MaxFuelCorrectedSST = ((soft_stint1['MaxFuelCorrectedLapTime'] - soft_stint1['MaxFuelCorrectedLapTime'].mean())**2).sum()
MaxFuelCorrectedR2 = 1 - MaxFuelCorrectedSSE/MaxFuelCorrectedSST

#print(f'SSE = {round(MaxFuelCorrectedSSE,3)}\nRMSE = {round(MaxFuelCorrectedRMSE,3)}s\nSST = {round(MaxFuelCorrectedSST,3)}\nR2 = {round(MaxFuelCorrectedR2,3)}')

soft_stint1['AbsResidual'] = soft_stint1['Residual'].abs()
print(soft_stint1[['LapNumber','TyreLife','LapTime_sec','Residual','AbsResidual']].sort_values(by="AbsResidual", ascending=False).head(5))



#print('\n\n Stint 3 : Soft')

soft_stint3 = model_laps.loc[model_laps['Stint']==3].copy()

#Coefficients of the linear regression
soft_slope, soft_intercept = np.polyfit(soft_stint3['TyreLife'],soft_stint3['LapTime_sec'],1)
#print(f'intercept = {round(soft_intercept,3)},slope = {round(soft_slope,3)}')

soft_stint3['PredictedLapTime'] = soft_intercept+soft_slope*soft_stint3['TyreLife']

fig9 = plt.figure()
compound = soft_stint3['Compound'].iloc[0]
plt.plot(soft_stint3['TyreLife'],soft_stint3['LapTime_sec'],markers[compound],label='Lap time')
plt.plot(soft_stint3['TyreLife'],soft_stint3['PredictedLapTime'],label='Predicted time')

plt.title('Linear regression of 3rd stint (soft)')
plt.xlabel('Tyre age [lap]')
plt.ylabel('Lap time [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

#Introduction of residual
soft_stint3['Residual'] = soft_stint3['LapTime_sec'] - soft_stint3['PredictedLapTime']

fig10 = plt.figure()
plt.plot(soft_stint3['TyreLife'],soft_stint3['Residual'],'+')
plt.axhline(0,color='r')

plt.title('Residual of linear regression 3rd stint')
plt.xlabel('Tyre age [laps]')
plt.ylabel('Residual [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

mean_residual = soft_stint3['Residual'].mean()
RMSE = np.sqrt(1/len(soft_stint3['Residual']) * (soft_stint3['Residual']**2).sum())
max_abs_residual = soft_stint3['Residual'].abs().max()
#print(f'Mean residual = {round(mean_residual,3)}s \nRMSE = {round(RMSE,3)}s \nMaximum absolute residual = {round(max_abs_residual,3)}s')

SSE = (soft_stint3['Residual']**2).sum()
SST = ((soft_stint3['LapTime_sec']-soft_stint3['LapTime_sec'].mean())**2).sum()
R2 = 1 - SSE/SST
#print(f'SSE = {round(SSE,3)}\nSST = {round(SST,3)}\nR2 = {round(R2,3)}')

soft_stint3['FuelCorrectedLapTime'] = soft_stint3['LapTime_sec'] + (soft_stint3['LapNumber'] - soft_stint3['LapNumber'].min())*FuelEffectPerLap

#print('\n\n Nominal')

soft_stint3['FuelCorrectedLapTime'] = soft_stint3['LapTime_sec'] + (soft_stint3['LapNumber'] - soft_stint3['LapNumber'].min())*FuelEffectPerLap

bf, af = np.polyfit(soft_stint3['TyreLife'], soft_stint3['FuelCorrectedLapTime'], 1)
#print(f'af = {round(af,3)},bf = {round(bf,3)}')

soft_stint3['PredictedFuelCorrectedLapTime'] = af + bf * soft_stint3['TyreLife']
soft_stint3['FuelCorrectedResidual'] = soft_stint3['FuelCorrectedLapTime'] - soft_stint3['PredictedFuelCorrectedLapTime']

FuelCorrectedSSE = (soft_stint3['FuelCorrectedResidual']**2).sum()
FuelCorrectedRMSE = np.sqrt(1/len(soft_stint3['FuelCorrectedResidual'])*FuelCorrectedSSE)
FuelCorrectedSST = ((soft_stint3['FuelCorrectedLapTime'] - soft_stint3['FuelCorrectedLapTime'].mean())**2).sum()
FuelCorrectedR2 = 1 - FuelCorrectedSSE/FuelCorrectedSST

#print(f'SSE = {round(FuelCorrectedSSE,3)}\nRMSE = {round(FuelCorrectedRMSE,3)}s\nSST = {round(FuelCorrectedSST,3)}\nR2 = {round(FuelCorrectedR2,3)}')

#print('\n\n MINIMUM Case')

soft_stint3['MinFuelCorrectedLapTime'] = soft_stint3['LapTime_sec'] + (soft_stint3['LapNumber'] - soft_stint3['LapNumber'].min())*MinFuelEffectPerLap

bfmin, afmin = np.polyfit(soft_stint3['TyreLife'], soft_stint3['MinFuelCorrectedLapTime'], 1)
#print(f'afmin = {round(afmin,3)},bfmin = {round(bfmin,3)}')

soft_stint3['PredictedMinFuelCorrectedLapTime'] = afmin + bfmin * soft_stint3['TyreLife']
soft_stint3['MinFuelCorrectedResidual'] = soft_stint3['MinFuelCorrectedLapTime'] - soft_stint3['PredictedMinFuelCorrectedLapTime']

MinFuelCorrectedSSE = (soft_stint3['MinFuelCorrectedResidual']**2).sum()
MinFuelCorrectedRMSE = np.sqrt(1/len(soft_stint3['MinFuelCorrectedResidual'])*MinFuelCorrectedSSE)
MinFuelCorrectedSST = ((soft_stint3['MinFuelCorrectedLapTime'] - soft_stint3['MinFuelCorrectedLapTime'].mean())**2).sum()
MinFuelCorrectedR2 = 1 - MinFuelCorrectedSSE/MinFuelCorrectedSST

#print(f'SSE = {round(MinFuelCorrectedSSE,3)}\nRMSE = {round(MinFuelCorrectedRMSE,3)}s\nSST = {round(MinFuelCorrectedSST,3)}\nR2 = {round(MinFuelCorrectedR2,3)}')

#print('\n\n MAXIMUM Case')

soft_stint3['MaxFuelCorrectedLapTime'] = soft_stint3['LapTime_sec'] + (soft_stint3['LapNumber'] - soft_stint3['LapNumber'].min())*MaxFuelEffectPerLap

bfmax, afmax = np.polyfit(soft_stint3['TyreLife'], soft_stint3['MaxFuelCorrectedLapTime'], 1)
#print(f'afmax = {round(afmax,3)},bfmax = {round(bfmax,3)}')

soft_stint3['PredictedMaxFuelCorrectedLapTime'] = afmax + bfmax * soft_stint3['TyreLife']
soft_stint3['MaxFuelCorrectedResidual'] = soft_stint3['MaxFuelCorrectedLapTime'] - soft_stint3['PredictedMaxFuelCorrectedLapTime']

MaxFuelCorrectedSSE = (soft_stint3['MaxFuelCorrectedResidual']**2).sum()
MaxFuelCorrectedRMSE = np.sqrt(1/len(soft_stint3['MaxFuelCorrectedResidual'])*MaxFuelCorrectedSSE)
MaxFuelCorrectedSST = ((soft_stint3['MaxFuelCorrectedLapTime'] - soft_stint3['MaxFuelCorrectedLapTime'].mean())**2).sum()
MaxFuelCorrectedR2 = 1 - MaxFuelCorrectedSSE/MaxFuelCorrectedSST

#print(f'SSE = {round(MaxFuelCorrectedSSE,3)}\nRMSE = {round(MaxFuelCorrectedRMSE,3)}s\nSST = {round(MaxFuelCorrectedSST,3)}\nR2 = {round(MaxFuelCorrectedR2,3)}')

#Test exclure tours aberrants
soft_stint3_test = soft_stint3.iloc[3:].copy()
#print(soft_stint3_test[['LapNumber','TyreLife','LapTime_sec']])

#Coefficients of the linear regression
soft_slope, soft_intercept = np.polyfit(soft_stint3_test['TyreLife'],soft_stint3_test['LapTime_sec'],1)
#print(f'intercept = {round(soft_intercept,3)},slope = {round(soft_slope,3)}')

soft_stint3_test['PredictedLapTime'] = soft_intercept+soft_slope*soft_stint3_test['TyreLife']

fig11 = plt.figure()
compound = soft_stint3_test['Compound'].iloc[0]
plt.plot(soft_stint3_test['TyreLife'],soft_stint3_test['LapTime_sec'],markers[compound],label='Lap time')
plt.plot(soft_stint3_test['TyreLife'],soft_stint3_test['PredictedLapTime'],label='Predicted time')

plt.title('Linear regression of 3rd stint (soft)')
plt.xlabel('Tyre age [lap]')
plt.ylabel('Lap time [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

#Introduction of residual
soft_stint3_test['Residual'] = soft_stint3_test['LapTime_sec'] - soft_stint3_test['PredictedLapTime']

fig12 = plt.figure()
plt.plot(soft_stint3_test['TyreLife'],soft_stint3_test['Residual'],'+')
plt.axhline(0,color='r')

plt.title('Residual of linear regression 3rd stint')
plt.xlabel('Tyre age [laps]')
plt.ylabel('Residual [s]')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.grid(True, alpha=0.25)

mean_residual = soft_stint3_test['Residual'].mean()
RMSE = np.sqrt(1/len(soft_stint3_test['Residual']) * (soft_stint3_test['Residual']**2).sum())
max_abs_residual = soft_stint3_test['Residual'].abs().max()
#print(f'Mean residual = {round(mean_residual,3)}s \nRMSE = {round(RMSE,3)}s \nMaximum absolute residual = {round(max_abs_residual,3)}s')

SSE = (soft_stint3_test['Residual']**2).sum()
SST = ((soft_stint3_test['LapTime_sec']-soft_stint3_test['LapTime_sec'].mean())**2).sum()
R2 = 1 - SSE/SST
#print(f'SSE = {round(SSE,3)}\nSST = {round(SST,3)}\nR2 = {round(R2,3)}')

#print('\n\n Nominal')

soft_stint3_test['FuelCorrectedLapTime'] = soft_stint3_test['LapTime_sec'] + (soft_stint3_test['LapNumber'] - soft_stint3_test['LapNumber'].min())*FuelEffectPerLap
#print(soft_stint3_test[['LapNumber', 'TyreLife', 'LapTime_sec', 'FuelCorrectedLapTime']])

bf, af = np.polyfit(soft_stint3_test['TyreLife'], soft_stint3_test['FuelCorrectedLapTime'], 1)
#print(f'af = {round(af,3)},bf = {round(bf,3)}')

soft_stint3_test['PredictedFuelCorrectedLapTime'] = af + bf * soft_stint3_test['TyreLife']
soft_stint3_test['FuelCorrectedResidual'] = soft_stint3_test['FuelCorrectedLapTime'] - soft_stint3_test['PredictedFuelCorrectedLapTime']

FuelCorrectedSSE = (soft_stint3_test['FuelCorrectedResidual']**2).sum()
FuelCorrectedRMSE = np.sqrt(1/len(soft_stint3_test['FuelCorrectedResidual'])*FuelCorrectedSSE)
FuelCorrectedSST = ((soft_stint3_test['FuelCorrectedLapTime'] - soft_stint3_test['FuelCorrectedLapTime'].mean())**2).sum()
FuelCorrectedR2 = 1 - FuelCorrectedSSE/FuelCorrectedSST

#print(f'SSE = {round(FuelCorrectedSSE,3)}\nRMSE = {round(FuelCorrectedRMSE,3)}s\nSST = {round(FuelCorrectedSST,3)}\nR2 = {round(FuelCorrectedR2,3)}')

#print('\n\n MINIMUM Case')
soft_stint3_test['MinFuelCorrectedLapTime'] = soft_stint3_test['LapTime_sec'] + (soft_stint3_test['LapNumber'] - soft_stint3_test['LapNumber'].min())*MinFuelEffectPerLap
#print(soft_stint3_test[['LapNumber', 'TyreLife', 'LapTime_sec', 'FuelCorrectedLapTime']])

bfmin, afmin = np.polyfit(soft_stint3_test['TyreLife'], soft_stint3_test['MinFuelCorrectedLapTime'], 1)
#print(f'afmin = {round(afmin,3)},bfmin = {round(bfmin,3)}')

soft_stint3_test['PredictedMinFuelCorrectedLapTime'] = afmin + bfmin * soft_stint3_test['TyreLife']
soft_stint3_test['MinFuelCorrectedResidual'] = soft_stint3_test['MinFuelCorrectedLapTime'] - soft_stint3_test['PredictedMinFuelCorrectedLapTime']

MinFuelCorrectedSSE = (soft_stint3_test['MinFuelCorrectedResidual']**2).sum()
MinFuelCorrectedRMSE = np.sqrt(1/len(soft_stint3_test['MinFuelCorrectedResidual'])*MinFuelCorrectedSSE)
MinFuelCorrectedSST = ((soft_stint3_test['MinFuelCorrectedLapTime'] - soft_stint3_test['MinFuelCorrectedLapTime'].mean())**2).sum()
MinFuelCorrectedR2 = 1 - MinFuelCorrectedSSE/MinFuelCorrectedSST

#print(f'SSE = {round(MinFuelCorrectedSSE,3)}\nRMSE = {round(MinFuelCorrectedRMSE,3)}s\nSST = {round(MinFuelCorrectedSST,3)}\nR2 = {round(MinFuelCorrectedR2,3)}')


#print('\n\n MAXIMUM Case')

soft_stint3_test['MaxFuelCorrectedLapTime'] = soft_stint3_test['LapTime_sec'] + (soft_stint3_test['LapNumber'] - soft_stint3_test['LapNumber'].min())*MaxFuelEffectPerLap
#print(soft_stint3_test[['LapNumber', 'TyreLife', 'LapTime_sec', 'FuelCorrectedLapTime']])

bfmax, afmax = np.polyfit(soft_stint3_test['TyreLife'], soft_stint3_test['MaxFuelCorrectedLapTime'], 1)
#print(f'afmax = {round(afmax,3)},bfmax = {round(bfmax,3)}')

soft_stint3_test['PredictedMaxFuelCorrectedLapTime'] = afmax + bfmax * soft_stint3_test['TyreLife']
soft_stint3_test['MaxFuelCorrectedResidual'] = soft_stint3_test['MaxFuelCorrectedLapTime'] - soft_stint3_test['PredictedMaxFuelCorrectedLapTime']

MaxFuelCorrectedSSE = (soft_stint3_test['MaxFuelCorrectedResidual']**2).sum()
MaxFuelCorrectedRMSE = np.sqrt(1/len(soft_stint3_test['MaxFuelCorrectedResidual'])*MaxFuelCorrectedSSE)
MaxFuelCorrectedSST = ((soft_stint3_test['MaxFuelCorrectedLapTime'] - soft_stint3_test['MaxFuelCorrectedLapTime'].mean())**2).sum()
MaxFuelCorrectedR2 = 1 - MaxFuelCorrectedSSE/MaxFuelCorrectedSST

#print(f'SSE = {round(MaxFuelCorrectedSSE,3)}\nRMSE = {round(MaxFuelCorrectedRMSE,3)}s\nSST = {round(MaxFuelCorrectedSST,3)}\nR2 = {round(MaxFuelCorrectedR2,3)}')


plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)
plt.close(fig6)
plt.close(fig7)
plt.close(fig8)

#plt.legend()
plt.show()