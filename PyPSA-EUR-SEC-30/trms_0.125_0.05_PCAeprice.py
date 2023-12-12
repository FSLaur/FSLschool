# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:44:53 2023

@author: laur1
"""

# Import packages
import pypsa
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os.path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(1,r"C:\Users\laur1\OneDrive\4. Civil - semester\Code")
from FunctionsPM import load_ALL
from FunctionsPM import importCSV
from FunctionsPM import meanYear
from FunctionsPM import meanMonth
from FunctionsPM import meanWeek
from FunctionsPM import minYear
from FunctionsPM import modGenerators
from FunctionsPM import getDroughts
from FunctionsPM import savefigure
from FunctionsPM import savefigure1
from FunctionsPM import savefigure2
from FunctionsPM import getGenandstores
from FunctionsPM import folders
from FunctionsPM import changeindex
from FunctionsPM import LinRegCountryPlot
from FunctionsPM import LinRegSystemPlot
from FunctionsPM import sumYear
from FunctionsPM import getminCFmonth
from FunctionsPM import getSeasonalCF

from FunctionsPM import PCA
from FunctionsPM import MAPPLOT
from FunctionsPM import season_plot
from FunctionsPM import FFT_plot
from FunctionsPM import screeplot
from FunctionsPM import AmplitudeArea
from FunctionsPM import AmplitudeAreaPC1PC2

constraint3 = '2XTRMS'
NAMES = ['AT', 'BA','BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB',
       'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT',
       'RO','RS','SE', 'SI', 'SK']
#%% Load 3 cheapest and 3 most expensive year
# 3 Cheapest sorted
file1998 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_1998_0.05.h5"
network1998 = pypsa.Network(file1998)
network1998.name = file1998

file1999 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_1999_0.05.h5"
network1999 = pypsa.Network(file1999)
network1999.name = file1999

file2000 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_2000_0.05.h5"
network2000 = pypsa.Network(file2000)
network2000.name = file2000

# 3 most expensive sorted
file2014 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_2014_0.05.h5"
network2014 = pypsa.Network(file2014)
network2014.name = file2014

file2013 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_2013_0.05.h5"
network2013 = pypsa.Network(file2013)
network2013.name = file2013

file1987 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_1987_0.05.h5"
network1987 = pypsa.Network(file1987)
network1987.name = file1987

#%% time index
time_index = network2014.loads_t.p.index


#%% Nodal or marginal price
# Cutting price above 1000 EUR/MWh, reason it will change PCA
# 1998
eprice1998 = network1998.buses_t.marginal_price[NAMES]
eprice1998 = np.clip(eprice1998,0,1000)

# 1999
eprice1999 = network1999.buses_t.marginal_price[NAMES]
eprice1999 = np.clip(eprice1999,0,1000)

# 2000
eprice2000 = network2000.buses_t.marginal_price[NAMES]
eprice2000 = np.clip(eprice2000,0,1000)

# 1987
eprice1987 = network1987.buses_t.marginal_price[NAMES]
eprice1987 = np.clip(eprice1987,0,1000)

# 2013
eprice2013 = network2013.buses_t.marginal_price[NAMES]
eprice2013 = np.clip(eprice2013,0,1000)

# 2014
eprice2014 = network2014.buses_t.marginal_price[NAMES]
eprice2014 = np.clip(eprice2014,0,1000)

#%%
eprice = [sumYear(eprice1998*network2014.loads_t.p),sumYear(eprice1999*network2014.loads_t.p),sumYear(eprice2000*network2014.loads_t.p),sumYear(eprice1987*network2014.loads_t.p),sumYear(eprice2013*network2014.loads_t.p),sumYear(eprice2014*network2014.loads_t.p)]
eprice = pd.concat(eprice)
eprice = (eprice)/network2014.loads_t.p.sum()
Loadfactor = network2014.loads_t.p.sum()/network2014.loads_t.p.sum().sum()
eprice_sum = (eprice*Loadfactor).sum(axis=1)
#%% PCA for 3 cheapest and 3 most expensive years

# PCA
eigen_values1998, eigen_vectors1998, Cov_mat1998, c1998, variance_explained1998, VT1998, X_norm1998 = PCA(eprice1998,NAMES)
eigen_values1999, eigen_vectors1999, Cov_mat1999, c1999, variance_explained1999, VT1999, X_norm1999 = PCA(eprice1999,NAMES)
eigen_values2000, eigen_vectors2000, Cov_mat2000, c2000, variance_explained2000, VT2000, X_norm2000 = PCA(eprice2000,NAMES)
eigen_values1987, eigen_vectors1987, Cov_mat1987, c1987, variance_explained1987, VT1987, X_norm1987 = PCA(eprice1987,NAMES)
eigen_values2013, eigen_vectors2013, Cov_mat2013, c2013, variance_explained2013, VT2013, X_norm2013 = PCA(eprice2013,NAMES)
eigen_values2014, eigen_vectors2014, Cov_mat2014, c2014, variance_explained2014, VT2014, X_norm2014 = PCA(eprice2014,NAMES)
years = [network1998,network1999,network2000,network1987,network2013,network2014]

# OBS when changing direction, we want same pattern sign of eig_vec does NOT matter
# 1998 reference year 
VT1998[[0,1,2]] = -1*VT1998[[0,1,2]]
VT1999[[1,2]] = -1*VT1999[[1,2]]
VT2000[[1,3]] = -1*VT2000[[1,3]] 
VT1987[[0,3]] = -1*VT1987[[0,3]] 
VT2013[[0,1]] = -1*VT2013[[0,1]]
VT2014[[3]] = -1*VT2014[[3]]

# Calculate Amplitudes of eigenvectors
a_k1998 = np.dot(X_norm1998,VT1998)
a_k1999 = np.dot(X_norm1999,VT1999)
a_k2000 = np.dot(X_norm2000,VT2000)
a_k1987 = np.dot(X_norm1987,VT1987)
a_k2013 = np.dot(X_norm2013,VT2013)
a_k2014 = np.dot(X_norm2014,VT2014)
# data frames
a_k1998df = pd.DataFrame(data=a_k1998,index=time_index)
a_k1999df = pd.DataFrame(data=a_k1999,index=time_index)
a_k2000df = pd.DataFrame(data=a_k2000,index=time_index)
a_k1987df = pd.DataFrame(data=a_k1987,index=time_index)
a_k2013df = pd.DataFrame(data=a_k2013,index=time_index)
a_k2014df = pd.DataFrame(data=a_k2014,index=time_index)

#%% Daily and hourly average values
a_k1998_avg_hour = a_k1998df.groupby(time_index.hour).mean() # Hour
a_k1998_avg_day = a_k1998df.groupby([time_index.month,time_index.day]).mean() # Day

a_k1999_avg_hour = a_k1999df.groupby(time_index.hour).mean() # Hour
a_k1999_avg_day = a_k1999df.groupby([time_index.month,time_index.day]).mean() # Day

a_k2000_avg_hour = a_k2000df.groupby(time_index.hour).mean() # Hour
a_k2000_avg_day = a_k2000df.groupby([time_index.month,time_index.day]).mean() # Day

a_k1987_avg_hour = a_k1987df.groupby(time_index.hour).mean() # Hour
a_k1987_avg_day = a_k1987df.groupby([time_index.month,time_index.day]).mean() # Day
 
a_k2013_avg_hour = a_k2013df.groupby(time_index.hour).mean() # Hour
a_k2013_avg_day = a_k2013df.groupby([time_index.month,time_index.day]).mean() # Day

a_k2014_avg_hour = a_k2014df.groupby(time_index.hour).mean() # Hour
a_k2014_avg_day = a_k2014df.groupby([time_index.month,time_index.day]).mean() # Day

# Hourly PC1
a_k_avg_hourPC1max = max([a_k1998_avg_hour[0].max(),a_k1999_avg_hour[0].max(),a_k2000_avg_hour[0].max(),
                      a_k1987_avg_hour[0].max(),a_k2013_avg_hour[0].max(),a_k2014_avg_hour[0].max()])
a_k_avg_hourPC1min = min([a_k1998_avg_hour[0].min(),a_k1999_avg_hour[0].min(),a_k2000_avg_hour[0].min(),
                      a_k1987_avg_hour[0].min(),a_k2013_avg_hour[0].min(),a_k2014_avg_hour[0].min()])
a_k_avghour = (a_k_avg_hourPC1max,a_k_avg_hourPC1min)

# Daily PC1
a_k_avg_dailyPC1max = max([a_k1998_avg_day[0].max(),a_k1999_avg_day[0].max(),a_k2000_avg_day[0].max(),
                      a_k1987_avg_day[0].max(),a_k2013_avg_day[0].max(),a_k2014_avg_day[0].max()])
a_k_avg_dailyPC1min = min([a_k1998_avg_day[0].min(),a_k1999_avg_day[0].min(),a_k2000_avg_day[0].min(),
                      a_k1987_avg_day[0].min(),a_k2013_avg_day[0].min(),a_k2014_avg_day[0].min()])

# Daily PC2
a_k_avg_dailyPC2max = max([a_k1998_avg_day[1].max(),a_k1999_avg_day[1].max(),a_k2000_avg_day[1].max(),
                      a_k1987_avg_day[1].max(),a_k2013_avg_day[1].max(),a_k2014_avg_day[1].max()])
a_k_avg_dailyPC2min = min([a_k1998_avg_day[1].min(),a_k1999_avg_day[1].min(),a_k2000_avg_day[1].min(),
                      a_k1987_avg_day[1].min(),a_k2013_avg_day[1].min(),a_k2014_avg_day[1].min()])

# Daily PC3
a_k_avg_dailyPC3max = max([a_k1998_avg_day[2].max(),a_k1999_avg_day[2].max(),a_k2000_avg_day[2].max(),
                      a_k1987_avg_day[2].max(),a_k2013_avg_day[2].max(),a_k2014_avg_day[2].max()])
a_k_avg_dailyPC3min = min([a_k1998_avg_day[2].min(),a_k1999_avg_day[2].min(),a_k2000_avg_day[2].min(),
                      a_k1987_avg_day[2].min(),a_k2013_avg_day[2].min(),a_k2014_avg_day[2].min()])
# Assembling
a_k_avgdailymax = (a_k_avg_dailyPC1max,a_k_avg_dailyPC2max,a_k_avg_dailyPC3max)
a_k_avgdailymin = (a_k_avg_dailyPC1min,a_k_avg_dailyPC2min,a_k_avg_dailyPC3min)

#%% Amplitude daily average duration

# Area1998, idx1998, MaxArea1998, a_kday1998 = AmplitudeArea(a_k1998df) # 1998
# Area1999, idx1999, MaxArea1999, a_kday1999 = AmplitudeArea(a_k1999df) # 1999
# Area2000, idx2000, MaxArea2000, a_kday2000 = AmplitudeArea(a_k2000df) # 2000
# Area1987, idx1987, MaxArea1987, a_kday1987 = AmplitudeArea(a_k1987df) # 1987
# Area2013, idx2013, MaxArea2013, a_kday2013 = AmplitudeArea(a_k2013df) # 2013
# Area2014, idx2014, MaxArea2014, a_kday2014 = AmplitudeArea(a_k2014df) # 2014

Area1998, idx1998, MaxArea1998, a_kday1998 = AmplitudeAreaPC1PC2(a_k1998df) # 1998
Area1999, idx1999, MaxArea1999, a_kday1999 = AmplitudeAreaPC1PC2(a_k1999df) # 1999
Area2000, idx2000, MaxArea2000, a_kday2000 = AmplitudeAreaPC1PC2(a_k2000df) # 2000
Area1987, idx1987, MaxArea1987, a_kday1987 = AmplitudeAreaPC1PC2(a_k1987df) # 1987
Area2013, idx2013, MaxArea2013, a_kday2013 = AmplitudeAreaPC1PC2(a_k2013df) # 2013
Area2014, idx2014, MaxArea2014, a_kday2014 = AmplitudeAreaPC1PC2(a_k2014df) # 2014



#%% Plot for all area durations curve
fig,ax=plt.subplots(figsize = (12,6))
yearint = [1,2,3,4,5,6] # [1998,1999,2000,1987,2013,2014] this is the order
y1 = [MaxArea1998['Area'],MaxArea1999['Area'],MaxArea2000['Area'],MaxArea1987['Area'],MaxArea2013['Area'],MaxArea2014['Area']]
y2 = [Area1998['AreaStep'].sum(),Area1999['AreaStep'].sum(),Area2000['AreaStep'].sum(),Area1987['AreaStep'].sum(),Area2013['AreaStep'].sum(),Area2014['AreaStep'].sum()]
y3 = [MaxArea1998['Count'],MaxArea1999['Count'],MaxArea2000['Count'],MaxArea1987['Count'],MaxArea2013['Count'],MaxArea2014['Count']]
labels = ['0','1998','1999','2000','1987','2013','2014']
l1, = ax.plot(yearint,y1, marker="o",color = 'lightblue',label = 'Maximum Area a1>0')
l2, = ax.plot(yearint,y2, marker="o",color = 'darkblue',label = 'Total Area a1>0')
ax.set_xlabel("Year",color="black",fontsize = 14, weight = 'bold')
ax.set_xticks(np.arange(len(labels))) # Remember to change
ax.set_xticklabels(labels)
ax.set_ylabel("Area",color="black",fontsize = 14, weight='bold')
ax.tick_params(axis='both', labelsize = 14)
ax.set_title('Amplitude Area PC1+PC2:'+' '+constraint3,fontsize = 20, weight = 'bold') #OBS check columns
ax2=ax.twinx()
l3, = ax2.plot(yearint,y3,".",color = 'black', ms = 14, label ='Duration')
ax2.set_ylabel("Days of duration for Maximum Area",color="black",fontsize=14, weight = 'bold')
ax2.set_ylim(0,max(y3)+2)
ax2.set_yticks(np.arange(0, max(y3)+2, 5.0))
plt.legend([l1,l2,l3],['Maximum Area a$_1$>0','Total Area a$_1$>0','Duration'], loc='upper left', borderaxespad=0.,fontsize=12)
plt.tight_layout()
savefigure1('System',constraint3,'PC1PC2areaplotAllyearsprice',fig)
#%% Scree plot
fig = screeplot(variance_explained1998,constraint3,'1998') # 1998
savefigure1('System', constraint3, 'screeplotprice'+'1998', fig)
fig = screeplot(variance_explained1999,constraint3,'1999') # 1999
savefigure1('System', constraint3, 'screeplotprice'+'1999', fig)
fig = screeplot(variance_explained2000,constraint3,'2000') # 2000
savefigure1('System', constraint3, 'screeplotprice'+'2000', fig)
fig = screeplot(variance_explained1987,constraint3,'1987') # 1987
savefigure1('System', constraint3, 'screeplotprice'+'1987', fig)
fig = screeplot(variance_explained2013,constraint3,'2013') # 2013
savefigure1('System', constraint3, 'screeplotprice'+'2013', fig)
fig = screeplot(variance_explained2014,constraint3,'2014') # 2014
savefigure1('System', constraint3, 'screeplotprice'+'2014', fig)

#%% MAPPLOT Cheapest years
fig = MAPPLOT(VT1998,constraint3,variance_explained1998,'1998') # 1998
savefigure1('System', constraint3, 'MAPPLOTprice'+'1998', fig)
#%% 1999 OBS when changing direction, we want same pattern sign of eig_vec does NOT matter
fig = MAPPLOT(VT1999,constraint3,variance_explained1999,'1999') # 1999
savefigure1('System', constraint3, 'MAPPLOTprice'+'1999', fig)
#%% 2000
fig = MAPPLOT(VT2000,constraint3,variance_explained2000,'2000') # 2000
savefigure1('System', constraint3, 'MAPPLOTprice'+'2000', fig)
#%% MAPPLOT most expensive years
fig = MAPPLOT(VT1987,constraint3,variance_explained1987,'1987') # 1987
savefigure1('System', constraint3, 'MAPPLOTprice'+'1987', fig)
#%%
fig = MAPPLOT(VT2013,constraint3,variance_explained2013,'2013') # 2013
savefigure1('System', constraint3, 'MAPPLOTprice'+'2013', fig)
#%%
fig = MAPPLOT(VT2014,constraint3,variance_explained2014,'2014') # 2014
savefigure1('System', constraint3, 'MAPPLOTprice'+'2014', fig)

#%% Season plot
fig = season_plot(a_k1998, time_index, constraint3,'1998',a_k_avgdailymax,a_k_avgdailymin,a_k_avghour) # 1998
savefigure1('System', constraint3, 'Seasonplotprice'+'1998', fig)
fig = season_plot(a_k1999, time_index, constraint3,'1999',a_k_avgdailymax,a_k_avgdailymin,a_k_avghour) # 1999
savefigure1('System', constraint3, 'Seasonplotprice'+'1999', fig)
fig = season_plot(a_k2000, time_index, constraint3,'2000',a_k_avgdailymax,a_k_avgdailymin,a_k_avghour) # 2000
savefigure1('System', constraint3, 'Seasonplotprice'+'2000', fig)
fig = season_plot(a_k1987, time_index, constraint3,'1987',a_k_avgdailymax,a_k_avgdailymin,a_k_avghour) # 1987
savefigure1('System', constraint3, 'Seasonplotprice'+'1987', fig)
fig = season_plot(a_k2013, time_index, constraint3,'2013',a_k_avgdailymax,a_k_avgdailymin,a_k_avghour) # 2013
savefigure1('System', constraint3, 'Seasonplotprice'+'2013', fig)
fig = season_plot(a_k2014, time_index, constraint3,'2014',a_k_avgdailymax,a_k_avgdailymin,a_k_avghour) # 2014
savefigure1('System', constraint3, 'Seasonplotprice'+'2014', fig)

#%% FFT PLOT
a_k1998df = pd.DataFrame(data=a_k1998,index=time_index)
a_k1999df = pd.DataFrame(data=a_k1999,index=time_index)
a_k2000df = pd.DataFrame(data=a_k2000,index=time_index)
a_k1987df = pd.DataFrame(data=a_k1987,index=time_index)
a_k2013df = pd.DataFrame(data=a_k2013,index=time_index)
a_k2014df = pd.DataFrame(data=a_k2014,index=time_index)

fig = FFT_plot(a_k1998df,constraint3,'1998') # 1998
savefigure1('System', constraint3, 'FFTprice'+'1998', fig)
fig = FFT_plot(a_k1999df,constraint3,'1999') # 1999
savefigure1('System', constraint3, 'FFTprice'+'1999', fig)
fig = FFT_plot(a_k2000df,constraint3,'2000') # 2000
savefigure1('System', constraint3, 'FFTprice'+'2000', fig)
fig = FFT_plot(a_k1987df,constraint3,'1987') # 1987
savefigure1('System', constraint3, 'FFTprice'+'1987', fig)
fig = FFT_plot(a_k2013df,constraint3,'2013') # 2013
savefigure1('System', constraint3, 'FFTprice'+'2013', fig)
fig = FFT_plot(a_k2014df,constraint3,'2014') # 2014
savefigure1('System', constraint3, 'FFTprice'+'2014', fig)
#%% PCA bar plot
# remember to change
yearint = [1998,1999,2000,1987,2013,2014]

# remember to change
variance_explainedALL = [variance_explained1998,variance_explained1999,variance_explained2000,variance_explained1987,
                         variance_explained2013,variance_explained2014]
variance_explaineddf = pd.DataFrame(columns=yearint)
labels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10-30']
j = 0
for i in yearint:
    variance_explaineddf[i] = variance_explainedALL[j]
    j +=1
variance_explaineddfplot = variance_explaineddf[0:9]
variance_explaineddfplot.loc[9] = 0     
variance_explaineddfplot.loc[9] = variance_explaineddf[9:].sum()
colors = ['b',"orange",'green','c','m',"y",'purple','pink','lightblue','k']
variance_explaineddfplot = variance_explaineddfplot.T  
fig = plt.figure()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12,8))
variance_explaineddfplot.plot(ax=axes,kind="bar", stacked=True,color=colors,rot=90,legend=False)
fig.legend(labels,loc='upper right', bbox_to_anchor=(1.16, 0.98),fontsize=17)
axes.tick_params(labelbottom=True,labelsize=16)
axes.set_xlabel('Years - [Cheapest to most expensive]',weight='bold',fontsize = 20)
axes.set_ylabel('Variance explained [%]',weight='bold',fontsize = 20)
axes.set_title('System '+'- '+constraint3,fontsize=20,weight='bold')
fig.tight_layout()
savefigure1('System', constraint3, 'electricityPCAplotprice', fig)
#%%
