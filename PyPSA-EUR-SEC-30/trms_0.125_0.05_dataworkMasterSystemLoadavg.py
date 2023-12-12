# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:05:06 2023

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

# Path - to load the data
year = np.arange(1979,2016,1)
#pathstart = r"C:\Users\laur1\OneDrive\3. Civil - semester\Specialization Project\postnetwork-elec_only_"
pathend = "_0.05.h5"
# OBBBBBBBS CHange when new dataset
constraint = "trms_0.125&CO2_0.05-New" # Used in titles
constraint2 = constraint.rstrip('-New') # Use for titles remember to change
constraint3 = '2XTRMS'

pathstart = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_"

NAMES = ['AT', 'BA','BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB',
       'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT',
       'RO','RS','SE', 'SI', 'SK']
#%% Import CSV
filepath = r"D:\Pre-project (Data)\pv_optimal.csv"
year_solar = np.arange(1979,2018,1)
CF_solar =  importCSV(filepath,year_solar)
filepath1 = r"D:\Pre-project (Data)\onshore_wind_1979-2017.csv"
CF_onWind = importCSV(filepath1,year_solar)
filepath2 = r"D:/Pre-project (Data)/offshore_wind_1979-2017.csv"
CF_offWind = importCSV(filepath2,year_solar)

OnesCF = pd.DataFrame(columns=CF_solar.columns,index=CF_solar.index)
OnesCF.fillna(1,inplace=True)
CF_onWind = CF_onWind/OnesCF
CF_onWind = CF_onWind.reindex(columns=CF_solar.columns)
CF_onWind.fillna(0,inplace=True)

#%% Import Data from System files
n,generators,generators_timeY,storage, storage_timeY,stores,stores_timeY, links,links_timep0Y,links_timep1Y,buses,systemcost,eprice,eprice_time,loads,hydroinflow_timeY,loads_timeY = load_ALL(year,pathstart,pathend)

#%%

#systemcost.to_csv('C:/Users/laur1/OneDrive/4. Civil - semester/systemcost2xtrms.csv',sep= ',')
#%% Mean capacity factors and loads

CF_solaryear = meanYear(CF_solar)
CF_solaryear = CF_solaryear[:len(CF_solaryear)-2]
#CF_solaryear = CF_solaryear.drop(['BIH','SRB'],axis = 1)


CF_onWindyear = meanYear(CF_onWind)
CF_onWindyear = CF_onWindyear[:len(CF_onWindyear)-2]




#loads = loads.drop(['BA','RS'],axis = 1)
#loads_timeY = loads_timeY.drop(['BA','RS'],axis = 1)


#%% Loadfactor and loadfactor capacity factors
# Rename column names for yearly capacity factor wind and solar so column names are matching
CF_solaryear.columns=loads.columns
CF_onWindyear.columns=loads.columns

loadssum = loads.sum(axis=0)
loadfactor = loadssum/loadssum.sum() # Same every year
#Scaled capacity factors
CF_solaryearloadfactortotal = CF_solaryear*loadfactor
CF_onWindyearloadfactortotal = CF_onWindyear*loadfactor

#%% CF minimum solar and wind
CFmeanmonthwind, CFtotalmeanmonth_allwind, CFminfactor_allwind, CFminfactor_yearwind, CFstdmonth_allwind, CFwind = getminCFmonth(CF_onWind,loads)
CFmeanmonthsolar, CFtotalmeanmonth_allsolar, CFminfactor_allsolar, CFminfactor_yearsolar, CFstdmonth_allsolar, CFsolar = getminCFmonth(CF_solar,loads)
# minimum montly CF factor for the system
CFminfactor_yearwindSystem = (CFminfactor_yearwind*loadfactor).sum(axis=1)
CFminfactor_yearsolarSystem = (CFminfactor_yearsolar*loadfactor).sum(axis=1)


#%% Plot CFmin factor DK 2014
CFmeanmonthwindDK14 = CFmeanmonthwind['DK'].loc['2014-01-31 00:00:00+00:00':'2014-12-31 00:00:00+00:00']
CFmeanmonthwindDK14.index=CFtotalmeanmonth_allwind.index
CFmeanmonthsolarDK14 = CFmeanmonthsolar['DK'].loc['2014-01-31 00:00:00+00:00':'2014-12-31 00:00:00+00:00']
CFmeanmonthsolarDK14.index = CFtotalmeanmonth_allsolar.index
fig = plt.figure(figsize=(12,6))
fig, plt.plot(CFmeanmonthwindDK14,'o',color='dodgerblue',label='$CF_{wind}$, montly avg. 2014')
fig, plt.plot(CFtotalmeanmonth_allwind['DK'],'o',color='Black',label='$CF_{wind}$, montly avg. all years')
fig, plt.fill_between(CFtotalmeanmonth_allwind['DK'].index, CFtotalmeanmonth_allwind['DK']+CFstdmonth_allwind['DK'],CFtotalmeanmonth_allwind['DK']-CFstdmonth_allwind['DK'],color='grey',alpha=0.2)
fig, plt.plot(CFmeanmonthsolarDK14,'x',color='orange',label='$CF_{solar}$, montly avg. 2014',ms=12)
fig, plt.plot(CFtotalmeanmonth_allsolar['DK'],'x',color='red',ms=12,label='$CF_{solar}$, montly avg. all years')
fig, plt.fill_between(CFtotalmeanmonth_allwind['DK'].index, CFtotalmeanmonth_allwind['DK']+CFstdmonth_allwind['DK'],CFtotalmeanmonth_allwind['DK']-CFstdmonth_allwind['DK'],color='lightblue',alpha=0.3,label='$\sigma_{wind}$, standard deviation')
fig, plt.fill_between(CFtotalmeanmonth_allsolar['DK'].index, CFtotalmeanmonth_allsolar['DK']+CFstdmonth_allsolar['DK'],CFtotalmeanmonth_allsolar['DK']-CFstdmonth_allsolar['DK'],color='yellow',alpha=0.3,label='$\sigma_{solar}$, standard deviation')
fig, plt.vlines(8,CFmeanmonthwindDK14[8],CFtotalmeanmonth_allwind['DK'][8],ls='dashed',color='grey', lw=3)
fig, plt.vlines(0,CFmeanmonthsolarDK14[0],CFtotalmeanmonth_allsolar['DK'][0],ls='dashed',color='grey', lw=3)
fig, plt.title('Denmark: Monthly avg. Capacity Factors - 2014',fontsize=18,weight='bold')
fig, plt.xlabel('Months', fontsize = 15, weight='bold')
fig, plt.xticks(CFmeanmonthwindDK14.index,['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],weight='bold')
fig, plt.yticks(weight='bold')
fig, plt.ylabel('Monthly avg. CF', fontsize = 15, weight='bold')
fig, plt.legend(loc='upper right',fontsize = 15,ncol=2)
fig.tight_layout()
savefigure('System','MINCFwindsolar','MinCFDK',fig)
#%% Seasonal CF
CFwinter_allwind, CFsummer_allwind, CFseasonalratiowind = getSeasonalCF(CF_onWind, loads)
CFwinter_allsolar, CFsummer_allsolar, CFseasonalratiosolar = getSeasonalCF(CF_solar, loads)
CFseasonalratiowindSystem = (CFseasonalratiowind*loadfactor).sum(axis=1)
CFseasonalratiosolarSystem = (CFseasonalratiosolar*loadfactor).sum(axis=1)

#%% Seasonal CF plot
CFmeanmonthwindES00 = CFmeanmonthwind['ES'].loc['2000-01-31 00:00:00+00:00':'2000-12-31 00:00:00+00:00']
CFmeanmonthwindES00.index = CFtotalmeanmonth_allwind.index
CFmeanmonthsolarES00 = CFmeanmonthsolar['ES'].loc['2000-01-31 00:00:00+00:00':'2000-12-31 00:00:00+00:00']
CFmeanmonthsolarES00.index = CFtotalmeanmonth_allsolar.index
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.plot(CFmeanmonthwindES00,color='dodgerblue',label = '$CF_{wind}$, montly avg. 2000')
ax1.set_xticks(CFmeanmonthwindES00.index)
ax1.set_yticklabels(round(CFmeanmonthwindES00,2),fontweight='bold')
ax1.set_xticklabels(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontweight='bold',fontsize=12)
ax2.plot(CFmeanmonthsolarES00,color='orange',label = '$CF_{solar}$, montly avg. 2000')
ax2.set_xticks(CFmeanmonthsolarES00.index)
ax2.set_yticklabels(round(CFmeanmonthsolarES00,2),fontweight='bold')
ax2.set_xticklabels(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontweight='bold',fontsize=12)
ax1.fill_between(CFmeanmonthwindES00.index.values[[0,1,2,3]],CFmeanmonthwindES00[[0,1,2,3]],color='grey',alpha = 0.3,label='Winter')
ax1.fill_between(CFmeanmonthwindES00.index.values[[9,10,11]],CFmeanmonthwindES00[[9,10,11]],color='grey',alpha = 0.3)
ax1.fill_between(CFmeanmonthwindES00.index.values[[3,4,5,6,7,8,9]],CFmeanmonthwindES00[[3,4,5,6,7,8,9]],color='black',alpha = 0.3,label='Summer')
ax1.legend(loc='best',fontsize=12)
ax2.fill_between(CFmeanmonthsolarES00.index.values[[0,1,2,3]],CFmeanmonthsolarES00[[0,1,2,3]],color='grey',alpha = 0.3,label='Winter')
ax2.fill_between(CFmeanmonthsolarES00.index.values[[9,10,11]],CFmeanmonthsolarES00[[9,10,11]],color='grey',alpha = 0.3)
ax2.fill_between(CFmeanmonthsolarES00.index.values[[3,4,5,6,7,8,9]],CFmeanmonthsolarES00[[3,4,5,6,7,8,9]],color='black',alpha = 0.3,label='Summer')
ax2.legend(loc='best',fontsize=12)
ax1.set_ylabel('CF',fontsize=15,weight='bold')
ax1.set_title('Spain: Wind CF - 2000',fontsize=18,weight='bold')
ax2.set_ylabel('CF',fontsize=15,weight='bold')
ax2.set_title('Spain: Solar CF - 2000',fontsize=18,weight='bold')
fig.tight_layout()
savefigure('System','SeasCFwindsolar','SeasCFES',fig)
#%% hydrofactor (Capacity factor)
hydroCap = storage.filter(like='hydro',axis=0) # Energy capacity
#hydroCap = hydroCap.drop(['BA hydro','RS hydro'],axis=0) 
hydroCap.index = hydroCap.index.str.rstrip(' hydro')
hydroOnes = pd.DataFrame(columns=hydroCap.columns, index=loads.columns)
hydroOnes.fillna(1,inplace=True)
hydroCap = hydroCap/hydroOnes
hydroCap.fillna(0,inplace=True)
hydroinflow_timeY2 = hydroinflow_timeY
#hydroinflow_timeY2 = hydroinflow_timeY2.drop(['BA hydro','RS hydro'],axis = 1)
hydroinflow_timeY2.columns = hydroinflow_timeY2.columns.str.rstrip(' hydro')
Ones = pd.DataFrame(columns=loads_timeY.columns,index=loads_timeY.index)
Ones.fillna(1,inplace=True)
hydroinflow_timeY2 = hydroinflow_timeY2/Ones
hydroinflow_timeY2.fillna(0,inplace=True)
#HydroCF = hydroinflow_timeY2/hydroCap.mean(axis=1)
HydroCF = (hydroinflow_timeY2)/(loads_timeY)

HydroCF.fillna(0,inplace=True)
hydroinflowfactor = meanYear(HydroCF)
hydroinflowfactorSystem = (hydroinflowfactor*loadfactor).sum(axis=1)
hydroinflowfactorSystem.index = CFminfactor_yearwindSystem.index

# hydroinflow_timeYsum = sumYear(hydroinflow_timeY)
# hydroinflow_timeYsum = hydroinflow_timeYsum.drop(['BA hydro','RS hydro'],axis = 1)
# hydroinflow_timeYsum.columns = hydroinflow_timeYsum.columns.str.rstrip(' hydro')
# hydroinflowfactor = hydroinflow_timeYsum/loadssum
# hydroinflowfactor = hydroinflowfactor.fillna(0)
# hydroinflowfactorSystem = (hydroinflowfactor*loadfactor).sum(axis=1)
# hydroinflowfactorSystem.index = CFminfactor_yearwindSystem.index
#%% Hydro plot
Country = 'NO'
date1 = '2013-01-01 00:00:00'
date2 = '2015-12-31 23:00:00'
fig = plt.figure()
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,6))
ax1.plot(hydroinflow_timeY2[Country].loc[date1:date2])
ax2.plot(HydroCF[Country].loc[date1:date2])
ax3.plot(hydroinflowfactor[Country])
ax1.set_ylabel('Inflow [MWh]', fontsize=14)
ax2.set_ylabel('hydro factor', fontsize=14)
ax3.set_ylabel('<hydro factor>', fontsize=14)
ax1.tick_params('both',labelsize=12)
ax2.tick_params('both',labelsize=12)
ax3.tick_params('both',labelsize=12)
fig.suptitle(Country+' Hydro electricity inflow - '+constraint3,fontsize=18,weight='bold')
fig.tight_layout()
savefigure1('System','Hydrofactor',Country,fig)


#%% Histoplot
fig, (ax1,ax2)  = plt.subplots(1,2,figsize=(14,8))
fig, sns.histplot(CF_solaryearloadfactortotal.sum(axis=1),label='Solar CF', color='orange',bins=12,kde=True,ax=ax1)
fig, sns.histplot(CF_onWindyearloadfactortotal.sum(axis=1),label='Wind CF', color='blue',bins=12,kde=True,ax=ax2)
ax1.set_xlabel('Load Weighted <CF> - Solar',fontsize=14,weight='bold')
ax2.set_xlabel('Load Weighted <CF> - Wind',fontsize=14,weight='bold')
ax1.set_title('System Solar CF',fontsize=18,weight='bold')
ax2.set_title('System Wind CF',fontsize=18,weight='bold')
savefigure('System','CFwindsolar','HistoplotSystem',fig)
#%% Multiple linear regression with load weigted capacity factors as input for system

# This is for 37 years of input yearly avg input pr load weighted CF and systemcost
SystemMultdf = pd.DataFrame()
SystemMultdf['CF wind'] = CF_onWindyearloadfactortotal.sum(axis=1)
SystemMultdf['CF solar'] = CF_solaryearloadfactortotal.sum(axis=1)
SystemMultdf['Min CF wind'] = CFminfactor_yearwindSystem
SystemMultdf['Min CF solar'] = CFminfactor_yearsolarSystem
SystemMultdf['Seas. CF wind'] = CFseasonalratiowindSystem
SystemMultdf['Seas. CF solar'] = CFseasonalratiosolarSystem
SystemMultdf['Hydrofactor'] = hydroinflowfactorSystem


#%% Mult lin reg CF wind and Solar 37 data points
# This data will not be scaled, due to easier to interpret the output regression
# Min max scaled data

x = SystemMultdf[['CF wind','CF solar']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = systemcost # System EUR/MWh

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(systemcost).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System37CFwind+CFsolar'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - Load avg. 37 data points - CF wind & solar \n')
file.write('\n'+print_model.as_text())


# Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
ax.scatter(x['CF wind'], x['CF solar'], y,label='data points')
ax.set_xlabel(" Load weighted <CF wind>", fontsize = 12,weight = 'bold')
ax.set_ylabel("Load weighted  <CF solar>",fontsize = 12,weight = 'bold')
ax.set_zlabel("System Cost [EUR/MWh]", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
x_reg = np.linspace(min(x['CF wind']),max(x['CF wind']),1000)
y_reg = np.linspace(min(x['CF solar']),max(x['CF solar']),1000)
x_reg,y_reg = np.meshgrid(x_reg,y_reg)
z_reg = coef['CF wind']*x_reg+coef['CF solar']*y_reg+coef['const']
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar']))
print("\n Normalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar']))
file.close()

my_cmap = plt.get_cmap('viridis_r') 

surf = ax.plot_surface(x_reg,y_reg,z_reg, alpha=0.5,cmap=my_cmap,label='Regression')
cbar = fig.colorbar(surf,ax = ax, shrink=0.2, aspect = 10)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('System Cost [EUR/MWh]',size=15,weight='bold')
surf.set_clim(50,65)
ax.view_init(20, -45)
#ax.view_init(90, -90)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.95)
fig.tight_layout()
savefigure1('System',constraint3,'MULTREG3DCFwindCFsolar37',fig)
#%% Mult lin reg CF wind and Solar, hydro 37 data points
# This data will not be scaled, due to easier to interpret the output regression


x = SystemMultdf[['CF wind','CF solar','Hydrofactor']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression

y = systemcost # System cost EUR/MWh

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(systemcost).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System37CFwind+CFsolar+hydro'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - Load avg. 37 data points - CF wind & solar & hydro \n')
file.write('\n'+print_model.as_text())


# # Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
img = ax.scatter(x['CF wind'], x['CF solar'], x['Hydrofactor'],c=y,cmap=plt.get_cmap('viridis_r'),label='data points',s=70)
ax.set_xlabel(" Load weighted <CF wind>", fontsize = 12,weight = 'bold')
ax.set_ylabel("Load weighted  <CF solar>",fontsize = 12,weight = 'bold')
ax.set_zlabel("<Hydrofactor>", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
cbar = fig.colorbar(img,ax = ax, shrink=0.2, aspect = 15)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('System Cost [EUR/MWh]',size=15,weight='bold')
img.set_clim(50,65)
ax.set_title('Multilinear regression '+constraint3, fontsize = 20, weight='bold',y=0.95)
ax.view_init(20, -45)
fig.tight_layout()

print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Hydrofactor']))
print("\nNormalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3),'Hydro factor:',round(NormSE[3],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}, Hydro factor: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2],NormSE[3]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Hydrofactor']))
file.close()
savefigure1('System',constraint3,'MULTREG3DCFwindCFsolarHydro37',fig)
# savefigure('System','2XTransmission','MULTREG3D',fig)
#%% Mult lin reg Minimum CF wind and Solar 37 data points
# This data will not be scaled, due to easier to interpret the output regression
# Min max scaled data

x = SystemMultdf[['Min CF wind','Min CF solar']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = systemcost # System cost Million Euros

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(systemcost).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System37MINCFwind+CFsolar'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - Load avg. 37 data points - min CF wind & solar \n')
file.write('\n'+print_model.as_text())


# Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
ax.scatter(x['Min CF wind'], x['Min CF solar'], y,label='data points')
ax.set_xlabel(" Load weighted Min CF wind.", fontsize = 12,weight = 'bold')
ax.set_ylabel("Load weighted  Min CF solar.",fontsize = 12,weight = 'bold')
ax.set_zlabel("System Cost [EUR/MWh]", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
x_reg = np.linspace(min(x['Min CF wind']),max(x['Min CF wind']),1000)
y_reg = np.linspace(min(x['Min CF solar']),max(x['Min CF solar']),1000)
x_reg,y_reg = np.meshgrid(x_reg,y_reg)
z_reg = coef['Min CF wind']*x_reg+coef['Min CF solar']*y_reg+coef['const']
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Min CF wind'],
                                                          coef['Min CF solar']))
print("\n Normalized standard errors: Constant: ",round(NormSE[0],3),'Min CF wind:',round(NormSE[1],3),'Min CF solar:',round(NormSE[2],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, Min CF wind: {:.3f}, Min CF solar: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Min CF wind'],
                                                          coef['Min CF solar']))
file.close()

my_cmap = plt.get_cmap('viridis_r') 

surf = ax.plot_surface(x_reg,y_reg,z_reg, alpha=0.5,cmap=my_cmap,label='Regression')
cbar = fig.colorbar(surf,ax = ax, shrink=0.2, aspect = 10)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('System Cost [EUR/MWh]',size=15,weight='bold')
surf.set_clim(50,65)
ax.view_init(20, -45)
#ax.view_init(90, -90)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.95)
fig.tight_layout()
savefigure1('System',constraint3,'MULTREG3DMINCFwindCFsolar37',fig)
#savefigure('System','2XTransmission','MULTREG3DCFwindCFsolar',fig)
#%% Mult lin reg seas CF wind and Solar 37 data points
# This data will not be scaled, due to easier to interpret the output regression
# Min max scaled data

x = SystemMultdf[['Seas. CF wind','Seas. CF solar']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = systemcost # System cost eur/mwh

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(systemcost).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System37SeasonalCFwind+CFsolar'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - Load avg. 37 data points - seasonal CF \n')
file.write('\n'+print_model.as_text())


# Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
ax.scatter(x['Seas. CF wind'], x['Seas. CF solar'], y,label='data points')
ax.set_xlabel(" Load weighted Seas. CF wind ratio.", fontsize = 12,weight = 'bold')
ax.set_ylabel("Load weighted  Seas. CF solar ratio.",fontsize = 12,weight = 'bold')
ax.set_zlabel("System Cost [EUR/MWh]", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
x_reg = np.linspace(min(x['Seas. CF wind']),max(x['Seas. CF wind']),1000)
y_reg = np.linspace(min(x['Seas. CF solar']),max(x['Seas. CF solar']),1000)
x_reg,y_reg = np.meshgrid(x_reg,y_reg)
z_reg = coef['Seas. CF wind']*x_reg+coef['Seas. CF solar']*y_reg+coef['const']
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Seas. CF wind'],
                                                          coef['Seas. CF solar']))
print("\n Normalized standard errors: Constant: ",round(NormSE[0],3),'Seas. CF wind:',round(NormSE[1],3),'Seas. CF solar:',round(NormSE[2],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, Seasonal CF wind ratio: {:.3f}, Seasonal CF solar ratio: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Seas. CF wind'],
                                                          coef['Seas. CF solar']))
file.close()

my_cmap = plt.get_cmap('viridis_r') 

surf = ax.plot_surface(x_reg,y_reg,z_reg, alpha=0.5,cmap=my_cmap,label='Regression')
cbar = fig.colorbar(surf,ax = ax, shrink=0.2, aspect = 10)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('System Cost [EUR/MWh]',size=15,weight='bold')
surf.set_clim(50,65)
ax.view_init(20, -45)
#ax.view_init(90, -90)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.95)
fig.tight_layout()
savefigure1('System',constraint3,'MULTREG3DSeas.CFwindCFsolar37',fig)

#savefigure('System','2XTransmission','MULTREG3DCFwindCFsolar',fig)
#%% Mult lin reg all data 37 data points
# This data will not be scaled, due to easier to interpret the output regression


x = SystemMultdf

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = systemcost # System cost EUR/MWh

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(systemcost).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System37all'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - Load avg. 37 data points - all \n')
file.write('\n'+print_model.as_text())



print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6 + {:.2f}x7".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Min CF wind'],coef['Min CF solar'],coef['Seas. CF wind'],coef['Seas. CF solar'],coef['Hydrofactor']))
print("\nNormalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3), 'Min CF wind:',round(NormSE[3],3),'Min CF solar:',round(NormSE[4],3),'Seas. CF wind:',round(NormSE[5],3),'Seas. CF solar:',round(NormSE[6],3), 'Hydro factor:',round(NormSE[7],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}, Min CF wind: {:.3f}, Min CF solar: {:.3f}, Seas. CF wind: {:.3f}, Seas. CF solar: {:.3f}, Hydro factor: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2],NormSE[3],NormSE[4],NormSE[5],NormSE[6],NormSE[7]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6 + {:.2f}x7".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'], coef['Min CF wind'], coef['Min CF solar'], coef['Seas. CF wind'], coef['Seas. CF solar'],coef['Hydrofactor']))
file.close()
#%% Pearson correlation matrix load weighted
corrSystemMultdf  = SystemMultdf.corr()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
fig = plt.figure(figsize=(12,8))
sns.set(font_scale=1.5)
sns.heatmap(corrSystemMultdf, annot=True,cmap=cmap,fmt='.2f',vmin=-1,vmax=1)
plt.xticks(fontsize=15,rotation=45,weight='bold')
plt.yticks(fontsize=15,rotation=45,weight='bold')
plt.title('Pearson correlation matrix: Load weighted '+constraint3,color = 'Black',fontsize=20,weight='bold')
fig.tight_layout()
savefigure1('System',constraint3,'Pearsonplot37',fig)
plt.rcdefaults()
#%% Linear regression for the 7 variables 37 datapoints
fig, ax = plt.subplots(3, 3, figsize=(8, 8))
fig.delaxes(ax[2,1]) # Delete axis
fig.delaxes(ax[2,2]) # Delete axis
x1 = SystemMultdf['CF wind']
x2 = SystemMultdf['CF solar']
x3 = SystemMultdf['Min CF wind']
x4 = SystemMultdf['Min CF solar']
x5 = SystemMultdf['Seas. CF wind']
x6 = SystemMultdf['Seas. CF solar']
x7 = SystemMultdf['Hydrofactor']
y = systemcost #EUR/MWh

# 1st plot CF wind
ax[0,0].scatter(x1,y, label='CF wind', alpha=0.5, color='dodgerblue')
ax[0,0].set_xlabel('<CF wind>',fontsize = 12, weight='bold')
ax[0,0].set_ylabel('System Cost [EUR/MWh]',fontsize = 12, weight='bold')
ax[0,0].tick_params(axis='both', labelsize=11)
X1 = sm.add_constant(x1)
model1 = sm.OLS(list(y),X1).fit()
predictions1 = model1.predict(X1) 
print_model1 = model1.summary()
coef1 = model1.params
p_value1 = model1.summary2().tables[1]['P>|t|'].loc['CF wind']
R21 = model1.rsquared
NormSE1 = model1.bse/(systemcost).mean()
ax[0,0].plot(x1,predictions1,color='darkblue',label='Reg.')
ax[0,0].legend()
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R21, ),
     r'$P=%.3f$' % (p_value1, ),
     r'$\bar{SE}=%.3f$' % (NormSE1[1], )))
ax[0,0].text(0.6, 0.95, textstr, transform=ax[0,0].transAxes, fontsize=11,
         verticalalignment='top')

# 2nd plot CF min wind
ax[0,1].scatter(x3,y, label='Min wind', alpha=0.5, color='dodgerblue')
ax[0,1].set_xlabel('Min. CF wind',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[0,1].tick_params(axis='x', labelsize=11)
ax[0,1].tick_params(axis='y', labelleft=False)
X3 = sm.add_constant(x3)
model3 = sm.OLS(list(y),X3).fit()
predictions3 = model3.predict(X3) 
print_model3 = model3.summary()
coef3 = model3.params
p_value3 = model3.summary2().tables[1]['P>|t|'].loc['Min CF wind']
R23 = model3.rsquared
NormSE3 = model3.bse/(systemcost).mean()
ax[0,1].plot(x3,predictions3,color='darkblue',label='Reg.')
ax[0,1].legend()
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R23, ),
     r'$P=%.3f$' % (p_value3, ),
     r'$\bar{SE}=%.3f$' % (NormSE3[1], )))
ax[0,1].text(0.05, 0.95, textstr, transform=ax[0,1].transAxes, fontsize=11,
         verticalalignment='top')

# 3rd plot CF seas wind
ax[0,2].scatter(x5,y, label='Seas. wind', alpha=0.5, color='dodgerblue')
ax[0,2].set_xlabel('Seas. CF wind',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[0,2].tick_params(axis='x', labelsize=11)
ax[0,2].tick_params(axis='y', labelleft=False)
X5 = sm.add_constant(x5)
model5 = sm.OLS(list(y),X5).fit()
predictions5 = model5.predict(X5) 
print_model5 = model5.summary()
coef5 = model5.params
p_value5 = model5.summary2().tables[1]['P>|t|'].loc['Seas. CF wind']
R25 = model5.rsquared
NormSE5 = model5.bse/(systemcost).mean()
ax[0,2].plot(x5,predictions5,color='darkblue',label='Reg.')
ax[0,2].legend(loc='lower right')
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R25, ),
      r'$P=%.3f$' % (p_value5, ),
      r'$\bar{SE}=%.3f$' % (NormSE5[1], )))
ax[0,2].text(0.6, 0.95, textstr, transform=ax[0,2].transAxes, fontsize=11,
          verticalalignment='top')

# 4th plot CF solar
ax[1,0].scatter(x2,y, label='CF solar', alpha=0.5, color='orange')
ax[1,0].set_xlabel('<CF solar>',fontsize = 12, weight='bold')
ax[1,0].set_ylabel('System Cost [EUR/MWh]',fontsize = 12, weight='bold')
ax[1,0].tick_params(axis='both', labelsize=11)
X2 = sm.add_constant(x2)
model2 = sm.OLS(list(y),X2).fit()
predictions2 = model2.predict(X2) 
print_model2 = model2.summary()
coef2 = model2.params
p_value2 = model2.summary2().tables[1]['P>|t|'].loc['CF solar']
R22 = model2.rsquared
NormSE2 = model2.bse/(systemcost).mean()
ax[1,0].plot(x2,predictions2,color='darkorange',label='Reg.')
ax[1,0].legend(loc='upper right')
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R22, ),
     r'$P=%.3f$' % (p_value2, ),
     r'$\bar{SE}=%.3f$' % (NormSE2[1], )))
ax[1,0].text(0.6, 0.35, textstr, transform=ax[1,0].transAxes, fontsize=11,
         verticalalignment='top')

# 5th plot CF min solar
ax[1,1].scatter(x4,y, label='Min solar', alpha=0.5, color='orange')
ax[1,1].set_xlabel('Min. CF solar',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[1,1].tick_params(axis='x', labelsize=11)
ax[1,1].tick_params(axis='y', labelleft=False)
X4 = sm.add_constant(x4)
model4 = sm.OLS(list(y),X4).fit()
predictions4 = model4.predict(X4) 
print_model4 = model4.summary()
coef4 = model4.params
p_value4 = model4.summary2().tables[1]['P>|t|'].loc['Min CF solar']
R24 = model4.rsquared
NormSE4 = model4.bse/(systemcost).mean()
ax[1,1].plot(x4,predictions4,color='darkorange',label='Reg.')
ax[1,1].legend(loc='lower left')
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R24, ),
      r'$P=%.3f$' % (p_value4, ),
      r'$\bar{SE}=%.3f$' % (NormSE4[1], )))
ax[1,1].text(0.05, 0.95, textstr, transform=ax[1,1].transAxes, fontsize=11,
          verticalalignment='top')

# 6th plot CF seas solar
ax[1,2].scatter(x6,y, label='Seas. solar', alpha=0.5, color='orange')
ax[1,2].set_xlabel('Seas. CF solar',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[1,2].tick_params(axis='x', labelsize=11)
ax[1,2].tick_params(axis='y', labelleft=False)
X6 = sm.add_constant(x6)
model6 = sm.OLS(list(y),X6).fit()
predictions6 = model6.predict(X6) 
print_model6 = model6.summary()
coef6 = model6.params
p_value6 = model6.summary2().tables[1]['P>|t|'].loc['Seas. CF solar']
R26 = model6.rsquared
NormSE6 = model6.bse/(systemcost).mean()
ax[1,2].plot(x6,predictions6,color='darkorange',label='Reg.')
ax[1,2].legend(loc='lower left')
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R26, ),
      r'$P=%.3f$' % (p_value6, ),
      r'$\bar{SE}=%.3f$' % (NormSE6[1], )))
ax[1,2].text(0.05, 0.95, textstr, transform=ax[1,2].transAxes, fontsize=11,
          verticalalignment='top')

# 7th plot hydrofactor
ax[2,0].scatter(x7,y, label='Hydrofactor', alpha=0.5, color='grey')
ax[2,0].set_xlabel('Hydrofactor',fontsize = 12, weight='bold')
ax[2,0].set_ylabel('System Cost [EUR/MWh]',fontsize = 12, weight='bold')
ax[2,0].tick_params(axis='both', labelsize=11)
X7 = sm.add_constant(x7,has_constant='add')
model7 = sm.OLS(list(y),X7).fit()
predictions7 = model7.predict(X7) 
print_model7 = model7.summary()
coef7 = model7.params
p_value7 = model7.summary2().tables[1]['P>|t|'].loc['Hydrofactor']
R27 = model7.rsquared
NormSE7 = model7.bse/(systemcost).mean()
ax[2,0].plot(x7,predictions7,color='darkgrey',label='Reg.')
ax[2,0].legend(loc='upper left')
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R27, ),
     r'$P=%.3f$' % (p_value7, ),
     r'$\bar{SE}=%.3f$' % (NormSE7[1], )))
ax[2,0].text(0.05, 0.35, textstr, transform=ax[2,0].transAxes, fontsize=11,
         verticalalignment='top')
fig.suptitle('Linear Regression: '+constraint3, fontsize=18,weight='bold')
fig.tight_layout()
savefigure1('System',constraint3,'Linreg37',fig)


#%%
# This is for all the countries yearly averaged data for all the years
SystemMultdf2 = pd.DataFrame()
MeltedCFwind = pd.melt(CF_onWindyear)
MeltedCFwindmin = pd.melt(CFminfactor_yearwind)
MeltedCFsolar = pd.melt(CF_solaryear)
MeltedCFsolarmin = pd.melt(CFminfactor_yearsolar)
MeltedCFhydro = pd.melt(hydroinflowfactor)
MeltedCFwindseas = pd.melt(CFseasonalratiowind)
MeltedCFsolarseas = pd.melt(CFseasonalratiosolar)
SystemMultdf2 = pd.concat([MeltedCFwind,MeltedCFsolar,MeltedCFwindmin,MeltedCFsolarmin,MeltedCFwindseas,MeltedCFsolarseas,MeltedCFhydro],axis=1).drop('variable',1)
SystemMultdf2.columns = ['CF wind', 'CF solar', 'Min CF wind', 'Min CF solar','Seas. CF wind','Seas. CF solar','Hydrofactor']
#%% 
# Calculating Average electricity price NOT load weigted

eprice2 = eprice_time
eprices_names = list(eprice2.columns)
e_price_country_names = eprices_names[0:30]
e_price_country = eprice2[e_price_country_names]
e_price_countryY = sumYear(e_price_country*loads_timeY)/sumYear(loads_timeY) # .drop(['BA','RS'],axis = 1)

Melted_eprice = pd.melt(e_price_countryY)
Sorted_Melted_eprice = Melted_eprice.sort_values(by='value')
#%% Colorcoding

colors = {'AT':'red', 'BA':'yellow', 'BE':'black','BG':'green','CH':'lightcoral','CZ':'dodgerblue',
          'DE':'darkgoldenrod','DK':'indianred','EE':'grey','ES':'darkred','FI':'lightblue','FR':'darkblue',
          'GB':'cyan','GR':'lightgrey','HR':'deepskyblue','HU':'darkgreen','IE':'lightgreen','IT':'seagreen',
          'LT':'orange','LU':'pink','LV':'brown','NL':'darkorange','NO':'violet','PL':'crimson',
          'PT':'firebrick','RO':'navajowhite','RS':'slategray','SE':'gold','SI':'chocolate','SK':'peachpuff'}


#%% Mult lin reg CF wind and Solar 1110 data points
# This data will not be scaled, due to easier to interpret the output regression
# Min max scaled data

x = SystemMultdf2[['CF wind','CF solar']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] # electricity price Euro/MWh

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036CFwind+CFsolar'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - CF wind & solar \n')
file.write('\n'+print_model.as_text())


# Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data 
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax.scatter(CF_onWindyear[Name], CF_solaryear[Name], e_price_countryY[Name],label=Name,c=c)
ax.legend(bbox_to_anchor=(0.05, 0.75),ncol=2)    
ax.set_xlabel("<CF wind>", fontsize = 12,weight = 'bold')
ax.set_ylabel("<CF solar>",fontsize = 12,weight = 'bold')
ax.set_zlabel("<Electricity Price> [EUR/MWh]", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
x_reg = np.linspace(min(x['CF wind']),max(x['CF wind']),1000)
y_reg = np.linspace(min(x['CF solar']),max(x['CF solar']),1000)
x_reg,y_reg = np.meshgrid(x_reg,y_reg)
z_reg = coef['CF wind']*x_reg+coef['CF solar']*y_reg+coef['const']
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar']))
print("\n Normalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar']))
file.close()

my_cmap = plt.get_cmap('viridis_r') 

surf = ax.plot_surface(x_reg,y_reg,z_reg, alpha=0.3,cmap=my_cmap,label='Regression')
cbar = fig.colorbar(surf,ax = ax, shrink=0.2, aspect = 10)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('<Electricity price> [EUR/MWh]',size=15,weight='bold')
surf.set_clim(5,140)
ax.view_init(5, -45)
#ax.view_init(90, -90)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.9)
fig.tight_layout()
savefigure1('System',constraint3,'MULTREG3DCFwindCFsolar1036',fig)
#%% Mult lin reg CF wind and Solar, hydro 1110 data points
# This data will not be scaled, due to easier to interpret the output regression


x = SystemMultdf2[['CF wind','CF solar','Hydrofactor']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] # electricity price Euro/MWh

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036CFwind+CFsolar+hydro'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - CF wind & solar & hydro \n')
file.write('\n'+print_model.as_text())


# # Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
img = ax.scatter(x['CF wind'], x['CF solar'], x['Hydrofactor'],c=y,cmap=plt.get_cmap('viridis_r'),label='data points',s=70)
ax.set_xlabel("<CF wind>", fontsize = 12,weight = 'bold')
ax.set_ylabel("<CF solar>",fontsize = 12,weight = 'bold')
ax.set_zlabel("<Hydrofactor>", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
cbar = fig.colorbar(img,ax = ax, shrink=0.2, aspect = 15)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('<Electricity price> [EUR/MWh]',size=15,weight='bold')
img.set_clim(5,140)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.95)
ax.view_init(20, -45)
fig.tight_layout()

print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Hydrofactor']))
print("\nNormalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3),'Hydro factor:',round(NormSE[3],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}, Hydro factor: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2],NormSE[3]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Hydrofactor']))
file.close()
savefigure1('System',constraint3,'MULTREG3DCFwindCFsolarhydro1036',fig)
# savefigure('System','2XTransmission','MULTREG3D1036',fig)
#%% Mult lin reg Minimum CF wind and Solar 1110 data points
# This data will not be scaled, due to easier to interpret the output regression
# Min max scaled data

x = SystemMultdf2[['Min CF wind','Min CF solar']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] # Eprice [EUR/MWh]

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036MINCFwind+CFsolar'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - min CF wind & solar \n')
file.write('\n'+print_model.as_text())


# Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax.scatter(CFminfactor_yearwind[Name], CFminfactor_yearsolar[Name], e_price_countryY[Name],label=Name,c=c)

ax.legend(bbox_to_anchor=(0.05, 0.75),ncol=2)   
#ax.scatter(x['Min CF wind'], x['Min CF solar'], y,label='data points')
ax.set_xlabel("Min CF wind", fontsize = 12,weight = 'bold')
ax.set_ylabel("Min CF solar",fontsize = 12,weight = 'bold')
ax.set_zlabel("<Electricity price> [EUR/MWh]", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
x_reg = np.linspace(min(x['Min CF wind']),max(x['Min CF wind']),1000)
y_reg = np.linspace(min(x['Min CF solar']),max(x['Min CF solar']),1000)
x_reg,y_reg = np.meshgrid(x_reg,y_reg)
z_reg = coef['Min CF wind']*x_reg+coef['Min CF solar']*y_reg+coef['const']
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Min CF wind'],
                                                          coef['Min CF solar']))
print("\n Normalized standard errors: Constant: ",round(NormSE[0],3),'Min CF wind:',round(NormSE[1],3),'Min CF solar:',round(NormSE[2],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, Min CF wind: {:.3f}, Min CF solar: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Min CF wind'],
                                                          coef['Min CF solar']))
file.close()

my_cmap = plt.get_cmap('viridis_r') 

surf = ax.plot_surface(x_reg,y_reg,z_reg, alpha=0.5,cmap=my_cmap,label='Regression')
cbar = fig.colorbar(surf,ax = ax, shrink=0.2, aspect = 10)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('<Electricity price> [EUR/MWh]',size=15,weight='bold')
surf.set_clim(5,140)
ax.view_init(20, -45)
#ax.view_init(90, -90)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.95)
fig.tight_layout()
savefigure1('System',constraint3,'MULTREG3DMINCFwindCFsolar1036',fig)
#savefigure('System','2XTransmission','MULTREG3DCFwindCFsolar1036',fig)
#%% Mult lin reg seas CF wind and Solar 1110 data points
# This data will not be scaled, due to easier to interpret the output regression
# Min max scaled data

x = SystemMultdf2[['Seas. CF wind','Seas. CF solar']]

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] # Eprice [EUR/MWh]

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036SeasonalCFwind+CFsolar'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - seasonal CF \n')
file.write('\n'+print_model.as_text())


# Create the plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Add the data points
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax.scatter(CFseasonalratiowind[Name], CFseasonalratiosolar[Name], e_price_countryY[Name],label=Name,c=c)

ax.legend(bbox_to_anchor=(0.05, 0.75),ncol=2)   
#ax.scatter(x['Seas. CF wind'], x['Seas. CF solar'], y,label='data points')
ax.set_xlabel("Seas. CF wind ratio.", fontsize = 12,weight = 'bold')
ax.set_ylabel("Seas. CF solar ratio.",fontsize = 12,weight = 'bold')
ax.set_zlabel("<Electricity price> [EUR/MWh]", fontsize = 12,weight = 'bold',labelpad=-3)
ax.tick_params(axis="z",direction="in", pad=-22,labelsize=12)
ax.tick_params(axis="y",direction="out", pad=0,labelsize=12)
ax.tick_params(axis="x",direction="out", pad=0,labelsize=12)
x_reg = np.linspace(min(x['Seas. CF wind']),max(x['Seas. CF wind']),1000)
y_reg = np.linspace(min(x['Seas. CF solar']),max(x['Seas. CF solar']),1000)
x_reg,y_reg = np.meshgrid(x_reg,y_reg)
z_reg = coef['Seas. CF wind']*x_reg+coef['Seas. CF solar']*y_reg+coef['const']
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Seas. CF wind'],
                                                          coef['Seas. CF solar']))
print("\n Normalized standard errors: Constant: ",round(NormSE[0],3),'Seas. CF wind:',round(NormSE[1],3),'Seas. CF solar:',round(NormSE[2],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, Seasonal CF wind ratio: {:.3f}, Seasonal CF solar ratio: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(coef['const'], coef['Seas. CF wind'],
                                                          coef['Seas. CF solar']))
file.close()

my_cmap = plt.get_cmap('viridis_r') 

surf = ax.plot_surface(x_reg,y_reg,z_reg, alpha=0.5,cmap=my_cmap,label='Regression')
cbar = fig.colorbar(surf,ax = ax, shrink=0.2, aspect = 10)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('<Electricity price> [EUR/MWh]',size=15,weight='bold')
surf.set_clim(5,140)
ax.view_init(5, -45)
#ax.view_init(90, -90)
ax.set_title('Multilinear regression - '+constraint3, fontsize = 20, weight='bold',y=0.95)
fig.tight_layout()
savefigure1('System',constraint3,'MULTREG3DSeasCFwindCFsolar1036',fig)
#savefigure('System','2XTransmission','MULTREG3DCFwindCFsolar1036',fig)
#%% Mult lin reg all data 1110 data points
# This data will not be scaled, due to easier to interpret the output regression


x = SystemMultdf2

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] #  Eprice [EUR/MWh]

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036all'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - all \n')
file.write('\n'+print_model.as_text())



print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6 + {:.2f}x7".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Min CF wind'],coef['Min CF solar'],coef['Seas. CF wind'],coef['Seas. CF solar'],coef['Hydrofactor']))
print("\nNormalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3), 'Min CF wind:',round(NormSE[3],3),'Min CF solar:',round(NormSE[4],3),'Seas. CF wind:',round(NormSE[5],3),'Seas. CF solar:',round(NormSE[6],3), 'Hydro factor:',round(NormSE[7],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}, Min CF wind: {:.3f}, Min CF solar: {:.3f}, Seas. CF wind: {:.3f}, Seas. CF solar: {:.3f}, Hydro factor: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2],NormSE[3],NormSE[4],NormSE[5],NormSE[6],NormSE[7]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6 + {:.2f}x7".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'], coef['Min CF wind'], coef['Min CF solar'], coef['Seas. CF wind'], coef['Seas. CF solar'],coef['Hydrofactor']))
file.close()
#%% Without  Min CF Wind
x = SystemMultdf2.drop(['Min CF wind'],axis=1)

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] #  Eprice [EUR/MWh]

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036all-minCFwind'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - Not CF wind and  \n')
file.write('\n'+print_model.as_text())



print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6".format(coef['const'],coef['CF wind'],
                                                          coef['CF solar'],coef['Min CF solar'],coef['Seas. CF wind'],coef['Seas. CF solar'],coef['Hydrofactor']))
print("\nNormalized standard errors: Constant: ",round(NormSE[0],3),'CF wind',round(NormSE[1],3),'CF solar:',round(NormSE[2],3),'Min CF solar:',round(NormSE[3],3),'Seas. CF wind:',round(NormSE[4],3),'Seas. CF solar:',round(NormSE[5],3), 'Hydro factor:',round(NormSE[6],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}, Min CF solar: {:.3f}, Seas. CF wind: {:.3f}, Seas. CF solar: {:.3f}, Hydro factor: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2],NormSE[3],NormSE[4],NormSE[5],NormSE[6]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'], coef['Min CF solar'], coef['Seas. CF wind'], coef['Seas. CF solar'],coef['Hydrofactor']))
file.close()
#%% Without hydro factor
x = SystemMultdf2.drop(['Hydrofactor'],axis=1)

x = sm.add_constant(x,has_constant='add') # Adding constant to the regression
y = Melted_eprice['value'] #  Eprice [EUR/MWh]

model = sm.OLS(list(y), x).fit()
predictions = model.predict(x) 
print_model = model.summary()
coef = model.params
NormSE = model.bse/(Melted_eprice['value']).mean()

print(print_model)
save_path = r"D:\Master\MULTREGLOGS"
FORMAT = '%Y-%m-%d-%H-%M-%S'
name_of_file = 'System1036all-hydro'+constraint2
completeName = os.path.join(save_path, name_of_file+".txt") 
# Open file        
file = open(completeName, "w")
# Time stamp
file.write(datetime.now().strftime(FORMAT))
# Write to file
file.write('\nData: Mult. linReg - System - 1036 data points - Not Hydro  \n')
file.write('\n'+print_model.as_text())

print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'],coef['Min CF wind'],coef['Min CF solar'],coef['Seas. CF wind'],coef['Seas. CF solar']))
print("\nNormalized standard errors: Constant: ",round(NormSE[0],3),'CF wind:',round(NormSE[1],3),'CF solar:',round(NormSE[2],3), 'Min CF wind:',round(NormSE[3],3),'Min CF solar:',round(NormSE[4],3),'Seas. CF wind:',round(NormSE[5],3),'Seas. CF solar:',round(NormSE[6],3))
file.write("\n Normalized standard errors: Constant: {:.3f}, CF wind: {:.3f}, CF solar: {:.3f}, Min CF wind: {:.3f}, Min CF solar: {:.3f}, Seas. CF wind: {:.3f}, Seas. CF solar: {:.3f}".format(NormSE[0],NormSE[1],NormSE[2],NormSE[3],NormSE[4],NormSE[5],NormSE[6]))
file.write("\n Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2 + {:.2f}x3 + {:.2f}x4 + {:.2f}x5 + {:.2f}x6".format(coef['const'], coef['CF wind'],
                                                          coef['CF solar'], coef['Min CF wind'], coef['Min CF solar'], coef['Seas. CF wind'], coef['Seas. CF solar']))
file.close()

#%% Pearson correlation matrix 1110
corrSystemMultdf2  = SystemMultdf2.corr()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
fig = plt.figure(figsize=(12,8))
sns.set(font_scale=1.5)
sns.heatmap(corrSystemMultdf2, annot=True,cmap=cmap,fmt='.2f',vmin=-1,vmax=1)
plt.xticks(fontsize=15,rotation=45,weight='bold')
plt.yticks(fontsize=15,rotation=45,weight='bold')
plt.title('Pearson correlation matrix: '+constraint3,color = 'Black',fontsize=20,weight='bold')
fig.tight_layout()
savefigure1('System',constraint3,'Pearsonplot1036',fig)
plt.rcdefaults()


#%% Linear regression for the 7 variables 1036 datapoints
fig, ax = plt.subplots(3, 3, figsize=(8, 8))
fig.delaxes(ax[2,1]) # Delete axis
fig.delaxes(ax[2,2]) # Delete axis
x1 = SystemMultdf2['CF wind']
x2 = SystemMultdf2['CF solar']
x3 = SystemMultdf2['Min CF wind']
x4 = SystemMultdf2['Min CF solar']
x5 = SystemMultdf2['Seas. CF wind']
x6 = SystemMultdf2['Seas. CF solar']
x7 = SystemMultdf2['Hydrofactor']
y = Melted_eprice['value'] #EUR/MWh

# 1st plot CF wind
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[0,0].scatter(CF_onWindyear[Name], e_price_countryY[Name],label=Name,c=c,alpha=0.5)

#ax[0,0].legend(bbox_to_anchor=(0.05, 0.75),ncol=2) 
# ax[0,0].scatter(x1,y, label='CF wind', alpha=0.5, color='dodgerblue')
ax[0,0].set_xlabel('<CF wind>',fontsize = 12, weight='bold')
ax[0,0].set_ylabel('<Elec. price> [EUR/MWh]',fontsize = 12, weight='bold')
ax[0,0].tick_params(axis='both', labelsize=11)
X1 = sm.add_constant(x1)
model1 = sm.OLS(list(y),X1).fit()
predictions1 = model1.predict(X1) 
print_model1 = model1.summary()
coef1 = model1.params
p_value1 = model1.summary2().tables[1]['P>|t|'].loc['CF wind']
R21 = model1.rsquared
NormSE1 = model1.bse/(y).mean()
ax[0,0].plot(x1,predictions1,color='darkblue',label='Reg.')
#ax[0,0].legend()
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R21, ),
     r'$P=%.3f$' % (p_value1, ),
     r'$\bar{SE}=%.3f$' % (NormSE1[1], )))
ax[0,0].text(0.05, 0.35, textstr, transform=ax[0,0].transAxes, fontsize=11,
         verticalalignment='top')

# 2nd plot CF min wind
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[0,1].scatter(CFminfactor_yearwind[Name], e_price_countryY[Name],c=c,alpha=0.5)
#ax[0,1].scatter(x3,y, label='Min wind', alpha=0.5, color='dodgerblue')
ax[0,1].set_xlabel('Min. CF wind',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[0,1].tick_params(axis='x', labelsize=11)
ax[0,1].tick_params(axis='y', labelleft=False)
X3 = sm.add_constant(x3)
model3 = sm.OLS(list(y),X3).fit()
predictions3 = model3.predict(X3) 
print_model3 = model3.summary()
coef3 = model3.params
p_value3 = model3.summary2().tables[1]['P>|t|'].loc['Min CF wind']
R23 = model3.rsquared
NormSE3 = model3.bse/(y).mean()
ax[0,1].plot(x3,predictions3,color='darkblue')
#ax[0,1].legend()
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R23, ),
     r'$P=%.3f$' % (p_value3, ),
     r'$\bar{SE}=%.3f$' % (NormSE3[1], )))
ax[0,1].text(0.6, 0.35, textstr, transform=ax[0,1].transAxes, fontsize=11,
         verticalalignment='top')

# 3rd plot CF seas wind
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[0,2].scatter(CFseasonalratiowind[Name], e_price_countryY[Name],c=c,alpha=0.5)
#ax[0,2].scatter(x5,y, label='Seas. wind', alpha=0.5, color='dodgerblue')
ax[0,2].set_xlabel('Seas. CF wind',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[0,2].tick_params(axis='x', labelsize=11)
ax[0,2].tick_params(axis='y', labelleft=False)
X5 = sm.add_constant(x5)
model5 = sm.OLS(list(y),X5).fit()
predictions5 = model5.predict(X5) 
print_model5 = model5.summary()
coef5 = model5.params
p_value5 = model5.summary2().tables[1]['P>|t|'].loc['Seas. CF wind']
R25 = model5.rsquared
NormSE5 = model5.bse/(y).mean()
ax[0,2].plot(x5,predictions5,color='darkblue')
#ax[0,2].legend(loc='lower right')
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R25, ),
      r'$P=%.3f$' % (p_value5, ),
      r'$\bar{SE}=%.3f$' % (NormSE5[1], )))
ax[0,2].text(0.6, 0.35, textstr, transform=ax[0,2].transAxes, fontsize=11,
          verticalalignment='top')

# 4th plot CF solar
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[1,0].scatter(CF_solaryear[Name], e_price_countryY[Name],c=c,alpha=0.5)
#ax[1,0].scatter(x2,y, label='CF solar', alpha=0.5, color='orange')
ax[1,0].set_xlabel('<CF solar>',fontsize = 12, weight='bold')
ax[1,0].set_ylabel('<Elec. price> [EUR/MWh]',fontsize = 12, weight='bold')
ax[1,0].tick_params(axis='both', labelsize=11)
X2 = sm.add_constant(x2)
model2 = sm.OLS(list(y),X2).fit()
predictions2 = model2.predict(X2) 
print_model2 = model2.summary()
coef2 = model2.params
p_value2 = model2.summary2().tables[1]['P>|t|'].loc['CF solar']
R22 = model2.rsquared
NormSE2 = model2.bse/(y).mean()
ax[1,0].plot(x2,predictions2,color='darkblue')
#ax[1,0].legend(loc='upper right')
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R22, ),
     r'$P=%.3f$' % (p_value2, ),
     r'$\bar{SE}=%.3f$' % (NormSE2[1], )))
ax[1,0].text(0.6, 0.35, textstr, transform=ax[1,0].transAxes, fontsize=11,
         verticalalignment='top')

# 5th plot CF min solar
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[1,1].scatter(CFminfactor_yearsolar[Name], e_price_countryY[Name],c=c,alpha=0.5)
#ax[1,1].scatter(x4,y, label='Min solar', alpha=0.5, color='orange')
ax[1,1].set_xlabel('Min. CF solar',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[1,1].tick_params(axis='x', labelsize=11)
ax[1,1].tick_params(axis='y', labelleft=False)
X4 = sm.add_constant(x4)
model4 = sm.OLS(list(y),X4).fit()
predictions4 = model4.predict(X4) 
print_model4 = model4.summary()
coef4 = model4.params
p_value4 = model4.summary2().tables[1]['P>|t|'].loc['Min CF solar']
R24 = model4.rsquared
NormSE4 = model4.bse/(y).mean()
ax[1,1].plot(x4,predictions4,color='darkblue')
#ax[1,1].legend(loc='lower left')
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R24, ),
      r'$P=%.3f$' % (p_value4, ),
      r'$\bar{SE}=%.3f$' % (NormSE4[1], )))
ax[1,1].text(0.6, 0.35, textstr, transform=ax[1,1].transAxes, fontsize=11,
          verticalalignment='top')

# 6th plot CF seas solar
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[1,2].scatter(CFseasonalratiosolar[Name], e_price_countryY[Name],c=c,alpha=0.5)
#ax[1,2].scatter(x6,y, label='Seas. solar', alpha=0.5, color='orange')
ax[1,2].set_xlabel('Seas. CF solar',fontsize = 12, weight='bold')
#ax[0,1].set_ylabel('System Cost [MEUR]',fontsize = 12, weight='bold')
ax[1,2].tick_params(axis='x', labelsize=11)
ax[1,2].tick_params(axis='y', labelleft=False)
X6 = sm.add_constant(x6)
model6 = sm.OLS(list(y),X6).fit()
predictions6 = model6.predict(X6) 
print_model6 = model6.summary()
coef6 = model6.params
p_value6 = model6.summary2().tables[1]['P>|t|'].loc['Seas. CF solar']
R26 = model6.rsquared
NormSE6 = model6.bse/(y).mean()
ax[1,2].plot(x6,predictions6,color='darkblue')
#ax[1,2].legend(loc='lower left')
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R26, ),
      r'$P=%.3f$' % (p_value6, ),
      r'$\bar{SE}=%.3f$' % (NormSE6[1], )))
ax[1,2].text(0.05, 0.35, textstr, transform=ax[1,2].transAxes, fontsize=11,
          verticalalignment='top')

# 7th plot hydrofactor
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[2,0].scatter(hydroinflowfactor[Name], e_price_countryY[Name],c=c,alpha=0.5)
#ax[2,0].scatter(x7,y, label='Hydrofactor', alpha=0.5, color='grey')
ax[2,0].set_xlabel('Hydrofactor',fontsize = 12, weight='bold')
ax[2,0].set_ylabel('<Elec. price> [EUR/MWh]',fontsize = 12, weight='bold')
ax[2,0].tick_params(axis='both', labelsize=11)
X7 = sm.add_constant(x7,has_constant='add')
model7 = sm.OLS(list(y),X7).fit()
predictions7 = model7.predict(X7) 
print_model7 = model7.summary()
coef7 = model7.params
p_value7 = model7.summary2().tables[1]['P>|t|'].loc['Hydrofactor']
R27 = model7.rsquared
NormSE7 = model7.bse/(y).mean()
ax[2,0].plot(x7,predictions7,color='darkblue')
#ax[2,0].legend()
#[2,0].legend(bbox_to_anchor=(1, 0.8),ncol=5,fontsize=10) 

textstr = '\n'.join((
     r'$R^2=%.3f$' % (R27, ),
     r'$P=%.3f$' % (p_value7, ),
     r'$\bar{SE}=%.3f$' % (NormSE7[1], )))
ax[2,0].text(0.6, 0.95, textstr, transform=ax[2,0].transAxes, fontsize=11,
         verticalalignment='top')
fig.suptitle('Linear Regression: '+constraint3, fontsize=18,weight='bold')
fig.legend(bbox_to_anchor=(0.95, 0.32),ncol=5,fontsize=10) 

fig.tight_layout()
savefigure1('System',constraint3,'Linreg1036',fig)

#%% plot solar vs wind
fig = plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,6))
# 1st plot
ax[0].scatter(SystemMultdf['CF wind'],SystemMultdf['CF solar'], alpha=0.5, color='black',label='Data')
ax[0].set_xlabel('<Load Weighted CF wind>',fontsize = 12, weight='bold')
ax[0].set_ylabel('<Load Weighted CF solar>',fontsize = 12, weight='bold')
ax[0].tick_params(axis='both', labelsize=11)
X1 = sm.add_constant(SystemMultdf['CF wind'])
model1 = sm.OLS(list(SystemMultdf['CF solar']),X1).fit()
predictions1 = model1.predict(X1) 
print_model1 = model1.summary()
coef1 = model1.params
p_value1 = model1.summary2().tables[1]['P>|t|'].loc['CF wind']
R21 = model1.rsquared
NormSE1 = model1.bse/(SystemMultdf['CF solar']).mean()
ax[0].plot(SystemMultdf['CF wind'],predictions1,color='Black',label='Reg.')
ax[0].legend()
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R21, ),
      r'$P=%.3f$' % (p_value1, ),
      r'$\bar{SE}=%.3f$' % (NormSE1[1], )))
ax[0].text(0.05, 0.15, textstr, transform=ax[0].transAxes, fontsize=11,
          verticalalignment='top')
# 2nd plot
# Add the data points
for Name in e_price_countryY.columns:
    c = colors[Name]
    ax[1].scatter(CF_onWindyear[Name], CF_solaryear[Name],label=Name,c=c)
#ax[1].scatter(SystemMultdf2['CF wind'],SystemMultdf2['CF solar'], alpha=0.5, color='black',label='Data')
ax[1].set_xlabel('<CF wind>',fontsize = 12, weight='bold')
ax[1].set_ylabel('<CF solar>',fontsize = 12, weight='bold')
ax[1].tick_params(axis='both', labelsize=11)
X2 = sm.add_constant(SystemMultdf2['CF wind'])
model2 = sm.OLS(list(SystemMultdf2['CF solar']),X2).fit()
predictions2 = model2.predict(X2) 
print_model2 = model2.summary()
coef2 = model2.params
p_value2 = model2.summary2().tables[1]['P>|t|'].loc['CF wind']
R22 = model2.rsquared
NormSE2 = model2.bse/(SystemMultdf2['CF solar']).mean()
ax[1].plot(SystemMultdf2['CF wind'],predictions2,color='Black',label='Reg.')
ax[1].legend(bbox_to_anchor=(1, 1),ncol=2)  
textstr = '\n'.join((
      r'$R^2=%.3f$' % (R22, ),
      r'$P=%.3f$' % (p_value2, ),
      r'$\bar{SE}=%.3f$' % (NormSE2[1], )))
ax[1].text(0.05, 0.15, textstr, transform=ax[1].transAxes, fontsize=11,
          verticalalignment='top')
fig.suptitle('Wind CF vs. Solar CF',fontsize = 18, weight = 'bold')
fig.tight_layout()
savefigure('System','CFwindVSCFsolar','Scatter',fig)


#%% Cheapest and most expensive years system cost and weighted electricity prices
# Calculate most expensive year and cheapest years 3 of each
Sortedsystemcost = systemcost.sort_values()

fig = plt.figure(figsize=(12,6))

fig, plt.plot(systemcost,color='red',lw=2,label='System Cost')
fig, plt.scatter(Sortedsystemcost.index[:3],Sortedsystemcost[:3],marker='X',s=100,color='green',label='Cheapest')
fig, plt.scatter(Sortedsystemcost.index[-3:],Sortedsystemcost[-3:],marker='X',s=100,color='Black',label='Most expensive')

fig, plt.title('System Cost '+constraint3,fontsize=15,weight='bold')
fig, plt.xlabel('Years',fontsize=12, weight='bold')
fig, plt.ylabel('System Cost [EUR/MWh]',fontsize=12,weight='bold')
fig, plt.xticks(weight='bold',fontsize=12)
fig, plt.yticks(weight='bold',fontsize=12)
fig, plt.legend()
factor = loads.sum().sum()/10**6
fig, plt.annotate("1998", xy=(Sortedsystemcost.index[0], Sortedsystemcost.values[0]), 
                   xytext=(1995, (152500/factor)), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("1999", xy=(Sortedsystemcost.index[1], Sortedsystemcost.values[1]), 
                   xytext=(2000, 152500/factor), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2000", xy=(Sortedsystemcost.index[2], Sortedsystemcost.values[2]), 
                   xytext=(2000, 155500/factor), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2014", xy=(Sortedsystemcost.index[-1], Sortedsystemcost.values[-1]), 
                  xytext=(2015-0.5, 175000/factor), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2013", xy=(Sortedsystemcost.index[-2], Sortedsystemcost.values[-2]), 
                  xytext=(2013-0.5, 167500/factor), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2013", xy=(Sortedsystemcost.index[-2], Sortedsystemcost.values[-2]), 
                  xytext=(2013-0.5, 167500/factor), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("1987", xy=(Sortedsystemcost.index[-3], Sortedsystemcost.values[-3]), 
                  xytext=(1986, 175000/factor), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig.tight_layout()

savefigure1('System',constraint3,'SystemCost',fig)

#%% # Tri plot
import matplotlib.tri as mtri
# Remember to change this
CheapCF = pd.DataFrame()
CheapCF['CF wind'] = SystemMultdf['CF wind'][19:22]
CheapCF['CF solar'] = SystemMultdf['CF solar'][19:22]
ExpCF = pd.DataFrame()
ExpCF['CF wind'] = SystemMultdf['CF wind'][[35,34,8]]
ExpCF['CF solar'] = SystemMultdf['CF solar'][[35,34,8]]
# Get X, Y, Z
X, Y, Z = SystemMultdf['CF wind'], SystemMultdf['CF solar'], systemcost

# Plot X,Y,Z
fig, axs = plt.subplots(figsize=(10,10))
axs.scatter(CheapCF['CF wind'],CheapCF['CF solar'],marker='X',color='green',s=200,label='Cheapest years',zorder=1)
axs.scatter(ExpCF['CF wind'],ExpCF['CF solar'],marker='X',color='black',s=200,label='Most expensive years', zorder=1)
axs.set_xlabel('Load Weighted  <Wind CF>',color='black',fontsize=15,weight='bold')
axs.set_ylabel('Load Weigthed  <Solar CF>',color='black',fontsize=15,weight='bold')
axs.set_title('System '+ constraint3,fontsize = 20,weight='bold')
plt.xticks(weight='bold',fontsize=12)
plt.yticks(weight='bold',fontsize=12)
axs.legend(prop={'size': 20})
# Text and arrow
axs.annotate("1998", xy=(CheapCF['CF wind'][0], CheapCF['CF solar'][0]), xytext=(0.26, 0.12125), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
axs.annotate("1999", xy=(CheapCF['CF wind'][1], CheapCF['CF solar'][1]), xytext=(0.255, 0.1225), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
axs.annotate("2000", xy=(CheapCF['CF wind'][2], CheapCF['CF solar'][2]), xytext=(0.25, 0.124), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
axs.annotate("2014", xy=(ExpCF['CF wind'][0], ExpCF['CF solar'][0]), xytext=(0.208, 0.12), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
axs.annotate("2013", xy=(ExpCF['CF wind'][1], ExpCF['CF solar'][1]), xytext=(0.215, 0.1180), bbox = dict(facecolor = 'grey', alpha = 0.2),
             fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
axs.annotate("1987", xy=(ExpCF['CF wind'][2], ExpCF['CF solar'][2]), xytext=(0.23, 0.113), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
#axs.set_aspect('equal')
#Triangulation plot
triang = mtri.Triangulation(X, Y)
tpc = axs.tripcolor(triang, Z, shading='flat',cmap='viridis_r',zorder=0)
cbar = fig.colorbar(tpc)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('System Cost [EUR/MWh]',size=15,weight='bold')
fig.tight_layout()
savefigure1('System',constraint3,'Triplot',fig)


#%% Several box plots

# Electricity prices
fig = plt.figure(figsize=(12,6))
fig, sns.boxplot(e_price_countryY)
fig, plt.xlabel('Countries',fontsize=15,weight='bold')
fig, plt.ylabel('Elec. price [EUR/MWh]',fontsize=15,weight='bold')
fig, plt.title('Boxplot Countries: Electricity Price - '+constraint3,fontsize=20, weight='bold')
fig, plt.xticks(fontsize=12, weight = 'bold',rotation=45)
fig, plt.yticks(fontsize=12, weight = 'bold')
fig.tight_layout()
savefigure1('System',constraint3,'BoxplotEprice',fig)

# CF wind
fig = plt.figure(figsize=(12,6))
fig, sns.boxplot(CF_onWindyear)
fig, plt.xlabel('Countries',fontsize=15,weight='bold')
fig, plt.ylabel('<CF wind>',fontsize=15,weight='bold')
fig, plt.title('Boxplot Countries: Wind Capacity Factors',fontsize=20, weight='bold')
fig, plt.xticks(fontsize=12, weight = 'bold',rotation=45)
fig, plt.yticks(fontsize=12, weight = 'bold')
fig.tight_layout()
savefigure('System','2XTransmission','BoxplotCFwind',fig)

# CF wind minimum
fig = plt.figure(figsize=(12,6))
fig, sns.boxplot(CFminfactor_yearwind)
fig, plt.xlabel('Countries',fontsize=15,weight='bold')
fig, plt.ylabel('Minimum CF wind',fontsize=15,weight='bold')
fig, plt.title('Boxplot Countries: Min. Wind Capacity Factors',fontsize=20, weight='bold')
fig, plt.xticks(fontsize=12, weight = 'bold',rotation=45)
fig, plt.yticks(fontsize=12, weight = 'bold')
fig.tight_layout()
savefigure('System','2XTransmission','BoxplotminCFwind',fig)

# CF solar
fig = plt.figure(figsize=(12,6))
fig, sns.boxplot(CF_solaryear)
fig, plt.xlabel('Countries',fontsize=15,weight='bold')
fig, plt.ylabel('<CF solar>',fontsize=15,weight='bold')
fig, plt.title('Boxplot Countries: Solar Capacity Factors',fontsize=20, weight='bold')
fig, plt.xticks(fontsize=12, weight = 'bold',rotation=45)
fig, plt.yticks(fontsize=12, weight = 'bold')
fig.tight_layout()
savefigure('System','2XTransmission','BoxplotCFsolar',fig)

# CF solar minimum
fig = plt.figure(figsize=(12,6))
fig, sns.boxplot(CFminfactor_yearsolar)
fig, plt.xlabel('Countries',fontsize=15,weight='bold')
fig, plt.ylabel('Minimum CF solar',fontsize=15,weight='bold')
fig, plt.title('Boxplot Countries: Min. Solar Capacity Factors',fontsize=20, weight='bold')
fig, plt.xticks(fontsize=12, weight = 'bold',rotation=45)
fig, plt.yticks(fontsize=12, weight = 'bold')
fig.tight_layout()
savefigure('System','2XTransmission','BoxplotminCFsolar',fig)


#%% Load generators
generators_timeYmod = modGenerators(generators_timeY)
#%%
Ones = pd.DataFrame(columns=loads_timeY.columns,index=loads_timeY.index)
Ones.fillna(1,inplace=True)
#%%
# Run of river
rorY = generators_timeYmod.filter(like='ror',axis=1)
#rorY = rorY.drop(['BA ror','RS ror'],axis=1)
rorY.columns = rorY.columns.str.rstrip(' ror')
rorY = rorY.sort_index(axis=1)
# Out commented below searches for same indexs in two dataframes
#common_cols = [col for col in set(loadssum.index).intersection(rorY.columns)]
#loadssumrorNames = loadssum[common_cols].sort_index()
rorYfactor = rorY/Ones
rorYfactor = rorYfactor.fillna(0)
rorYfactorsum = sumYear(rorYfactor)/sumYear(loads_timeY)

# onwind
onwindY = generators_timeYmod.filter(like='onwind',axis=1)
#onwindYsum = onwindYsum.drop(['BA onwind','RS onwind'],axis=1)
onwindY.columns = onwindY.columns.str.rstrip(' onwind')
onwindYfactor = onwindY/Ones
onwindYfactor = onwindYfactor.fillna(0)
onwindYfactorsum = sumYear(onwindYfactor)/sumYear(loads_timeY)

# solar 
solarY = generators_timeYmod.filter(like='solar',axis=1)
#solarY = solarY.drop(['BA solar','RS solar'],axis=1)
solarY.columns = solarY.columns.str.rstrip(' solar')
solarYfactor = solarY/Ones#/loads_timeY
solarYfactor = solarYfactor.fillna(0)
solarYfactorsum = sumYear(solarYfactor)/sumYear(loads_timeY)

# off wind
offwindY = generators_timeYmod.filter(like='offwind',axis=1) 
#onwindYsum = onwindYsum.drop(['BA onwind','RS onwind'],axis=1)
offwindY.columns = offwindY.columns.str.rstrip(' offwind')
offwindYfactor = offwindY/Ones#/loads_timeY
offwindYfactor = offwindYfactor.fillna(0)
offwindYfactorsum = sumYear(offwindYfactor)/sumYear(loads_timeY)

# PHS 
PHSY = storage_timeY.filter(like='PHS',axis=1)
#PHSY = PHSY.drop(['BA PHS','RS PHS'],axis=1)
PHSY.columns = PHSY.columns.str.rstrip('PHS')
PHSY.columns = PHSY.columns.str.rstrip(' ')
PHSYfactor = PHSY/Ones#/loads_timeY
PHSYfactor = PHSYfactor.fillna(0)
PHSYfactorsum = sumYear(PHSYfactor)/sumYear(loads_timeY)

# Hydro
hydroY = (storage_timeY.filter(like='hydro',axis=1))
#hydroY = hydroY.drop(['BA hydro','RS hydro'],axis=1)
hydroY.columns = hydroY.columns.str.rstrip(' hydro')
hydroYfactor = hydroY/Ones#/loads_timeY
hydroYfactor = hydroYfactor.fillna(0)
hydroYfactorsum = sumYear(hydroYfactor)/sumYear(loads_timeY)

# h2 electro is taken from electricity line hence minus (bus0 electricity) and reason why p0, because it takes the power before from the line and first after efficieny
h2electroY = (links_timep0Y.filter(like='Electrolysis',axis=1))
#h2electroY = h2electroY.drop(['BA H2 Electrolysis','RS H2 Electrolysis'],axis=1)
h2electroY.columns = h2electroY.columns.str.rstrip(' Electrolysis')
h2electroY.columns = h2electroY.columns.str.rstrip('H2')
h2electroY.columns = h2electroY.columns.str.rstrip(' ')
h2electroYfactor = -(h2electroY)/Ones#/loads_timeY)
h2electroYfactor = h2electroYfactor.fillna(0)
h2electroYfactorsum = sumYear(h2electroYfactor)/sumYear(loads_timeY)

# Fuel cell to electricity line hence  (bus0 store)
h2fuelY = (links_timep1Y.filter(like='Fuel Cell',axis=1))
#h2fuelY = h2fuelY.drop(['BA H2 Fuel Cell','RS H2 Fuel Cell'],axis=1)
h2fuelY.columns = h2fuelY.columns.str.rstrip(' Fuel Cell')
h2fuelY.columns = h2fuelY.columns.str.rstrip('H2')
h2fuelY.columns = h2fuelY.columns.str.rstrip(' ')
h2fuelYfactor = -(h2fuelY/Ones)#/loads_timeY)
h2fuelYfactor = h2fuelYfactor.fillna(0)
h2fuelYfactorsum = sumYear(h2fuelYfactor)/sumYear(loads_timeY)

# Battery charge is take from the electricity bus (bus0 electricity)
batchargeY = (links_timep0Y.filter(like='battery charger',axis=1))
#batchargeY = batchargeY.drop(['BA battery charger','RS battery charger'],axis=1)
batchargeY.columns = batchargeY.columns.str.rstrip(' charger')
batchargeY.columns = batchargeY.columns.str.rstrip('battery')
batchargeY.columns = batchargeY.columns.str.rstrip(' ')
batchargeYfactor = -(batchargeY/Ones)#/loads_timeY)
batchargeYfactor = batchargeYfactor.fillna(0)
batchargeYfactorsum = sumYear(batchargeYfactor)/sumYear(loads_timeY)

# Battery discharge is taken from the battery (bus0 battery)
batdischargeY = (links_timep1Y.filter(like='battery discharger',axis=1))
#batdischargeY = batdischargeY.drop(['BA battery discharger','RS battery discharger'],axis=1)
batdischargeY.columns = batdischargeY.columns.str.rstrip(' discharger')
batdischargeY.columns = batdischargeY.columns.str.rstrip('battery')
batdischargeY.columns = batdischargeY.columns.str.rstrip(' ')
batdischargeYfactor = -(batdischargeY/Ones)#/loads_timeY)
batdischargeYfactor = batdischargeYfactor.fillna(0)
batdischargeYfactorsum = sumYear(batdischargeYfactor)/sumYear(loads_timeY)

# gas usage (bus0 is gasline bus1 is electricity)
OCGTY = (links_timep1Y.filter(like='OCGT',axis=1))
#OCGTY = OCGTY.drop(['BA OCGT','RS OCGT'],axis=1)
OCGTY.columns = OCGTY.columns.str.rstrip('OCGT')
OCGTY.columns = OCGTY.columns.str.rstrip(' ')
OCGTYfactor = -(OCGTY/Ones)#/loads_timeY)
OCGTYfactor = OCGTYfactor.fillna(0)
OCGTYfactorsum = sumYear(OCGTYfactor)/sumYear(loads_timeY)
#%%
# Transmission
TransNames = list(links.index[150:])
TransY = (links_timep0Y.filter(items=TransNames,axis=1))
TransY2 = (links_timep1Y.filter(items=TransNames,axis=1))
TransYCountry0 = pd.DataFrame() # Bus 0
TransYCountry1 = pd.DataFrame() # Bus 1
for i in NAMES:
    TransYCountry0[i] = TransY.filter(regex=i+'-').sum(axis=1)
    TransYCountry1[i] = TransY2.filter(regex='-'+i).sum(axis=1)
TransYCountryfactor0 = -(TransYCountry0/Ones)#/loads_timeY - because bus0 is posive if withdrawing
TransYCountryfactor1 = -(TransYCountry1/Ones)#/loads_timeY - because bus1 is positive if withdrawing
TransYCountryfactor0 = TransYCountryfactor0.fillna(0)
TransYCountryfactor1 = TransYCountryfactor1.fillna(0)
TransYCountryfactorsum0 = sumYear(TransYCountryfactor0)/sumYear(loads_timeY)
TransYCountryfactorsum1 = sumYear(TransYCountryfactor1)/sumYear(loads_timeY)
# Double check Names - Links can be connected to or connect to a country
TransYNames = TransY.filter(regex='BA-',axis=1)
#%% load
ElecLoadsum = -sumYear(loads_timeY)/sumYear(loads_timeY)


#%%
ALLgen = rorYfactor+onwindYfactor+offwindYfactor+solarYfactor+hydroYfactor+PHSYfactor+h2electroYfactor+h2fuelYfactor+batchargeYfactor+batdischargeYfactor+OCGTYfactor+TransYCountryfactor0+TransYCountryfactor1-loads_timeY
ALLgensum = rorYfactorsum+onwindYfactorsum+offwindYfactorsum+solarYfactorsum+hydroYfactorsum+PHSYfactorsum+h2electroYfactorsum+h2fuelYfactorsum+batchargeYfactorsum+batdischargeYfactorsum+OCGTYfactorsum+TransYCountryfactorsum0+TransYCountryfactorsum1

#%% SUmmed pr year hourly generation plot hereunder 
# Wind gen, solar gen, ror gen, bat discharge, h2 fuel, transmission into, gas into, PHS into, hydro into are all calculated postive into the country
# Bat charge, h2 electrolysis, PHS out, hydro into, transmission out of are all calculated out of system

Country = 'NO'
Country2 = 'DK'
BarNames = {'ror': (rorYfactorsum[Country]),
            'onwind':(onwindYfactorsum[Country]),
            'offwind':(offwindYfactorsum[Country]),
            'solar':(solarYfactorsum[Country]),
            'PHS':(PHSYfactorsum[Country]),
            'hydro':(hydroYfactorsum[Country]),
            'H2electro':(h2electroYfactorsum[Country]),
            'H2fuel':(h2fuelYfactorsum[Country]),
            'Batcharge':(batchargeYfactorsum[Country]),
            'Batdischarge':(batdischargeYfactorsum[Country]),
            'OCGT':(OCGTYfactorsum[Country]),
            'TRMS0':(TransYCountryfactorsum0[Country]),
            'TRMS1':(TransYCountryfactorsum1[Country]),
            'Elec. demand':(ElecLoadsum[Country])
            }
BarNames2 = {'ror': (rorYfactorsum[Country2]),
            'onwind':(onwindYfactorsum[Country2]),
            'offwind':(offwindYfactorsum[Country2]),
            'solar':(solarYfactorsum[Country2]),
            'PHS':(PHSYfactorsum[Country2]),
            'hydro':(hydroYfactorsum[Country2]),
            'H2electro':(h2electroYfactorsum[Country2]),
            'H2fuel':(h2fuelYfactorsum[Country2]),
            'Batcharge':(batchargeYfactorsum[Country2]),
            'Batdischarge':(batdischargeYfactorsum[Country2]),
            'OCGT':(OCGTYfactorsum[Country2]),
            'TRMS0':(TransYCountryfactorsum0[Country2]),
            'TRMS1':(TransYCountryfactorsum1[Country2]),
            'Elec. demand':(ElecLoadsum[Country2])
            }
BarNamesSystem = {'ror': (sumYear(rorYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'onwind':(sumYear(onwindYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'offwind':(sumYear(offwindYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'solar':(sumYear(solarYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'PHS':(sumYear(PHSYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'hydro':(sumYear(hydroYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'H2electro':(sumYear(h2electroYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'H2fuel':(sumYear(h2fuelYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'Batcharge':(sumYear(batchargeYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'Batdischarge':(sumYear(batdischargeYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'OCGT':(sumYear(OCGTYfactor.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'TRMS0':(sumYear(TransYCountryfactor0.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'TRMS1':(sumYear(TransYCountryfactor1.sum(axis=1))/sumYear(loads_timeY.sum(axis=1))),
            'Elec. demand':-(sumYear(loads_timeY.sum(axis=1))/sumYear(loads_timeY.sum(axis=1)))
            }

dfBarnames = pd.DataFrame.from_dict(BarNames)
dfBarnames.index = dfBarnames.index.year
dfBarnames2 = pd.DataFrame.from_dict(BarNames2)
dfBarnames2.index = dfBarnames2.index.year
dfBarnamesSystem = pd.DataFrame.from_dict(BarNamesSystem)
dfBarnamesSystem.index = dfBarnamesSystem.index.year
mycolors = 'b','darkblue','aliceblue','yellow','purple','cyan','darkgreen','lightgreen','black','grey','brown','r','darkred','lightpink'
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12,8))
fig, dfBarnames.plot(ax=axes[0],kind="bar", stacked=True,rot=90,color=mycolors,legend=False)
fig.legend(loc='upper right', bbox_to_anchor=(1.21, 0.79),fontsize=17)
fig, dfBarnames2.plot(ax=axes[1],kind="bar", stacked=True,rot=90,color=mycolors,legend=False)
axes[0].tick_params(labelbottom=False,labelsize=16)
axes[1].tick_params(labelsize=16)
axes[0].set_title(Country,fontsize=14,weight='bold')
axes[1].set_title(Country2,fontsize=14,weight='bold')
fig.suptitle('Electricity Overview - '+constraint3, fontsize=20,weight='bold')
axes[1].set_xlabel('Years',weight='bold',fontsize = 20)
axes[1].set_xticklabels(dfBarnamesSystem.index,fontweight='bold')
fig.text(-0.0125, 0.5, 'acc. $\sum$X/$\sum$ load', va='center', rotation='vertical',fontsize=20,weight='bold')
fig.tight_layout()
savefigure1('System',constraint3,'accelectricityplot'+Country+Country2,fig)

# # System
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12,8))
fig, dfBarnamesSystem.plot(ax=axes,kind="bar", stacked=True,rot=90,color=mycolors,legend=False)
fig.legend(loc='upper right', bbox_to_anchor=(1.21, 0.79),fontsize=17)
axes.tick_params(labelbottom=True,labelsize=16)
axes.set_title('System - '+constraint3+' - Electricity Overview',fontsize=18,weight='bold')
axes.set_xticklabels(dfBarnamesSystem.index,fontweight='bold')
axes.set_ylabel('acc. $\sum$X/$\sum$ load',fontsize=20, weight='bold')
fig.tight_layout()
savefigure1('System',constraint3,'accelectricityplotsystem',fig)

#%% Gas Usage plot - Calendar plot
from matplotlib import colors
Country = 'DK'
average = '2d' # Frequency
OCGTDaily = OCGTYfactor.groupby(pd.Grouper(freq=average)).mean() # mean frequency
OCGTDaily.dropna()
OCGTDailySystem = (OCGTDaily).sum(axis=1) # System average
OCGTDailyX = OCGTDaily
X = OCGTDailyX.index.strftime('%m-%d')
fig, ax = plt.subplots(figsize=(18,6))
j = 0
Y2 = 0
for i in range(1979,2016):
    XX = OCGTDaily.index.year == i
    XX = XX.astype(np.int)
    #print(XX)
    Y = sum(XX)
    Y2 +=Y
    Ylen = np.full((1,Y),i)
    Xlen = X[(Y2-Y):Y2]
    Xlen2 = np.linspace(1,Y,Y)
    #print(Xlen)
    C = OCGTDaily[Country][(Y2-Y):Y2]
    #print(C)
    j +=1
    sct = plt.scatter(Xlen2,Ylen,s=50,c=C,cmap='YlOrRd',alpha=1,norm=colors.PowerNorm(gamma=0.4),vmin=0,vmax=3)# Powernorm colorbar
    # Color cmap = YlOrRd
#ax.xaxis.set_major_locator(MonthLocator(interval=1))
#ax.xaxis.set_major_formatter(DateFormatter('%m'))
fig, plt.vlines(92/2,1979,2016,lw=1.5,linestyles='dashed',color='black')
fig, plt.vlines(275/2,1979,2016,lw=1.5,linestyles='dashed',color='black')
plt.annotate("Primo Apr.", xy=(92/2, 1979), xytext=(92/2-5, 1973), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.annotate("Primo Oct.", xy=(275/2, 1979), xytext=(275/2-5, 1973), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})

fig, plt.xlabel(average+'*days', fontsize = 18, weight='bold')
fig, plt.xticks(fontsize=12, weight='bold')
fig, plt.ylabel('Years', fontsize=18, weight='bold')
fig, plt.yticks(fontsize=12, weight='bold')
fig, plt.title(Country+' '+average+' avg. Gas Usage - '+constraint3, fontsize=20, weight='bold')
cbar = fig.colorbar(sct,pad = 0.01)
cbar.set_label(average+' avg. gas usage [MWh]', size=15, weight='bold')
cbar.ax.tick_params(labelsize=15) 
fig.tight_layout()
savefigure1('System',constraint3,'Calendarplot'+Country,fig)

#%% System gas usage calendar plot
fig, ax = plt.subplots(figsize=(18,6))
j = 0
Y2 = 0
for i in range(1979,2016):
    XX = OCGTDaily.index.year == i
    XX = XX.astype(np.int)
    #print(XX)
    Y = sum(XX)
    Y2 +=Y
    Ylen = np.full((1,Y),i)
    Xlen = X[(Y2-Y):Y2]
    Xlen2 = np.linspace(1,Y,Y)
    #print(Xlen)
    C = OCGTDailySystem[(Y2-Y):Y2]
    #print(C)
    j +=1
    sct = plt.scatter(Xlen2,Ylen,s=50,c=C,cmap='YlOrRd',alpha=1,norm=colors.PowerNorm(gamma=2),vmin=0,vmax=68000)
#ax.xaxis.set_major_locator(MonthLocator(interval=1))
#ax.xaxis.set_major_formatter(DateFormatter('%m'))
fig, plt.vlines(92/2,1979,2016,lw=1.5,linestyles='dashed',color='black')
fig, plt.vlines(275/2,1979,2016,lw=1.5,linestyles='dashed',color='black')
plt.annotate("Primo Apr.", xy=(92/2, 1979), xytext=(92/2-5, 1973), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.annotate("Primo Oct.", xy=(275/2, 1979), xytext=(275/2-5, 1973), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})

fig, plt.xlabel(average+'*days', fontsize = 18, weight='bold')
fig, plt.xticks(fontsize=12, weight='bold')
fig, plt.ylabel('Years', fontsize=18, weight='bold')
fig, plt.yticks(fontsize=12, weight='bold')
fig, plt.title('System'+' ' +average+' avg. Gas Usage - '+constraint3, fontsize=20, weight='bold')
cbar = fig.colorbar(sct,pad = 0.01)
cbar.set_label(average+' avg. gas usage [MWh]', size=15, weight='bold')
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
savefigure1('System',constraint3,'Calendarplot'+'System',fig)

#%% Calculations for map plots
CF_windmean = CF_onWindyear.mean(axis=0)
CF_windstd = CF_onWindyear.std(axis=0)
CF_windCV = ((CF_windstd/CF_windmean)).fillna(0) # Coefficient of variation
CF_solarmean = CF_solaryear.mean(axis=0)
CF_solarstd = CF_solaryear.std(axis=0)
CF_solarCV = (CF_solarstd/CF_solarmean).fillna(0) # Coefficient of variation
e_pricemean = e_price_countryY.mean(axis=0)
e_pricestd = e_price_countryY.std(axis=0)
e_priceCV = (e_pricestd/e_pricemean).fillna(0)

#%% Map plot CF wind
fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=0.7)
ax.coastlines(resolution='110m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.6)
ax.set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                       'ME','MK','MT','RU','SM','UA','VA','XK'}
shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_1 = reader.records()
name_loop = 'start'
for country in countries_1:
    if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'                
        else:
            name_loop = country.attributes['ISO_A2']
        for country_CF in CF_windCV.index.values:
            if country_CF == name_loop:
                color_value = CF_windCV.loc[country_CF] #[PC_NO-1]
                if color_value <= 0:
                    color_value = np.absolute(color_value)*5 # Multipliing with 5 to amplify color and the dividing vmax 
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 0, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(CF_windmean.loc[country_CF],2))+'\n$\pm$'+str(round(CF_windstd.loc[country_CF],2)),fontsize=13,weight='bold')
                else:
                    color_value = np.absolute(color_value)*5
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(0, 0, 1), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(CF_windmean.loc[country_CF],2))+'\n$\pm$'+str(round(CF_windstd.loc[country_CF],2)),fontsize=13,weight='bold')
    else:
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=0.7, 
                             edgecolor="black", label=country.attributes['ADM0_A3'])

cmap = LinearSegmentedColormap.from_list('mycmap', ['white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1/5)
cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_xlabel('Coefficient of variation',fontsize=18,weight='bold')
ax.set_title('Annual CF wind variability 1979-2015',fontsize=22, weight='bold')
savefigure1('System',constraint3,'Mapplot'+'CFwind',fig)


#plt.subplots_adjust(hspace=0.02, wspace=0.04)
#%% Map plot CF solar
fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=0.7)
ax.coastlines(resolution='110m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.6)
ax.set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                       'ME','MK','MT','RU','SM','UA','VA','XK'}
shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_1 = reader.records()
name_loop = 'start'
for country in countries_1:
    if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'                
        else:
            name_loop = country.attributes['ISO_A2']
        for country_CF in CF_solarCV.index.values:
            if country_CF == name_loop:
                color_value = CF_solarCV.loc[country_CF] #[PC_NO-1]
                if color_value >= 0:
                    color_value = np.absolute(color_value)*10
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 0, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(CF_solarmean.loc[country_CF],2))+'\n$\pm$'+str(round(CF_solarstd.loc[country_CF],2)),fontsize=13,weight='bold')
                else:
                    color_value = np.absolute(color_value)*10
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(0, 0, 1), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(CF_solarmean.loc[country_CF],2))+'\n$\pm$'+str(round(CF_solarstd.loc[country_CF],2)),fontsize=13,weight='bold')
    else:
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=0.7, 
                             edgecolor="black", label=country.attributes['ADM0_A3'])

cmap = LinearSegmentedColormap.from_list('mycmap', ['white',(1,0.666,0.666),(1,0.333,0.333),(1,0,0)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1/10)
cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_xlabel('Coefficient of variation',fontsize=18,weight='bold')
ax.set_title('Annual CF solar variability 1979-2015',fontsize=22, weight='bold')
savefigure1('System',constraint3,'Mapplot'+'CFsolar',fig)
#%% Map plot electricity
fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=0.7)
ax.coastlines(resolution='110m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.6)
ax.set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                       'ME','MK','MT','RU','SM','UA','VA','XK'}
shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_1 = reader.records()
name_loop = 'start'
for country in countries_1:
    if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'                
        else:
            name_loop = country.attributes['ISO_A2']
        for country_CF in CF_solarCV.index.values:
            if country_CF == name_loop:
                color_value = e_priceCV.loc[country_CF] #[PC_NO-1]
                if color_value >= 0:
                    color_value = np.absolute(color_value)*4
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 0, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(e_pricemean.loc[country_CF],1))+'\n$\pm$'+str(round(e_pricestd.loc[country_CF],1)),fontsize=13,weight='bold')
                else:
                    color_value = np.absolute(color_value)*4
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(0, 0, 1), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(e_pricemean.loc[country_CF],1))+'\n$\pm$'+str(round(e_pricestd.loc[country_CF],1)),fontsize=13,weight='bold')
    else:
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=0.7, 
                             edgecolor="black", label=country.attributes['ADM0_A3'])

cmap = LinearSegmentedColormap.from_list('mycmap', ['white',(1,0.666,0.666),(1,0.333,0.333),(1,0,0)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1/4)
cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_xlabel('Coefficient of variation',fontsize=18,weight='bold')
ax.set_title('Annual electricity price variability 1979-2015 - '+constraint3,fontsize=22, weight='bold')
savefigure1('System',constraint3,'Mapplot'+'eprice',fig)

#%% Transmission capacity of each country calculation
# Summing transmission capacity connecting to a country for each countri
Countrytransmission = pd.DataFrame()
Countrytransmission = links[150:].T
TRMScap = pd.DataFrame()
for i in NAMES:
    TRMScap[i] = Countrytransmission.filter(regex=i).sum(axis=1)
# To double check names and trms. capacities
TRMS_country = Countrytransmission.filter(regex='DE')
#%% Transmission calc for map plot
TRMS_coef = (TRMScap/loadssum)*1000 # Per mille
TRMS_mean = TRMS_coef.mean(axis=0)
TRMS_std = TRMS_coef.std(axis=0)
TRMS_CV = (TRMS_std/TRMS_mean).fillna(0)
#%% Map plot - Transmission coefficient
fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=0.7)
ax.coastlines(resolution='110m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.6)
ax.set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                       'ME','MK','MT','RU','SM','UA','VA','XK'}
shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_1 = reader.records()
name_loop = 'start'
for country in countries_1:
    if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'                
        else:
            name_loop = country.attributes['ISO_A2']
        for country_CF in CF_solarCV.index.values:
            if country_CF == name_loop:
                color_value = TRMS_CV.loc[country_CF] #[PC_NO-1]
                if color_value >= 0:
                    color_value = np.absolute(color_value)*2.5
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 0, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(TRMS_mean.loc[country_CF],1))+'\n$\pm$'+str(round(TRMS_std.loc[country_CF],1)),fontsize=13,weight='bold')
                else:
                    color_value = np.absolute(color_value)*2.5
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(0, 0, 1), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(TRMS_mean.loc[country_CF],1))+'\n$\pm$'+str(round(TRMS_std.loc[country_CF],1)),fontsize=13,weight='bold')
    else:
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=0.7, 
                             edgecolor="black", label=country.attributes['ADM0_A3'])

cmap = LinearSegmentedColormap.from_list('mycmap', ['white',(1,0.666,0.666),(1,0.333,0.333),(1,0,0)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1/2.5)
cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_xlabel('Coefficient of variation',fontsize=18,weight='bold')
ax.set_title('Annual Transmission variability 1979-2015 - '+constraint3,fontsize=22, weight='bold')
savefigure1('System',constraint3,'Mapplot'+'TRMScoef.',fig)
# #%% PCA - GAS usage for 37 years of data
# X = OCGTYfactor
# X_mean = X.mean(axis=0)
# X_cent = X-X_mean                                          # Centering data 
# c = 1/np.sqrt((X_cent**2).mean().sum(axis=0))
# #c = 1/np.sqrt(np.sum((X_cent)**2)/len(X))                 # Standardization constant (1/sigma)
# X_norm = c*X_cent                                          # Standardization
# Cov_mat = np.cov(X_norm.T,bias=True)                       # Co-variance matrix
# eig_val, eig_vec = np.linalg.eig(Cov_mat)                  # Eigen values and eigen vectors 
# a_k = np.dot(X_norm,eig_vec)                               # Amplitudes of eigenvectors

# # map plots 4 first PC
# fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
# linewidth = 0.8
# panels = ['(a)', '(b)', '(c)', '(d)']
# eigen_values = eig_val
# eigen_vectors = -eig_vec #invert eigenvector to make the results more intuitive
# variance_explained = []
# for j in eigen_values:
#       variance_explained.append((j/sum(eigen_values))*100)
# variance_explained_cumulative = np.cumsum(variance_explained)
# data_names = OCGTYfactor.columns
# VT = pd.DataFrame(data=eigen_vectors, index=data_names)

# for i in range(4):
#     ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
#     ax[i].coastlines(resolution='110m')
#     ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
#     ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
#     europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
#                             'ME','MK','MT','RU','SM','UA','VA','XK'}
#     shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
#     reader = shpreader.Reader(shpfilename)
#     countries_1 = reader.records()
#     name_loop = 'start'
#     PC_NO = i+1
#     for country in countries_1:
#         if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
#             if country.attributes['NAME'] == 'Norway':
#                 name_loop = 'NO'
#             elif country.attributes['NAME'] == 'France':
#                 name_loop = 'FR'                
#             else:
#                 name_loop = country.attributes['ISO_A2']
#             for country_PSA in VT.index.values:
#                 if country_PSA == name_loop:
#                     color_value = VT.loc[country_PSA][PC_NO-1]
#                     if color_value <= 0:
#                         color_value = np.absolute(color_value)
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#                     else:
#                         color_value = np.absolute(color_value)
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#         else:
#             ax[i].add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
#                                   edgecolor="black", label=country.attributes['ADM0_A3'])

#     ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes);
#     ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%', 
#                 fontsize=12, transform=ax[i].transAxes);

# cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
# shrink = 0.08
# ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
# cbar.ax.tick_params(labelsize=12)
# plt.subplots_adjust(hspace=0.02, wspace=0.04)                               

# #%% PCA year 2014
# # remember to change back to 2014
# file2014 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_2014_0.05.h5"
# network2014 = pypsa.Network(file2014)
# network2014.name = file2014
# #%%
# OCGTY2014 = network2014.links_t.p0.filter(regex='OCGT')
# OCGTY2014.columns = NAMES

# X = OCGTY2014
# X_mean = X.mean(axis=0)
# X_cent = X-X_mean                                          # Centering data 
# #c = 1/np.sqrt((X_cent**2).mean().sum(axis=0)) 
# c = 1/np.sqrt(np.sum((X_cent)**2)/len(X))             # standardizing constant (1/sigma)
# #c=1
# X_norm = c*X_cent
#                                         # standardinzing
# Cov_mat = np.cov(X_norm.T,bias=True)                       # Co-variance matrix
# eig_val, eig_vec = np.linalg.eig(Cov_mat)                  # Eigen values and eigen vectors 
# a_k = np.dot(X_norm,eig_vec)                               # Amplitudes of eigenvectors


# #%% map plots 4 first PC
# fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
# linewidth = 0.8
# panels = ['(a)', '(b)', '(c)', '(d)']
# eigen_values = eig_val
# eigen_vectors = -eig_vec #invert eigenvector to make the results more intuitive
# variance_explained = []
# for j in eigen_values:
#       variance_explained.append((j/sum(eigen_values))*100)
# variance_explained_cumulative = np.cumsum(variance_explained)
# data_names = NAMES
# VT = pd.DataFrame(data=eigen_vectors, index=data_names)

# for i in range(4):
#     ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
#     ax[i].coastlines(resolution='110m')
#     ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
#     ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
#     europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
#                             'ME','MK','MT','RU','SM','UA','VA','XK'}
#     shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
#     reader = shpreader.Reader(shpfilename)
#     countries_1 = reader.records()
#     name_loop = 'start'
#     PC_NO = i+1
#     for country in countries_1:
#         if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
#             if country.attributes['NAME'] == 'Norway':
#                 name_loop = 'NO'
#             elif country.attributes['NAME'] == 'France':
#                 name_loop = 'FR'                
#             else:
#                 name_loop = country.attributes['ISO_A2']
#             for country_PSA in VT.index.values:
#                 if country_PSA == name_loop:
#                     color_value = VT.loc[country_PSA][PC_NO-1]
#                     if color_value <= 0:
#                         color_value = np.absolute(color_value)
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#                     else:
#                         color_value = np.absolute(color_value)
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#         else:
#             ax[i].add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
#                                   edgecolor="black", label=country.attributes['ADM0_A3'])

#     ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes);
#     ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%', 
#                 fontsize=12, transform=ax[i].transAxes);

# cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
# shrink = 0.08
# ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
# cbar.ax.tick_params(labelsize=12)
# plt.subplots_adjust(hspace=0.02, wspace=0.04)  
#%%
# # time_index = network2014.loads_t.p.index
# # season_plot(a_k, time_index, 'For gas usage')
# # #%% PCA year 1998
# # file1998 = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_1998_0.05.h5"
# # network1998 = pypsa.Network(file1998)
# # network1998.name = file1998
# # #%% Reconstruct original data
# # a1 = np.dot(a_k,eig_vec.T)
# # a2 = 1/c2.values
# # a3 = a1*a2
# # a4 = X_mean.values
# # a5 = a4+a2*a1
# #%%
# OCGTY1998 = network1998.links_t.p0.filter(regex='OCGT')
# OCGTY1998.columns = NAMES

# X = OCGTY1998
# X_mean = X.mean(axis=0)
# X_cent = X-X_mean                                          # Centering data 
# c = 1/np.sqrt((X_cent**2).mean().sum(axis=0))              # Standardization constant (1/sqrt(Var))
# c2 = 1/np.sqrt(np.sum((X_cent)**2)/len(X))
# X_norm = c2*X_cent                                          # Standardization
# Cov_mat = np.cov(X_norm.T,bias=True)                       # Co-variance matrix
# eig_val, eig_vec = np.linalg.eig(Cov_mat)                  # Eigen values and eigen vectors 
# a_k = np.dot(X_norm,eig_vec)                               # Amplitudes of eigenvectors

# # map plots 4 first PC
# fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
# linewidth = 0.8
# panels = ['(a)', '(b)', '(c)', '(d)']
# eigen_values = eig_val
# eigen_vectors = -eig_vec #invert eigenvector to make the results more intuitive
# variance_explained = []
# for j in eigen_values:
#       variance_explained.append((j/sum(eigen_values))*100)
# variance_explained_cumulative = np.cumsum(variance_explained)
# data_names = NAMES
# VT = pd.DataFrame(data=eigen_vectors, index=data_names)

# for i in range(4):
#     ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
#     ax[i].coastlines(resolution='110m')
#     ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
#     ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
#     europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
#                             'ME','MK','MT','RU','SM','UA','VA','XK'}
#     shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
#     reader = shpreader.Reader(shpfilename)
#     countries_1 = reader.records()
#     name_loop = 'start'
#     PC_NO = i+1
#     for country in countries_1:
#         if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
#             if country.attributes['NAME'] == 'Norway':
#                 name_loop = 'NO'
#             elif country.attributes['NAME'] == 'France':
#                 name_loop = 'FR'                
#             else:
#                 name_loop = country.attributes['ISO_A2']
#             for country_PSA in VT.index.values:
#                 if country_PSA == name_loop:
#                     color_value = VT.loc[country_PSA][PC_NO-1]
#                     if color_value <= 0:
#                         color_value = np.absolute(color_value)
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#                     else:
#                         color_value = np.absolute(color_value)
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#         else:
#             ax[i].add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
#                                   edgecolor="black", label=country.attributes['ADM0_A3'])

#     ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes);
#     ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%', 
#                 fontsize=12, transform=ax[i].transAxes);

# cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
# shrink = 0.08
# ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
# cbar.ax.tick_params(labelsize=12)
# plt.subplots_adjust(hspace=0.02, wspace=0.04)  

# #%%
# #file = r'D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_0.125_0.05.h5'
# #file = '../data/postnetwork-elec_only_0.125_0.05.h5'
# file = r"D:\Pre-project (Data)\transmission_0.125\transmission_0.125\postnetwork-elec_only_2015_0.05.h5"
# network = pypsa.Network(file)
# network.name = file

# # file1 = r"C:\Users\laur1\OneDrive\4. Civil - semester\postnetwork-elec_only_0.125_0.05.h5"
# # network1 = pypsa.Network(file1)
# # network1.name = file1


# #%%
# generation = network.generators_t.p.groupby(network.generators.bus, axis=1).sum()
# load = network.loads_t.p_set
# mismatch = generation - load

# X = mismatch
# X_mean = np.mean(X,axis=0)
# X_mean = np.array(X_mean.values).reshape(30,1)
# X_cent = np.subtract(X,X_mean.T)
# c = 1/np.sqrt(np.sum(np.mean(((X_cent.values)**2),axis=0)))

# #c2 = 1/np.sqrt(np.sum((X-X_mean)**2)/len(X))
# B = c*(X_cent.values)
# #B = c2*(X_cent)
# #B = X_cent
# #C_new = np.dot(B.T,B)*1/(8760-1)
# C = np.cov(B.T,bias=True) 
# eig_val, eig_vec = np.linalg.eig(C) 
# T = np.dot(B,eig_vec)




# # In[14]:


# fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
# linewidth = 0.8
# panels = ['(a)', '(b)', '(c)', '(d)']
# eigen_values = eig_val
# eigen_vectors = - eig_vec #invert eigenvector to make the results more intuitive
# variance_explained = []
# for j in eigen_values:
#       variance_explained.append((j/sum(eigen_values))*100)
# variance_explained_cumulative = np.cumsum(variance_explained)
# data_names = network.loads_t.p.columns
# VT = pd.DataFrame(data=eigen_vectors, index=data_names)

# for i in range(4):
#     ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
#     ax[i].coastlines(resolution='110m')
#     ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
#     ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
#     europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
#                             'ME','MK','MT','RU','SM','UA','VA','XK'}
#     shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
#     reader = shpreader.Reader(shpfilename)
#     countries_1 = reader.records()
#     name_loop = 'start'
#     PC_NO = i+1
#     for country in countries_1:
#         if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
#             if country.attributes['NAME'] == 'Norway':
#                 name_loop = 'NO'
#             elif country.attributes['NAME'] == 'France':
#                 name_loop = 'FR'                
#             else:
#                 name_loop = country.attributes['ISO_A2']
#             for country_PSA in VT.index.values:
#                 if country_PSA == name_loop:
#                     color_value = VT.loc[country_PSA][PC_NO-1]
#                     if color_value <= 0:
#                         color_value = np.absolute(color_value)*1.5
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#                     else:
#                         color_value = np.absolute(color_value)*1.5
#                         ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
#                                               alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
#         else:
#             ax[i].add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
#                                   edgecolor="black", label=country.attributes['ADM0_A3'])

#     ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes);
#     ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%', 
#                 fontsize=12, transform=ax[i].transAxes);

# cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
# shrink = 0.08
# ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
# cbar.ax.tick_params(labelsize=12)
# plt.subplots_adjust(hspace=0.02, wspace=0.04)
# #plt.savefig('figures/sec3_combined_PCs.pdf', bbox_inches='tight')

# #%%
# def season_plot(T, time_index, file_name):
#     """
#     Parameters
#     ----------
#     T : Matrix
#         Principle component amplitudes. Given by: B*eig_val (so the centered and scaled data dotted with the eigen values)
#     data_index : panda index information
#         index for a year (used by panda's dataframe')
#     file_name: array of strings
#         Name of the datafile there is worked with

#     Returns
#     -------
#     Plot of seasonal distribution
#     """
#     T = pd.DataFrame(data=T,index=time_index)
#     T_avg_hour = T.groupby(time_index.hour).mean() # Hour
#     T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day

#     # Upper left figure
#     plt.figure(figsize=(16,10))
#     plt.subplot(2,2,1)
#     plt.plot(T_avg_hour[0],label='k=1')
#     plt.plot(T_avg_hour[1],label='k=2')
#     plt.plot(T_avg_hour[2],label='k=3')
#     plt.xticks(ticks=range(0,24,2))
#     plt.legend(loc='upper right',bbox_to_anchor=(1,1))
#     plt.xlabel("Hours")
#     plt.ylabel("a_k interday")
#     plt.title("Hourly average for k-values for 2015 ")
#     # Upper right figure
#     x_ax = range(len(T_avg_day[0])) # X for year plot
#     plt.subplot(2,2,2)
#     plt.plot(x_ax,T_avg_day[0],label='k=1')
#     plt.plot(x_ax,T_avg_day[1],label='k=2')
#     plt.plot(x_ax,T_avg_day[2],label='k=3')
#     plt.legend(loc='upper left',bbox_to_anchor=(1,1))
#     plt.xlabel("day")
#     plt.ylabel("a_k seasonal")
#     plt.title("daily average for k-values for 2015 ")
#     # Lower left figure
#     plt.subplot(2,2,3)
#     plt.plot(T_avg_hour[3],label='k=4',color="c")
#     plt.plot(T_avg_hour[4],label='k=5',color="m")
#     plt.plot(T_avg_hour[5],label='k=6',color="y")
#     plt.xticks(ticks=range(0,24,2))
#     plt.legend(loc='upper right',bbox_to_anchor=(1,1))
#     plt.xlabel("Hours")
#     plt.ylabel("a_k interday")
#     plt.title("Hourly average for k-values for 2015 ")
#     # Lower right figure
#     x_ax = range(len(T_avg_day[0])) # X for year plot
#     plt.subplot(2,2,4)
#     plt.plot(x_ax,T_avg_day[3],label='k=4',color="c")
#     plt.plot(x_ax,T_avg_day[4],label='k=5',color="m")
#     plt.plot(x_ax,T_avg_day[5],label='k=6',color="y")
#     plt.legend(loc='upper left',bbox_to_anchor=(1,1))
#     plt.xlabel("day")
#     plt.ylabel("a_k seasonal")
#     plt.title("daily average for k-values for 2015 ")
#     # Figure title
#     plt.suptitle(file_name,fontsize=20,x=.51,y=0.932) #,x=.51,y=1.07
    
#     return plt.show(all)
# #%%

# #%%
# time_index = network.loads_t.p.index
# season_plot(-T, time_index, 'For the mismatch')


#%%
