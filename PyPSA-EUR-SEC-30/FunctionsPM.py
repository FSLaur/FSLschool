# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:17:29 2023

@author: laur1
"""

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
from calendar import isleap
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# For loading data from network files
def load_ALL(year,pathstart,pathend):
    Columns = list(map(str,year))
    generators = pd.DataFrame()
    storage = pd.DataFrame()
    stores = pd.DataFrame()
    links = pd.DataFrame()
    generators_timeY = pd.DataFrame()
    storage_timeY = pd.DataFrame()
    stores_timeY = pd.DataFrame()
    links_timep0Y = pd.DataFrame()
    links_timep1Y = pd.DataFrame()
    systemcost = pd.Series()
    eprice = pd.DataFrame()
    hydroinflow_timeY = pd.DataFrame()
    loads_timeY = pd.DataFrame()
    eprice_time = pd.DataFrame()

    for i in range(len(year)):
        path = pathstart+str(year[i])+pathend
        n = pypsa.Network()
        n.import_from_hdf5(path,skip_time=False)
        generators[i] = n.generators.p_nom_opt # Optimal installed generator capacity
        generators.rename(columns={i:year[i]},inplace=True)
        generators_time = n.generators_t.p # dispatch generators
        generators_timeY = generators_timeY.append(generators_time)
        storage[i] = n.storage_units.p_nom_opt # Optimal storage nominal power
        storage.rename(columns={i:year[i]},inplace=True)
        storage_time = n.storage_units_t.p # Dispatch or charge storage
        storage_timeY = storage_timeY.append(storage_time)
        stores[i] = n.stores.e_nom_opt # Optimal nominal energy capacity
        stores.rename(columns={i:year[i]},inplace=True)
        stores_time = n.stores_t.e # Dispatch or charge stores
        stores_timeY = stores_timeY.append(stores_time)
        links[i] = n.links.p_nom_opt # Optimal installed link capacity
        links.rename(columns={i:year[i]},inplace=True)
        links_timep0 = n.links_t.p0
        links_timep0Y = links_timep0Y.append(links_timep0)
        links_timep1 = n.links_t.p1
        links_timep1Y = links_timep1Y.append(links_timep1)
        buses = n.buses # Location with more
        loads_time = n.loads_t.p_set #Demand or electricity load
        loads_timeY = loads_timeY.append(loads_time) 
        syscost = pd.Series(n.objective,index = [year[i]]) #Euro
        syscost = syscost/loads_time.sum().sum() # EUR/MWh
        systemcost = systemcost.append(syscost)
        eprice = n.buses_t.marginal_price #EUR/MWh
        eprice_time = eprice_time.append(eprice)
        eprice.rename(columns={i:year[i]},inplace=True)
        hydroinflow_time = n.storage_units_t.inflow # Inflow hydro
        hydroinflow_timeY = hydroinflow_timeY.append(hydroinflow_time)
        
        
    timeindex = pd.DataFrame(index=pd.date_range(str(year[0])+"-01-01", str(year[-1])+'-12-31-23:00:00', freq="h"))
    timeindex =timeindex[~((timeindex.index.month == 2) & (timeindex.index.day == 29))]  # SKIP leap days
    generators_timeY.index=timeindex.index
    storage_timeY.index=timeindex.index
    stores_timeY.index=timeindex.index
    links_timep0Y.index = timeindex.index
    links_timep1Y.index = timeindex.index
    hydroinflow_timeY.index = timeindex.index
    loads_timeY.index = timeindex.index
    eprice_time.index = timeindex.index
    systemcost.rename('SystemCost',inplace = True)
    loads = n.loads_t.p_set
    return n,generators,generators_timeY,storage, storage_timeY,stores,stores_timeY, links,links_timep0Y,links_timep1Y,buses, systemcost,eprice,eprice_time,loads,hydroinflow_timeY,loads_timeY

# Importing CSV files
def importCSV(filepath,year):
    dfname = pd.read_csv(filepath,sep = ';',index_col = 0)
    dfname.index = pd.to_datetime(dfname.index)
    return dfname

# Mean and minimum
def meanYear(df):
    df_mean = df.groupby([pd.Grouper( freq='y')]).mean()
    return df_mean
def meanWeek(df):
    df_mean = df.groupby([pd.Grouper( freq='w')]).mean()
    return df_mean
def meanMonth(df):
    df_mean = df.groupby([pd.Grouper( freq='m')]).mean()
    return df_mean
def minYear(df):
    df_min = df.groupby([pd.Grouper( freq='w')]).mean().groupby([pd.Grouper( freq='y')]).min()
    return df_min
def sumYear(df):
    df_sum = df.groupby([pd.Grouper( freq='y')]).sum()
    return df_sum

# Collecting all countries with more than one to one country
def modGenerators(generators_timeY):
    df_add = pd.DataFrame()
    df_add['DE onwind'] = generators_timeY['DE0 onwind']+generators_timeY['DE1 onwind']+generators_timeY['DE2 onwind']
    df_add['ES onwind'] = generators_timeY['ES0 onwind']+generators_timeY['ES1 onwind']+generators_timeY['ES2 onwind']+generators_timeY['ES3 onwind']
    df_add['FI onwind'] = generators_timeY['FI0 onwind']+generators_timeY['FI1 onwind']+generators_timeY['FI2 onwind']
    df_add['FR onwind'] = generators_timeY['FR0 onwind']+generators_timeY['FR1 onwind']+generators_timeY['FR2 onwind']+generators_timeY['FR3 onwind']
    df_add['GB onwind'] = generators_timeY['GB0 onwind']+generators_timeY['GB1 onwind']
    df_add['IT onwind'] = generators_timeY['IT0 onwind']+generators_timeY['IT1 onwind']+generators_timeY['IT2 onwind']
    df_add['NO onwind'] = generators_timeY['NO0 onwind']+generators_timeY['NO1 onwind']+generators_timeY['NO2 onwind']
    df_add['PL onwind'] = generators_timeY['PL0 onwind']+generators_timeY['PL1 onwind']+generators_timeY['PL2 onwind']
    df_add['RO onwind'] = generators_timeY['RO0 onwind']+generators_timeY['RO1 onwind']
    df_add['SE onwind'] = generators_timeY['SE0 onwind']+generators_timeY['SE1 onwind']+generators_timeY['SE2 onwind']+generators_timeY['SE3 onwind']
    generators_timeYmod = pd.DataFrame()
    generators_timeYmod['AT onwind'] = generators_timeY['AT onwind']
    generators_timeYmod['BA onwind'] = generators_timeY['BA onwind']
    generators_timeYmod['BE onwind'] = generators_timeY['BE onwind']
    generators_timeYmod['BG onwind'] = generators_timeY['BG onwind']
    generators_timeYmod['CH onwind'] = generators_timeY['CH onwind']
    generators_timeYmod['CZ onwind'] = generators_timeY['CZ onwind']
    generators_timeYmod['DE onwind'] = df_add['DE onwind']
    generators_timeYmod['DK onwind'] = generators_timeY['DK onwind']
    generators_timeYmod['EE onwind'] = generators_timeY['EE onwind']
    generators_timeYmod['ES onwind'] = df_add['ES onwind']
    generators_timeYmod['FI onwind'] = df_add['FI onwind']
    generators_timeYmod['FR onwind'] = df_add['FR onwind']
    generators_timeYmod['GB onwind'] = df_add['GB onwind']
    generators_timeYmod['GR onwind'] = generators_timeY['GR onwind']
    generators_timeYmod['HR onwind'] = generators_timeY['HR onwind']
    generators_timeYmod['HU onwind'] = generators_timeY['HU onwind']
    generators_timeYmod['IE onwind'] = generators_timeY['IE onwind']
    generators_timeYmod['IT onwind'] = df_add['IT onwind']
    generators_timeYmod['LT onwind'] = generators_timeY['LT onwind']
    generators_timeYmod['LU onwind'] = generators_timeY['LU onwind']
    generators_timeYmod['LV onwind'] = generators_timeY['LV onwind']
    generators_timeYmod['NL onwind'] = generators_timeY['NL onwind']
    generators_timeYmod['NO onwind'] = df_add['NO onwind']
    generators_timeYmod['PL onwind'] = df_add['PL onwind']
    generators_timeYmod['PT onwind'] = generators_timeY['PT onwind']
    generators_timeYmod['RO onwind'] = df_add['RO onwind']
    generators_timeYmod['RS onwind'] = generators_timeY['RS onwind']
    generators_timeYmod['SE onwind'] = df_add['SE onwind']
    generators_timeYmod['SI onwind'] = generators_timeY['SI onwind']
    generators_timeYmod['SK onwind'] = generators_timeY['SK onwind']
    generators_timeYnew = generators_timeY. iloc[:, 51:]
    generators_timeYmod = pd.concat([generators_timeYmod,generators_timeYnew],axis=1)
    return generators_timeYmod

# Remember generators_timeYmod
def getDroughts(generators_timeY):
    # Droughts calculated for total mean for wind and montly mean for solar
    #List of all column names in generators modified
    c = list(generators_timeY.columns)
    # # onshore list
    # onWindnames = c[0:51]
    # # off shore list
    # offWindnames = c[51:72]
    # # solar list
    # solarnames = c[72:102]
    # onshore list
    onWindnames = c[0:28]
    # off shore list
    offWindnames = c[28:49]
    # solar list
    solarnames = c[49:79]
    
    # weekly mean of on, off wind and solar 
    onWindYweek = meanWeek(generators_timeY[onWindnames])
    offWindYweek = meanWeek(generators_timeY[offWindnames])
    solarYweek = meanWeek(generators_timeY[solarnames])
    solarYweek.drop(['BA solar','RS solar'],axis = 1,inplace=True)
    onWindYweeksystem = onWindYweek.sum(axis=1)
    solarYweeksystem = solarYweek.sum(axis=1)
    
    # For VRE onshore wind + solar = VRE
    solarYweekVRE = meanWeek(generators_timeY[solarnames])
    solarYweekVRE = solarYweekVRE.drop(['BA solar','RS solar'],axis = 1)
    VREheaders =(onWindYweek.columns)
    VREheaders = [s.replace('onwind', 'VRE') for s in VREheaders] # Replacing
    VREYweek = pd.DataFrame()
    VREYweek = np.add(onWindYweek,solarYweekVRE)
    VREYweek.columns = [VREheaders]
    
    
    
    
    
    # yearly mean of on and off wind  
    onWindYyear = meanYear(generators_timeY[onWindnames])
    offWindYyear = meanYear(generators_timeY[offWindnames])
    onWindYyearsystem = onWindYyear.sum(axis=1)
    
    # monthly mean of solar
    solarYmonth = meanMonth(generators_timeY[solarnames])
    solarYmonthsystem = solarYmonth.sum(axis=1)
    solarYmonth.drop(['BA solar','RS solar'],axis = 1,inplace=True)
    # FOr VRE
    solarYmonthVRE = solarYmonth
    
    # Monthle mean onwind
    onWindYmonth = meanMonth(generators_timeY[onWindnames])
    onWindYmonthsystem = onWindYmonth.sum(axis=1)
    
    # mean of mean year for on and off shore wind
    onWindmean = np.mean(onWindYyear) 
    offWindmean = np.mean(offWindYyear)
    onWindmeansystem = np.mean(onWindYyearsystem)
    factor = 0.3
    # Energy droughts wind week below 10 %
    onWindfactor01 = factor*onWindmean
    offWindfactor01 = factor*offWindmean
    onWindsystemfactor01 = factor*onWindmeansystem
    onWindbelow10 = (onWindYweek <=onWindfactor01)
    offWindbelow10 = (offWindYweek<=offWindfactor01)
    onWindsystembelow10 = (onWindYweeksystem <=onWindsystemfactor01).astype(int)
    # count energy droughts pr. year
    onWinddrought = onWindbelow10.groupby([pd.Grouper( freq='y')]).sum()
    offWinddrought = offWindbelow10.groupby([pd.Grouper( freq='y')]).sum()
    onWindsystemdrought = onWindsystembelow10.groupby([pd.Grouper( freq='y')]).sum()
    
    # mean of specific month for all the years
    solarmeanmonths = solarYmonth.groupby([solarYmonth.index.month]).mean()
    solarmeanmonthssystem = solarYmonthsystem.groupby([solarYmonthsystem.index.month]).mean()
    solarfactor01 = factor*solarmeanmonths
    solarsystemfactor01 = factor*solarmeanmonthssystem
    
    # VRE
    solarVREmeanmonths = solarYmonthVRE.groupby([solarYmonthVRE.index.month]).mean()
    solarVREfactor01 = factor*solarVREmeanmonths 
    VREfactor01 = np.add(solarVREfactor01,onWindfactor01)
    VREfactor01.columns = [VREheaders]
    # extract all values from a certain month
    #January
    solarJan = solarYweek.loc[(solarYweek.index.month==1)]
    solarbelow10jan = (solarJan<=solarfactor01.loc[1,:])
    solarsystemJan = solarYweeksystem.loc[(solarYweeksystem.index.month==1)]
    solarsystembelow10jan = (solarsystemJan<=solarsystemfactor01.loc[1])
    # VRE
    VREJan = VREYweek.loc[(VREYweek.index.month==1)]
    VREbelow10jan = (VREJan<=VREfactor01.loc[1,:])
    # february
    solarFeb = solarYweek.loc[(solarYweek.index.month==2)]
    solarbelow10feb = (solarFeb<=solarfactor01.loc[2,:])
    solarsystemFeb = solarYweeksystem.loc[(solarYweeksystem.index.month==2)]
    solarsystembelow10feb = (solarsystemFeb<=solarsystemfactor01.loc[2])
    # VRE
    VREFeb = VREYweek.loc[(VREYweek.index.month==2)]
    VREbelow10feb = (VREFeb<=VREfactor01.loc[2,:])
    # march
    solarMar = solarYweek.loc[(solarYweek.index.month==3)]
    solarbelow10mar = (solarMar<=solarfactor01.loc[3,:])
    solarsystemMar = solarYweeksystem.loc[(solarYweeksystem.index.month==3)]
    solarsystembelow10mar = (solarsystemMar<=solarsystemfactor01.loc[3])
    # VRE
    VREMar = VREYweek.loc[(VREYweek.index.month==3)]
    VREbelow10mar = (VREMar<=VREfactor01.loc[3,:])
    # April
    solarApr = solarYweek.loc[(solarYweek.index.month==4)]
    solarbelow10apr = (solarApr<=solarfactor01.loc[4,:])
    solarsystemApr = solarYweeksystem.loc[(solarYweeksystem.index.month==4)]
    solarsystembelow10apr = (solarsystemApr<=solarsystemfactor01.loc[4])
    # VRE
    VREApr = VREYweek.loc[(VREYweek.index.month==4)]
    VREbelow10apr = (VREApr<=VREfactor01.loc[4,:])
    # May
    solarMay = solarYweek.loc[(solarYweek.index.month==5)]
    solarbelow10may = (solarMay<=solarfactor01.loc[5,:])
    solarsystemMay = solarYweeksystem.loc[(solarYweeksystem.index.month==5)]
    solarsystembelow10may = (solarsystemMay<=solarsystemfactor01.loc[5])
    # VRE
    VREMay = VREYweek.loc[(VREYweek.index.month==5)]
    VREbelow10may = (VREMay<=VREfactor01.loc[5,:])
    #June
    solarJun = solarYweek.loc[(solarYweek.index.month==6)]
    solarbelow10jun = (solarJun<=solarfactor01.loc[6,:])
    solarsystemJun = solarYweeksystem.loc[(solarYweeksystem.index.month==6)]
    solarsystembelow10jun = (solarsystemJun<=solarsystemfactor01.loc[6])
    # VRE
    VREJun = VREYweek.loc[(VREYweek.index.month==6)]
    VREbelow10jun = (VREJun<=VREfactor01.loc[6,:])
    # July
    solarJul = solarYweek.loc[(solarYweek.index.month==7)]
    solarbelow10jul = (solarJul<=solarfactor01.loc[7,:])
    solarsystemJul = solarYweeksystem.loc[(solarYweeksystem.index.month==7)]
    solarsystembelow10jul = (solarsystemJul<=solarsystemfactor01.loc[7])
    # VRE
    VREJul = VREYweek.loc[(VREYweek.index.month==7)]
    VREbelow10jul = (VREJul<=VREfactor01.loc[7,:])
    # August
    solarAug = solarYweek.loc[(solarYweek.index.month==8)]
    solarbelow10aug= (solarAug<=solarfactor01.loc[8,:])
    solarsystemAug = solarYweeksystem.loc[(solarYweeksystem.index.month==8)]
    solarsystembelow10aug = (solarsystemAug<=solarsystemfactor01.loc[8])
    # VRE
    VREAug = VREYweek.loc[(VREYweek.index.month==8)]
    VREbelow10aug = (VREAug<=VREfactor01.loc[8,:])
    # September
    solarSep = solarYweek.loc[(solarYweek.index.month==9)]
    solarbelow10sep= (solarSep<=solarfactor01.loc[9,:])
    solarsystemSep = solarYweeksystem.loc[(solarYweeksystem.index.month==9)]
    solarsystembelow10sep = (solarsystemSep<=solarsystemfactor01.loc[9])
    # VRE
    VRESep = VREYweek.loc[(VREYweek.index.month==9)]
    VREbelow10sep = (VRESep<=VREfactor01.loc[9,:])
    # October
    solarOct = solarYweek.loc[(solarYweek.index.month==10)]
    solarbelow10oct= (solarOct<=solarfactor01.loc[10,:])
    solarsystemOct = solarYweeksystem.loc[(solarYweeksystem.index.month==10)]
    solarsystembelow10oct = (solarsystemOct<=solarsystemfactor01.loc[10])
    # VRE
    VREOct = VREYweek.loc[(VREYweek.index.month==10)]
    VREbelow10oct = (VREOct<=VREfactor01.loc[10,:])
    # November
    solarNov = solarYweek.loc[(solarYweek.index.month==11)]
    solarbelow10nov= (solarNov<=solarfactor01.loc[11,:])
    solarsystemNov = solarYweeksystem.loc[(solarYweeksystem.index.month==11)]
    solarsystembelow10nov = (solarsystemNov<=solarsystemfactor01.loc[11])
    # VRE
    VRENov = VREYweek.loc[(VREYweek.index.month==11)]
    VREbelow10nov = (VRENov<=VREfactor01.loc[11,:])
    #December
    solarDec = solarYweek.loc[(solarYweek.index.month==12)]
    solarbelow10dec= (solarDec<=solarfactor01.loc[12,:])
    solarsystemDec = solarYweeksystem.loc[(solarYweeksystem.index.month==12)]
    solarsystembelow10dec = (solarsystemDec<=solarsystemfactor01.loc[12])
    # VRE
    VREDec = VREYweek.loc[(VREYweek.index.month==12)]
    VREbelow10dec = (VREDec<=VREfactor01.loc[12,:])
    # count energy droughts for a month pr. year
    #Jan
    SolardroughtJan = solarbelow10jan.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtJan = solarsystembelow10jan.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtJan = VREbelow10jan.groupby([pd.Grouper( freq='y')]).sum()
    #Feb
    SolardroughtFeb = solarbelow10feb.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtFeb = solarsystembelow10feb.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtFeb = VREbelow10feb.groupby([pd.Grouper( freq='y')]).sum()
    #Marts
    SolardroughtMar = solarbelow10mar.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtMar = solarsystembelow10mar.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtMar = VREbelow10mar.groupby([pd.Grouper( freq='y')]).sum()
    #April
    SolardroughtApr = solarbelow10apr.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtApr = solarsystembelow10apr.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtApr = VREbelow10apr.groupby([pd.Grouper( freq='y')]).sum()
    # May
    SolardroughtMay = solarbelow10may.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtMay = solarsystembelow10may.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtMay = VREbelow10may.groupby([pd.Grouper( freq='y')]).sum()
    # June
    SolardroughtJun = solarbelow10jun.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtJun = solarsystembelow10jun.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtJun = VREbelow10jun.groupby([pd.Grouper( freq='y')]).sum()
    # July
    SolardroughtJul = solarbelow10jul.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtJul = solarsystembelow10jul.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtJul = VREbelow10jul.groupby([pd.Grouper( freq='y')]).sum()
    # August
    SolardroughtAug = solarbelow10aug.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtAug = solarsystembelow10aug.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtAug = VREbelow10aug.groupby([pd.Grouper( freq='y')]).sum()
    # September
    SolardroughtSep = solarbelow10sep.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtSep = solarsystembelow10sep.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtSep = VREbelow10sep.groupby([pd.Grouper( freq='y')]).sum()
    # October
    SolardroughtOct = solarbelow10oct.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtOct = solarsystembelow10oct.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtOct = VREbelow10oct.groupby([pd.Grouper( freq='y')]).sum()
    # November
    SolardroughtNov = solarbelow10nov.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtNov = solarsystembelow10nov.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtNov = VREbelow10nov.groupby([pd.Grouper( freq='y')]).sum()
    
    # December
    SolardroughtDec = solarbelow10dec.groupby([pd.Grouper( freq='y')]).sum()
    SolarsystemdroughtDec = solarsystembelow10dec.groupby([pd.Grouper( freq='y')]).sum()
    VREdroughtDec = VREbelow10dec.groupby([pd.Grouper( freq='y')]).sum()
    # Total
    Solardrought = SolardroughtJan+SolardroughtFeb+SolardroughtMar+SolardroughtApr+SolardroughtMay+SolardroughtJun+SolardroughtJul+SolardroughtAug+SolardroughtSep+SolardroughtOct+SolardroughtNov+SolardroughtDec
    Solarsystemdrought = (SolarsystemdroughtJan+SolarsystemdroughtFeb+SolarsystemdroughtMar+SolarsystemdroughtApr+SolarsystemdroughtMay+SolarsystemdroughtJun+SolarsystemdroughtJul+SolarsystemdroughtAug+SolarsystemdroughtSep+SolarsystemdroughtOct+SolarsystemdroughtNov+SolarsystemdroughtDec).astype(int)
    VREdrought = (VREdroughtJan+VREdroughtFeb+VREdroughtMar+VREdroughtApr+VREdroughtMay+VREdroughtJun+VREdroughtJul+VREdroughtAug+VREdroughtSep+VREdroughtOct+VREdroughtNov+VREdroughtDec).astype(int)
    
    return onWinddrought, onWindsystemdrought,offWinddrought,Solardrought,Solarsystemdrought, VREdrought,onWindfactor01, onWindsystemfactor01,offWindfactor01, solarfactor01,solarsystemfactor01, VREfactor01, onWindYweek, onWindYweeksystem,offWindYweek, solarYweek, solarYweeksystem,VREYweek, solarYmonth, solarYmonthsystem, onWindYmonth, onWindYmonthsystem

def getDroughtsweek(generators_timeYmod):
    # Droughts calculated for weekly means for all the years
    #List of all column names in generators modified
    c = list(generators_timeYmod.columns)
    # # onshore list
    # onWindnames = c[0:51]
    # # off shore list
    # offWindnames = c[51:72]
    # # solar list
    # solarnames = c[72:102]
    # onshore list
    onWindnames = c[0:28]
    # off shore list
    offWindnames = c[28:49]
    # solar list
    solarnames = c[49:79]
    
    # weekly mean of on, off wind and solar 
    onWindYweek = meanWeek(generators_timeYmod[onWindnames])
    offWindYweek = meanWeek(generators_timeYmod[offWindnames])
    solarYweek = meanWeek(generators_timeYmod[solarnames])
    solarYweek = solarYweek.drop(['BA solar','RS solar'],axis = 1)
    # Onshore wind, solar = VRE
    VREheaders =(onWindYweek.columns)
    VREheaders = [s.replace('onwind', 'VRE') for s in VREheaders] # Replacing
    VREYweek = pd.DataFrame()
    VREYweek = np.add(onWindYweek,solarYweek)
    VREYweek.columns = [VREheaders]
    
    # Weekly mean for specific week for all year
    onWindYmeanweek = onWindYweek.groupby([onWindYweek.index.week]).mean()
    offWindYmeanweek = offWindYweek.groupby([offWindYweek.index.week]).mean()
    solarYmeanweek = solarYweek.groupby([solarYweek.index.week]).mean()
    VREYmeanweek = VREYweek.groupby([VREYweek.index.week]).mean()
       
    # Energy droughts wind and solar week and VRE below 10 %
    onWindfactor01week = 0.1*onWindYmeanweek
    onWindfactor01week.reset_index(drop=True, inplace=True)
    offWindfactor01week = 0.1*offWindYmeanweek
    offWindfactor01week.reset_index(drop=True, inplace=True)
    solarfactor01week = 0.1*solarYmeanweek
    solarfactor01week.reset_index(drop=True, inplace=True)
    VREfactor01week = 0.1*VREYmeanweek
    VREfactor01week.reset_index(drop=True, inplace=True)
    
    # Initializing
    onWinddroughtloop = pd.DataFrame()
    offWinddroughtloop = pd.DataFrame()
    solardroughtloop = pd.DataFrame()
    VREdroughtloop = pd.DataFrame()
    # Loop for checking and appending droughts
    for i in range(53):
        # Onwind
        onWindweekloop = onWindYweek.loc[(onWindYweek.index.week==i)]
        onWindbelow10loop = (onWindweekloop<=onWindfactor01week.loc[i,:])
        onWinddroughtloop = onWinddroughtloop.append(onWindbelow10loop)
        onWinddroughtloop.sort_index(inplace=True)
        
        # Offwind
        offWindweekloop = offWindYweek.loc[(offWindYweek.index.week==i)]
        offWindbelow10loop = (offWindweekloop<=offWindfactor01week.loc[i,:])
        offWinddroughtloop = offWinddroughtloop.append(offWindbelow10loop)
        offWinddroughtloop.sort_index(inplace=True)
        
        # Solar
        solarweekloop = solarYweek.loc[(solarYweek.index.week==i)]
        solarbelow10loop = (solarweekloop<=solarfactor01week.loc[i,:])
        solardroughtloop = solardroughtloop.append(solarbelow10loop)
        solardroughtloop.sort_index(inplace=True)
        
        # VRE
        VREweekloop = VREYweek.loc[(VREYweek.index.week==i)]
        VREbelow10loop = (VREweekloop<=VREfactor01week.loc[i,:])
        VREdroughtloop = VREdroughtloop.append(VREbelow10loop)
        VREdroughtloop.sort_index(inplace=True)
    
        
    onWinddroughtloopsum = onWinddroughtloop.groupby([pd.Grouper( freq='y')]).sum()    
    offWinddroughtloopsum = offWinddroughtloop.groupby([pd.Grouper( freq='y')]).sum()
    solardroughtloopsum = solardroughtloop.groupby([pd.Grouper( freq='y')]).sum()    
    VREdroughtloopsum = VREdroughtloop.groupby([pd.Grouper(freq='y')]).sum()
    return onWinddroughtloopsum, offWinddroughtloopsum, solardroughtloopsum, VREdroughtloopsum

# minimum average monthly capacity factor calculation
def getminCFmonth(CF,loads):
    CF = CF.loc[:'2015-12-31 23:00:00+00:00']
    #if set(['BIH','SRB']).issubset(CF.columns):
    #    CF = CF.drop(['BIH','SRB'],axis=1)
    ##else:
    #    CF = CF
    CF.columns=loads.columns
    CFmeanmonth = CF.groupby([pd.Grouper( freq='m')]).mean()
    CFtotalmeanmonth_all = pd.DataFrame()
    CFminfactor_all = pd.DataFrame()
    CFstdmonth_all = pd.DataFrame()
    for i in range(1,13):
        CFtotalmeanmonth = CFmeanmonth.loc[(CFmeanmonth.index.month==i)].mean(axis=0)
        CFstdmonth = CFmeanmonth.loc[(CFmeanmonth.index.month==i)].std(axis=0)
        CFtotalmeanmonth_all = CFtotalmeanmonth_all.append(CFtotalmeanmonth,ignore_index=True)
        CFstdmonth_all = CFstdmonth_all.append(CFstdmonth,ignore_index=True)
    for i in range(12):
        CFminfactor = CFtotalmeanmonth_all.loc[i]-CFmeanmonth.loc[(CFmeanmonth.index.month==i+1)]
        CFminfactor_all = CFminfactor_all.append(CFminfactor)
        CFminfactor_all.sort_index(inplace=True)
    CFminfactor_year = CFminfactor_all.groupby([pd.Grouper( freq='y')]).max()                  
    return CFmeanmonth, CFtotalmeanmonth_all, CFminfactor_all, CFminfactor_year, CFstdmonth_all, CF

# Capacity factor seasonal
def getSeasonalCF(CF,loads):
    CF = CF.loc[:'2015-12-31 23:00:00+00:00']
    #if set(['BIH','SRB']).issubset(CF.columns):
    #    CF = CF.drop(['BIH','SRB'],axis=1)
    #else:
    #    CF = CF
    CF.columns=loads.columns
    winter = [1,2,3,10,11,12]
    summer = [4,5,6,7,8,9]
    CFwinter_all = pd.DataFrame()
    CFsummer_all = pd.DataFrame()
    CFmeanmonth = CF.groupby([pd.Grouper( freq='m')]).mean()
    for (i,j) in zip(winter,summer):
        CFwinter = CFmeanmonth.loc[(CFmeanmonth.index.month==i)]
        CFsummer = CFmeanmonth.loc[(CFmeanmonth.index.month==j)]
        CFwinter_all = CFwinter_all.append(CFwinter,ignore_index=False)
        CFsummer_all = CFsummer_all.append(CFsummer,ignore_index=False)
    CFseasonalratio = CFsummer_all.groupby([pd.Grouper(freq='y')]).mean()/CFwinter_all.groupby([pd.Grouper(freq='y')]).mean()
    CFseasonalratio.fillna(0,inplace=True)
    return CFwinter_all, CFsummer_all, CFseasonalratio

# Function which saves figures
# NAMES list of country names, constraint and name of figure fx blabla.png
def savefigure(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = r"D:\Master\Pictures"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')
    #fig.clear()
def savefigure1(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = r"D:\Master\Pictures\TRMSX2_95Red"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')
    #fig.clear()
def savefigure2(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = r"D:\Master\Pictures\TRMSX0_95Red"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')
    #fig.clear()
def savefigure3(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = r"D:\Master\Pictures\TRMSX2_95Red2.0"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')
    #fig.clear()        
    
def savefigure4(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = r"D:\Master\Pictures\TRMSX0_95Red2.0"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')
# getting names of different link generators MW
def getGenandstores(links,stores,generators,eprice):
    link_names = list(links.index) 
    OCGT_names = link_names[0:30]
    H2_Enames = link_names[30:60] # Electrolysis
    H2_FCnames = link_names[60:90] # Fuel Cell
    Bat_Chnames = link_names[90:120] # charger
    Bat_DChnames = link_names[120:150] # Discharger
    store_names = list(stores.index)
    GAS_storenames = store_names[0:30]
    H2_storenames = store_names[30:60]
    Bat_storenames = store_names[60:90]
    
    # Transposing dataframes
    links2 = links.T
    stores2 = stores.T
    # Creating new dataframes for link generators
    OCGT_Y = links2[OCGT_names]
    H2_EY = links2[H2_Enames]
    H2_FCY = links2[H2_FCnames]
    Bat_ChY =  links2[Bat_Chnames]
    Bat_DChY = links2[Bat_DChnames]
    
    # Creating new dataframes for stores
    GAS_storeY = stores2[GAS_storenames]
    H2_storeY = stores2[H2_storenames]
    Bat_storeY = stores2[Bat_storenames]
    
    # For renewable generators
    generators2 = generators.T
    # moderated generators
    generators2Mod = modGenerators(generators2)
    generators_names = list(generators2Mod.columns)
    onWindgen_names = generators_names[0:28]
    offWindgen_names = generators_names[28:49]
    solargen_names = generators_names[49:79]
    rorgen_names = generators_names[79:106]
    #renewables generators
    onWindgen = generators2Mod[onWindgen_names]
    offWindgen = generators2Mod[offWindgen_names]
    solargen = generators2Mod[solargen_names]
    rorgen = generators2Mod[rorgen_names]
    
    # dropping some countries for solar so onwind and solar can be plotted together
    OCGTgen = OCGT_Y.drop(['BA OCGT','RS OCGT'],axis=1)
    H2_Egen = H2_EY.drop(['BA H2 Electrolysis','RS H2 Electrolysis'],axis=1)
    H2_FCgen = H2_FCY.drop(['BA H2 Fuel Cell','RS H2 Fuel Cell'],axis=1)
    Bat_genCh = Bat_ChY.drop(['BA battery charger','RS battery charger'],axis=1)
    Bat_genDCh = Bat_DChY.drop(['BA battery discharger','RS battery discharger'],axis=1)
    solargen = solargen.drop(['BA solar','RS solar'],axis=1) 
    
    # Stores dropping some countries 
    GAS_storeY = GAS_storeY.drop(['BA gas Store','RS gas Store'],axis=1)
    H2_storeY = H2_storeY.drop(['BA H2 Store','RS H2 Store'],axis=1)
    Bat_storeY = Bat_storeY.drop(['BA battery','RS battery'],axis=1)
    
    # Prices
    eprice2 = eprice.T
    eprices_names = list(eprice2.columns)
    e_price_country_names = eprices_names[0:30]
    e_price_country = eprice2[e_price_country_names]
    e_price_countryY = e_price_country.drop(['BA','RS'],axis=1)
    
    # Transmission
    linksnames = list(links.index)
    transmissionnames = linksnames[150:]
    transmission =  links.loc[transmissionnames]
    # system
    systemdf = pd.DataFrame()
    systemdf['onWind'] = onWindgen.sum(axis=1)
    systemdf['solar'] = solargen.sum(axis=1)
    systemdf['OCGT'] = OCGTgen.sum(axis=1)
    systemdf['H2 elec'] = H2_Egen.sum(axis=1)
    systemdf['H2 Fuel'] = H2_FCgen.sum(axis=1)
    systemdf['Bat charge'] = Bat_genCh.sum(axis=1)
    systemdf['Bat discharge'] = Bat_genDCh.sum(axis=1)
    systemdf['GAS store'] = GAS_storeY.sum(axis=1)
    systemdf['H2 store'] = H2_storeY.sum(axis=1)
    systemdf['Bat store'] = Bat_storeY.sum(axis=1)
    systemdf['Transmission'] = transmission.sum()
    
    return onWindgen,offWindgen,solargen,OCGTgen,H2_Egen,H2_FCgen,Bat_genCh,Bat_genDCh,GAS_storeY,H2_storeY,Bat_storeY,rorgen,eprice2,e_price_countryY, systemdf, transmission    

# Function which creates folders
# Input should names of countries fx e_price_countryY.columns or NAMES
# If system folder ['SYSTEM']
# Constraint is string defined in top of document
def folders(NAMES,constraint):
    import os
    pathlist = list(NAMES)
    for i in pathlist:
        # specify the path for the directory – make sure to surround it with quotation marks
        root_path = r"D:\Pre-project (Data)\Pictures"
        path = os.path.join(root_path, i+constraint)
    
    
        # create new single directory
        os.mkdir(path)


# Change index of columns
def changeindex(df,index):
    df.reset_index(drop=True,inplace=True)
    df.set_axis(index,inplace=True)
    return df

def LinRegCountryPlot(x_vec, y1_vec, y2_vec, NAMES,color1,label1,label2,xlabel,y1label,y2label,constraint,constraint2,figurename):
    for (i,j,k,l) in zip(x_vec.columns,y1_vec.columns,y2_vec.columns,NAMES):
        fig, (ax1,ax2) = plt.subplots(2,figsize=(12,6))
        # Top plot
        ax1.scatter(x_vec[i],y1_vec[j],color=color1,label=label1)
        X = x_vec[i].values.reshape(len(x_vec[i]),1)
        y1 = y1_vec[j].values.reshape(len(y1_vec[j]),1)
        reg = LinearRegression().fit(X,y1)
        ax1.plot(X,reg.predict(X),color='black',lw=1.5,label='LinReg')
        ax1.legend(loc='upper right')
        alpha = reg.coef_
        bias = reg.intercept_
        R2 = reg.score(X,y1)
        std_dev = np.std(y1_vec[j])
        textstr = '\n'.join((
            r'$\alpha=%.2f$' % (alpha, ),
            r'$\beta=%.2f$' % (bias, ),
            r'$R^2=%.2f$' % (R2, ),
            r'$\sigma=%.2f$'% (std_dev, )))
        ax1.text(0.01, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top')
        ax1.set_title(l+ ' ' + constraint2,fontsize=18)
        ax1.set_xlabel(xlabel,color="black",fontsize=14)
        ax1.set_ylabel(y1label,color="black",fontsize=14)
        # Bottom plot
        ax2.scatter(x_vec[i],y2_vec[k],color=color1,label=label2)
        y2 = y2_vec[k].values.reshape(len(y2_vec[k]),1)
        reg = LinearRegression().fit(X,y2)
        ax2.plot(X,reg.predict(X),color='black',lw=1.5,label='LinReg')
        ax2.legend(loc='upper right')
        alpha = reg.coef_
        bias = reg.intercept_
        R2 = reg.score(X,y2)
        std_dev = np.std(y2_vec[k])
        textstr = '\n'.join((
            r'$\alpha=%.2f$' % (alpha, ),
            r'$\beta=%.2f$' % (bias, ),
            r'$R^2=%.2f$' % (R2, ),
            r'$\sigma=%.2f$'% (std_dev, )))
        ax2.text(0.01, -0.5, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top')
        ax2.set_title(l+ ' '+ constraint2,fontsize=18)
        ax2.set_xlabel(xlabel,color="black",fontsize=14)
        ax2.set_ylabel(y2label,color="black",fontsize=14)
        fig.tight_layout()
        savefigure(l,constraint,figurename,fig)
        
def LinRegSystemPlot(x_vec, y1_vec, y2_vec,color1,label1,label2,xlabel,y1label,y2label,constraint,constraint2,figurename):
    fig, (ax1,ax2) = plt.subplots(2,figsize=(12,6))
    # Top plot
    ax1.scatter(x_vec,y1_vec,color=color1,label=label1)
    X = x_vec.values.reshape(len(x_vec),1)
    y1 = y1_vec.values.reshape(len(y1_vec),1)
    reg = LinearRegression().fit(X,y1)
    ax1.plot(X,reg.predict(X),color='black',lw=1.5,label='LinReg')
    ax1.legend(loc='upper right')
    alpha = reg.coef_
    bias = reg.intercept_
    R2 = reg.score(X,y1)
    std_dev = np.std(y1_vec)
    textstr = '\n'.join((
        r'$\alpha=%.2f$' % (alpha, ),
        r'$\beta=%.2f$' % (bias, ),
        r'$R^2=%.2f$' % (R2, ),
        r'$\sigma=%.2f$'% (std_dev, )))
    ax1.text(0.01, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top')
    ax1.set_title('System'+ ' ' + constraint2,fontsize=18)
    ax1.set_xlabel(xlabel,color="black",fontsize=14)
    ax1.set_ylabel(y1label,color="black",fontsize=14)
    # Bottom plot
    ax2.scatter(x_vec,y2_vec,color=color1,label=label2)
    y2 = y2_vec.values.reshape(len(y2_vec),1)
    reg = LinearRegression().fit(X,y2)
    ax2.plot(X,reg.predict(X),color='black',lw=1.5,label='LinReg')
    ax2.legend(loc='upper right')
    alpha = reg.coef_
    bias = reg.intercept_
    R2 = reg.score(X,y2)
    std_dev = np.std(y2_vec)
    textstr = '\n'.join((
        r'$\alpha=%.2f$' % (alpha, ),
        r'$\beta=%.2f$' % (bias, ),
        r'$R^2=%.2f$' % (R2, ),
        r'$\sigma=%.2f$'% (std_dev, )))
    ax2.text(0.01, -0.5, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top')
    ax2.set_title('System'+ ' '+ constraint2,fontsize=18)
    ax2.set_xlabel(xlabel,color="black",fontsize=14)
    ax2.set_ylabel(y2label,color="black",fontsize=14)
    fig.tight_layout()
    savefigure('SYSTEM',constraint,figurename,fig)
    

def PCA(X,NAMES):
    # Input: 
    # Data matrix, X
    # Names of countries: NAMES
    # Output:
    # Eigenvalues and eigen vectors (PC's)
    # Dataframe with eigenvectors - VT
    # Variance explained - Variance explained
    # Normalization or standardization vector - c
    # amplitudes of eigenvectors - a_k
    # Covariance matrix - Cov_mat
    X.columns = NAMES
    X_mean = X.mean(axis=0)
    X_cent = X-X_mean                                         # Centering data
    c = 1/np.sqrt(np.sum((X_cent)**2)/len(X))                 # Standardization constant (1/sqrt(Var))
    X_norm = c*X_cent                                         # Standardization
    Cov_mat = np.cov(X_norm.T,bias=True)                      # Co-variance matrix
    eigen_values, eigen_vectors = np.linalg.eig(Cov_mat)      # Eigen values and eigen vectors 
    #a_k = np.dot(X_norm,eigen_vectors)                        # Amplitudes of eigenvectors
    variance_explained = (eigen_values/eigen_values.sum())*100
    VT = pd.DataFrame(data=eigen_vectors, index=NAMES)
    return (eigen_values, eigen_vectors, Cov_mat, c, variance_explained, VT, X_norm)

def PCAnoconstant(X,NAMES):
    # Input: 
    # Data matrix, X
    # Names of countries: NAMES
    # Output:
    # Eigenvalues and eigen vectors (PC's)
    # Dataframe with eigenvectors - VT
    # Variance explained - Variance explained
    # Normalization or standardization vector - c
    # amplitudes of eigenvectors - a_k
    # Covariance matrix - Cov_mat
    X.columns = NAMES
    X_mean = X.mean(axis=0)
    X_cent = X-X_mean                                         # Centering data
    c = 1/np.sqrt((np.sum((X_cent)**2)/len(X)).sum())                # Standardization constant (1/sqrt(Var))
    #c = 1
    X_norm = c*X_cent                                         # Standardization
    Cov_mat = np.cov(X_norm.T,bias=True)                      # Co-variance matrix
    eigen_values, eigen_vectors = np.linalg.eig(Cov_mat)      # Eigen values and eigen vectors 
    #a_k = np.dot(X_norm,eigen_vectors)                        # Amplitudes of eigenvectors
    variance_explained = (eigen_values/eigen_values.sum())*100
    VT = pd.DataFrame(data=eigen_vectors, index=NAMES)
    return (eigen_values, eigen_vectors, Cov_mat, c, variance_explained, VT, X_norm)

def MAPPLOT(VT,constraint3,variance_explained,yearstring):
    # Input: Eigenvectors in dataframe, constraint e.g. 2XTRMS, yearstring is a string with the year 
    # Output: Mapplot of 4 first PC's
    fig, ax = plt.subplots(figsize=(17, 4), nrows=1, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()})
    linewidth = 0.8
    panels = ['(a)', '(b)', '(c)', '(d)']
    for i in range(4):
        ax[i].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=linewidth)
        ax[i].coastlines(resolution='110m')
        ax[i].add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.30)
        ax[i].set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
        europe_not_included = {'AD','AL','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                                'ME','MK','MT','RU','SM','UA','VA','XK'}
        shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        countries_1 = reader.records()
        name_loop = 'start'
        PC_NO = i+1
        for country in countries_1:
            if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
                if country.attributes['NAME'] == 'Norway':
                    name_loop = 'NO'
                elif country.attributes['NAME'] == 'France':
                    name_loop = 'FR'                
                else:
                    name_loop = country.attributes['ISO_A2']
                for country_PSA in VT.index.values:
                    if country_PSA == name_loop:
                        color_value = VT.loc[country_PSA][PC_NO-1]
                        if color_value <= 0:
                            color_value = np.absolute(color_value)
                            ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(1, 0, 0), 
                                                  alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                        else:
                            color_value = np.absolute(color_value)
                            ax[i].add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=linewidth, facecolor=(0, 0, 1), 
                                                  alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
            else:
                ax[i].add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=linewidth, 
                                      edgecolor="black", label=country.attributes['ADM0_A3'])

        ax[i].text(0.018, 0.92, panels[i], fontsize=15.5, transform=ax[i].transAxes,weight ='bold');
        ax[i].text(0.026, 0.84, r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%', 
                    fontsize=12, transform=ax[i].transAxes, weight = 'bold');

    cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0.333,0.333),(1,0.666,0.666),'white',(0.666,0.666,1),(0.333,0.333,1),(0,0,1)])
    shrink = 0.08
    ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cbar = ax1.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(hspace=0.02, wspace=0.04)
    fig.suptitle(yearstring+' - '+constraint3,fontsize = 18, weight = 'bold')
    return fig



def season_plot(a_k, time_index, constraint3,yearstring,a_k_avgdailymax,a_k_avgdailymin,a_k_avghour):
    # Input: 
        # a_k = amplitudes of eigenvectors
        # time_index = time_index
        # constraint3 = transmission constraint
        # yearstring = string of the year of data
        # a_k_avgdailymax = max of all 6 years of interest
        # a_k_avgdailymin = min of all 6 years of interest
        # a_k_avghour = max and min of all years
    # Output: 
        # diurnal and seasonal plots
    T = pd.DataFrame(data=a_k,index=time_index)
    T_avg_hour = T.groupby(time_index.hour).mean() # Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day

    # left figure figure
    fig, ax = plt.subplot_mosaic([['left','upper right'],['left','center right'],['left','bottom right']],figsize=(12,6))
    ax['left'].plot(T_avg_hour[0],label='k=1',color='b')
    ax['left'].plot(T_avg_hour[1],label='k=2',color='orange')
    ax['left'].plot(T_avg_hour[2],label='k=3',color = 'green',alpha=0.5)
    ax['left'].plot(T_avg_hour[3],label='k=4',color="c",alpha=0.2)
    ax['left'].plot(T_avg_hour[4],label='k=5',color="m",alpha=0.2)
    ax['left'].plot(T_avg_hour[5],label='k=6',color="y",alpha=0.2)
    ax['left'].set_xticks(ticks=range(0,24,2))
    ax['left'].legend(loc='lower right')
    ax['left'].set_xlabel("Hours", weight='bold')
    ax['left'].set_ylabel("a$_k$ diurnal", weight='bold')
    ax['left'].set_ylim((a_k_avghour[1]-0.5,a_k_avghour[0]+0.5))
    ax['left'].set_title("Hourly average for k-values for"+' '+yearstring,weight='bold')
    
    # right side figures
    x_ax = range(0,len(T_avg_day[0]),1) # X for year plot
    ax['upper right'].plot(x_ax,T_avg_day[0],label='k=1', color = 'b')
    ax['upper right'].set_title('daily average for k = 1', weight = 'bold')
    ax['upper right'].set_ylabel("a$_k$ seasonal",weight='bold')
    ax['upper right'].tick_params(axis='x',labelbottom=False)
    ax['upper right'].set_ylim((a_k_avgdailymin[0]-0.5,a_k_avgdailymax[0]+0.5))
    ax['center right'].plot(x_ax,T_avg_day[1],label='k=2', color = 'orange')
    ax['center right'].set_title('daily average for k = 2', weight = 'bold')
    ax['center right'].set_ylabel("a$_k$ seasonal",weight='bold')
    ax['center right'].tick_params(axis='x',labelbottom=False)
    ax['center right'].set_ylim((a_k_avgdailymin[1]-0.5,a_k_avgdailymax[1]+0.5))
    ax['bottom right'].plot(x_ax,T_avg_day[2],label='k=3', color = 'green',alpha=0.5)
    ax['bottom right'].plot(x_ax,T_avg_day[3],label='k=4',color="c", alpha = 0.2)
    ax['bottom right'].plot(x_ax,T_avg_day[4],label='k=5',color="m", alpha = 0.2)
    ax['bottom right'].plot(x_ax,T_avg_day[5],label='k=6',color="y", alpha = 0.2)
    ax['bottom right'].set_title('daily average for k = 3-6', weight = 'bold')
    ax['bottom right'].set_ylabel("a$_k$ seasonal",weight='bold')
    ax['bottom right'].set_ylim((a_k_avgdailymin[2]-0.5,a_k_avgdailymax[2]+0.5))
    #plt.legend(loc='upper right',bbox_to_anchor=(1.22,1.05),ncol=1)
    ax['bottom right'].set_xlabel("Days",weight='bold')
    fig.suptitle("Hourly average for k-values for"+ ' '+yearstring)

    # Figure title
    plt.suptitle(yearstring+' - '+constraint3,fontsize=18, weight='bold') #,x=.51,y=1.07
    fig.tight_layout()
    return fig

def FFT_plot(T, constraint3,yearstring):
    """
    Parameters
    ----------
    T : Matrix
        Principle component amplitudes. Given by: B*eig_val (so the centered and scaled data dotted with the eigen values)
    constraint3: transmission scenario
    yearstring: string with year working with

    Returns
    -------
    Plot all
    """
    # Frequency calc.
    N = len(T[0])
    n = np.arange(N) # array of samplings counter 1 to 8760
    sr = 1/(3600) # Sampling rate (once an hour)
    T_period = N/sr # Sampling period
    freq = n/T_period
    # Onsided freqency
    n_oneside = N//2
    f_oneside = freq[:n_oneside]
    # Convert frequency to hour
    t_h = 1/f_oneside / (60 * 60)
    # Start plot for FFT
    fig, ax = plt.subplots(3,1,figsize=(12,6))
    # Top plot k=1
    FFT=np.fft.fft(T[0]) # FFT PC's
    FFT=abs(FFT[:n_oneside]/n_oneside)
    FFT = FFT/max(FFT)
    ax[0].plot(t_h,FFT,color='b',label='k=1')
    ax[0].set_xscale('log')
    ax[0].vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[0].vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[0].vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[0].vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[0].vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[0].legend(loc='upper left',fontsize = 14)
    ax[0].text(10,0.9,"1/2 Day",ha='right',weight = 'bold')
    ax[0].text(22,0.9,"Day",ha='right',weight = 'bold')
    ax[0].text(22*7,0.9,"Week",ha='right',weight = 'bold')
    ax[0].text(22*7*4,0.9,"Month",ha='right',weight = 'bold')
    ax[0].text(22*365,0.9,"Year",ha='right',weight = 'bold')
    #ax[0].set_xlabel('Hours', weight= 'bold',fontsize = 12)
    ax[0].set_title('One-sided Fourier Power Spectra for PC1 amplitudes', fontsize = 14, weight ='bold')
    ax[0].tick_params(axis='x',labelbottom=False)
    ax[0].tick_params(axis='y',labelsize=12)
   
    # Center plot k = 2
    FFT=np.fft.fft(T[1]) # FFT PC's
    FFT=abs(FFT[:n_oneside]/n_oneside)
    FFT = FFT/max(FFT)
    ax[1].plot(t_h,FFT,color='orange',label='k=2')
    ax[1].set_xscale('log')
    ax[1].vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[1].vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[1].vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[1].vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[1].vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[1].legend(loc='upper left',fontsize = 14)
    ax[1].text(10,0.9,"1/2 Day",ha='right',weight = 'bold')
    ax[1].text(22,0.9,"Day",ha='right',weight = 'bold')
    ax[1].text(22*7,0.9,"Week",ha='right',weight = 'bold')
    ax[1].text(22*7*4,0.9,"Month",ha='right',weight = 'bold')
    ax[1].text(22*365,0.9,"Year",ha='right',weight = 'bold')
    #ax[0].set_xlabel('Hours', weight= 'bold',fontsize = 12)
    ax[1].set_title('One-sided Fourier Power Spectra for PC2 amplitudes', fontsize = 14, weight ='bold')
    ax[1].tick_params(axis='x',labelbottom=False)
    ax[1].tick_params(axis='y',labelsize=12)
    
    # bottom plot k = 3-6
    FFT=np.fft.fft(T[2]) # FFT PC's
    FFT=abs(FFT[:n_oneside]/n_oneside)
    FFT = FFT/max(FFT)
    ax[2].plot(t_h,FFT,color='green',label='k=3',alpha=0.5)
    FFT=np.fft.fft(T[3]) # FFT PC's
    FFT=abs(FFT[:n_oneside]/n_oneside)
    FFT = FFT/max(FFT)
    ax[2].plot(t_h,FFT,color='c',label='k=4',alpha=0.2)
    FFT=np.fft.fft(T[3]) # FFT PC's
    FFT=abs(FFT[:n_oneside]/n_oneside)
    FFT = FFT/max(FFT)
    ax[2].plot(t_h,FFT,color='m',label='k=5',alpha=0.2)
    FFT=np.fft.fft(T[5]) # FFT PC's
    FFT=abs(FFT[:n_oneside]/n_oneside)
    FFT = FFT/max(FFT)
    ax[2].plot(t_h,FFT,color='y',label='k=6',alpha=0.2)
    ax[2].set_xscale('log')
    ax[2].vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[2].vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[2].vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[2].vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[2].vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2)
    ax[2].legend(loc='upper left',fontsize = 10)
    ax[2].text(10,0.9,"1/2 Day",ha='right',weight = 'bold')
    ax[2].text(22,0.9,"Day",ha='right',weight = 'bold')
    ax[2].text(22*7,0.9,"Week",ha='right',weight = 'bold')
    ax[2].text(22*7*4,0.9,"Month",ha='right',weight = 'bold')
    ax[2].text(22*365,0.9,"Year",ha='right',weight = 'bold')
    ax[2].set_xlabel('Hours', weight= 'bold',fontsize = 12)
    ax[2].set_title('One-sided Fourier Power Spectra for PC3-6 amplitudes', fontsize = 14, weight ='bold')
    ax[2].tick_params(axis='x',labelsize=12)
    ax[2].tick_params(axis='y',labelsize=12)
           
    
    plt.subplots_adjust(wspace=0, hspace=0.28)
    plt.suptitle(yearstring+' - '+constraint3,fontsize=18,weight = 'bold') #,x=.51,y=1.07
    fig.tight_layout()
    return fig

def screeplot(variance_explained, constraint3,yearstring):
    """
    Parameters
    ----------
    variance_explained : variance explained, (eigen_values relative)
        DESCRIPTION.
    constraint3 :  Transmission case
        DESCRIPTION.
    yearstring : Year of data string format
        DESCRIPTION.

    Returns
    -------
    Scree plot

    """
    fig, ax = plt.subplots(1,1,figsize=(12,4))
    ax.plot(np.arange(1,len(variance_explained)+1),np.cumsum(variance_explained),'o-', lw = 2, color = 'darkblue')
    ax.set_xlabel('Principal Component', fontsize=14, weight='bold')
    ax.set_ylabel('Variance explained [%]', fontsize=14, weight='bold')
    ax.tick_params(axis = 'both',labelsize = 12)
    ax.set_xticks(np.arange(1,len(variance_explained)+1))
    ax.set_yticks(np.arange(0,110,20))
    ax.set_title(yearstring+' - '+constraint3, fontsize = 20, weight='bold')
    return fig

def AmplitudeArea(a_kdf):
    """
    # for PC1

    Parameters
    ----------
    a_kdf : Data Frame
        PC's Amplitude'

    Returns
    -------
    Area : Area of first PC for consective day above threshold.
    idx : Int
        max index.
    MaxArea : Data Frame
        Max area
        Consective days above threshold
        Binary 1 if above threshold.
    a_kday: Dataframe
        Daily average amplitudes

    """
    a_kday = a_kdf.groupby([pd.Grouper( freq='d')]).mean() # Day
    x_ax = range(0,len(a_kday[0]),1) # X for year plot
    a_kday.index = x_ax
    
    count = 0 # Initializing
    area = 0
    binary = 0
    area2 = 0
    Area = pd.DataFrame(columns=['Binary','Count','Area','AreaStep'],index=a_kday.index) # DataFrame
    threshold = 0 # This is equal to the mean for the centered data
    
    # Checking if data point is above threshold otherwise zero
    a_kabove = a_kday.where(a_kday>threshold,other=0)
    for i in range(len(a_kabove[0])-1):
        if (a_kabove[0][i] > 0) & (a_kabove[0][i+1]>0):
            count +=1
            binary = 1
            area = (a_kabove.index[i+1]-a_kabove.index[i])*(a_kabove[0][i]+a_kabove[0][i+1])/2+area # integration, midpoint rule
            area2 = (a_kabove.index[i+1]-a_kabove.index[i])*(a_kabove[0][i]+a_kabove[0][i+1])/2 # For summation
        else:
            count  = 0
            area = 0
            area2 = 0
            binary = 0
        Area['Binary'][i] = binary
        Area['Count'][i] = count
        Area['Area'][i] = area
        Area['AreaStep'][i] = area2
    Area.fillna(0, inplace=True)
    idx = pd.Series(Area['Area']).idxmax()
    MaxArea = Area.loc[idx]
    return (Area, idx, MaxArea, a_kday)

def AmplitudeAreaPC1PC2(a_kdf):
    """
    # for PC1+PC2

    Parameters
    ----------
    a_kdf : Data Frame
        PC's Amplitude'

    Returns
    -------
    Area : Area of first PC for consective day above threshold.
    idx : Int
        max index.
    MaxArea : Data Frame
        Max area
        Consective days above threshold
        Binary 1 if above threshold.
    a_kday: Dataframe
        Daily average amplitudes

    """
    a_kday = a_kdf.groupby([pd.Grouper( freq='d')]).mean() # Day
    x_ax = range(0,len(a_kday[0]),1) # X for year plot
    a_kday.index = x_ax

    count = 0 # Initializing
    area = 0
    binary = 0
    area2 = 0
    Area = pd.DataFrame(columns=['Binary','Count','Area','AreaStep'],index=a_kday.index) # DataFrame
    threshold = 0 # This is equal to the mean for the centered data

    # Checking if data point is above threshold otherwise zero
    a_kabove1 = a_kday.where(a_kday>threshold,other=0)
    a_kabove = a_kday
    for i in range(len(a_kabove[0])-1):
        if ((a_kabove[0][i]+a_kabove[1][i]) > 0) & ((a_kabove[0][i+1]+a_kabove[1][i+1])>0):
            count +=1
            binary = 1
            area = (a_kabove.index[i+1]-a_kabove.index[i])*((a_kabove[0][i]+a_kabove[1][i])+(a_kabove[0][i+1]+a_kabove[1][i+1]))/2+area # integration, midpoint rule
            area2 = (a_kabove.index[i+1]-a_kabove.index[i])*((a_kabove[0][i]+a_kabove[1][i])+(a_kabove[0][i+1]+a_kabove[1][i+1]))/2 # For summation
        else:
            count  = 0
            area = 0
            area2 = 0
            binary = 0
        Area['Binary'][i] = binary
        Area['Count'][i] = count
        Area['Area'][i] = area
        Area['AreaStep'][i] = area2
    Area.fillna(0, inplace=True)
    idx = pd.Series(Area['Area']).idxmax()
    MaxArea = Area.loc[idx]
    return (Area, idx, MaxArea, a_kday)

def AmplitudeAreaHour(a_kdf,time_index):
    """
    

    Parameters
    ----------
    a_kdf : Data Frame
        PC's Amplitude'

    Returns
    -------
    Area : Area of first PC for consective day above threshold.
    idx : Int
        max index.
    MaxArea : Data Frame
        Max area
        Consective days above threshold
        Binary 1 if above threshold.
    a_kday: Dataframe
        Daily average amplitudes

    """
    a_kday = a_kdf.groupby(time_index.hour).mean() # hour
    x_ax = range(0,len(a_kday[0]),1) # X for year plot
    a_kday.index = x_ax
    
    count = 0 # Initializing
    area = 0
    binary = 0
    area2 = 0
    Area = pd.DataFrame(columns=['Binary','Count','Area','AreaStep'],index=a_kday.index) # DataFrame
    threshold = 0 # This is equal to the mean for the centered data
    
    # Checking if data point is above threshold otherwise zero
    a_kabove = a_kday.where(a_kday>threshold,other=0)
    for i in range(len(a_kabove[0])-1):
        if (a_kabove[0][i] > 0) & (a_kabove[0][i+1]>0):
            count +=1
            binary = 1
            area = (a_kabove.index[i+1]-a_kabove.index[i])*(a_kabove[0][i]+a_kabove[0][i+1])/2+area # integration, midpoint rule
            area2 = (a_kabove.index[i+1]-a_kabove.index[i])*(a_kabove[0][i]+a_kabove[0][i+1])/2 # For summation
        else:
            count  = 0
            area = 0
            area2 = 0
            binary = 0
        Area['Binary'][i] = binary
        Area['Count'][i] = count
        Area['Area'][i] = area
        Area['AreaStep'][i] = area2
    Area.fillna(0, inplace=True)
    idx = pd.Series(Area['Area']).idxmax()
    MaxArea = Area.loc[idx]
    a_khour = a_kday
    return (Area, idx, MaxArea, a_khour)

def CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2):
    """
    Spatial Coherence
    Parameters
    ----------
    df1 : TYPE: Data Frame
        DESCRIPTION: Eigenvector dataframe 1
    df2 : TYPE: Data Frame 
        DESCRIPTION: Eigenvector dataframe 2
    lambda1 : Type: array
         Description: Eigenvalues for eigenvectors df 1
    lambda2 : Type: array
         Description: Eigenvalues for eigenvectors df 2    

    Returns
    -------
    CoherenceMatrix : Type: np.array size NxN
        DESCRIPTION_ CoherenceMatrix for eigen vectors
    

    """

    # Calculate Coherence between eigenvectors
    # i is rows((Up and down)) j is  columns(left of right)
    CoherenceMatrix = np.zeros((len(df1),len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            CoherenceMatrix[i,j] = abs(np.dot(df1[i],df2[j].T)*np.sqrt(lambda1[i]/100*lambda2[j]/100))
    return (CoherenceMatrix)

def CoherenceMatrixEigen(df1,df2):
    """
    Spatial Coherence
    Parameters
    ----------
    df1 : TYPE: Data Frame
        DESCRIPTION: Eigenvector dataframe 1
    df2 : TYPE: Data Frame 
        DESCRIPTION: Eigenvector dataframe 2    

    Returns
    -------
    CoherenceMatrix : Type: np.array size NxN
        DESCRIPTION_ CoherenceMatrix for eigen vectors

    """

    # Calculate Coherence between eigenvectors
    # i is rows((Up and down)) j is  columns(left of right)
    CoherenceMatrix = np.zeros((len(df1),len(df2)))
    for i in range(len(df1)):
        for j in range(len(df2)):
            CoherenceMatrix[i,j] = abs(np.dot(df1[i],df2[j].T))
    
    return (CoherenceMatrix)

def CoherenceMatrixAmplitude(df1,df2):
    """
    

    Parameters
    ----------
    df1 : TYPE: Data Frame
        DESCRIPTION: Amplitude df
    df2 : TYPE
        DESCRIPTION.

    Returns
    -------
    CoherenceMatrix : NxN ndarray
        DESCRIPTION.Coherence MAtrix of amplitude

    """
    CoherenceMatrix = np.zeros((np.size(df1,1),np.size(df2,1)))
    for i in range(np.size(df1,1)):
        for j in range(np.size(df1,1)):
            CoherenceMatrix[i,j] = (1/(np.sqrt((df1[i]**2).mean()*(df2[j]**2).mean())))*(df1[i]*df2[j]).mean()
    return CoherenceMatrix

def HeatMapCoherencePlot(CoherenceMatrix1,CoherenceMatrix2,CoherenceMatrix3,df1string,df2string,yearstring,constraint3):
    """

    Parameters
    ----------
    CoherenceMatrix1 : ndarray
        DESCRIPTION. Coherence Matrix of eigenvector
    CoherenceMatrix2 : ndarray
        DESCRIPTION. Coherence Matrix of eigenvector weigted eigen values
    CoherenceMatrix3 : ndarray
        DESCRIPTION. Coherence Matrix of amplitudes
    df1string : string
        DESCRIPTION. Y-axis label match df1
    df2string : String
        DESCRIPTION.  x-axis label match df2
    yearstring : TYPE string
        DESCRIPTION. Year of interest
    constraint3 : String
        DESCRIPTION. Constraint of tranmission

    Returns
    -------
    fig : matplotlib
        DESCRIPTION: fig

    """

    ystring = df1string
    xstring = df2string
    cmap = sns.diverging_palette(200, 0, as_cmap=True)
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    sns.set(font_scale=1.2)
    im = sns.heatmap(CoherenceMatrix1[:4,:4], annot=True,cmap=cmap,fmt='.3f',vmin=-1,vmax=1,ax=ax[0],cbar=False)
    sns.heatmap(CoherenceMatrix2[:4,:4], annot=True,cmap=cmap,fmt='.3f',vmin=-1,vmax=1,ax=ax[1],cbar=False)
    sns.heatmap(CoherenceMatrix3[:4,:4], annot=True,cmap=cmap,fmt='.3f',vmin=-1,vmax=1,ax=ax[2],cbar=False)
    locs0 = ax[0].get_xticks()
    locs1 = ax[1].get_xticks()
    locs2 = ax[2].get_xticks()
    ax[0].set_xticklabels(labels=['PC1','PC2','PC3','PC4'],fontsize=12,weight='bold')
    ax[1].set_xticklabels(labels=['PC1','PC2','PC3','PC4'],fontsize=12,weight='bold')
    ax[2].set_xticklabels(labels=['PC1','PC2','PC3','PC4'],fontsize=12,weight='bold')
    ax[0].tick_params(top=False,labeltop=True,bottom=False,labelbottom=False)
    ax[1].tick_params(top=False,labeltop=True,bottom=False,labelbottom=False)
    ax[2].tick_params(top=False,labeltop=True,bottom=False,labelbottom=False)
    ax[0].set_yticklabels(labels=['PC1','PC2','PC3','PC4'],fontsize=12,weight='bold')
    ax[1].set_yticklabels(labels=['PC1','PC2','PC3','PC4'],fontsize=12,weight='bold')
    ax[2].set_yticklabels(labels=['PC1','PC2','PC3','PC4'],fontsize=12,weight='bold')
    ax[0].set_xlabel(xstring, fontsize = 18, weight = 'bold')
    ax[1].set_xlabel(xstring, fontsize = 18, weight = 'bold')
    ax[2].set_xlabel(xstring, fontsize = 18, weight = 'bold')
    ax[0].set_ylabel(ystring, fontsize = 18, weight = 'bold')
    ax[1].set_ylabel(ystring, fontsize = 18, weight = 'bold')
    ax[2].set_ylabel(ystring, fontsize = 18, weight = 'bold')
    ax[0].set_title('Coherence: $c^{(1)}$',weight='bold')
    ax[1].set_title('Coherence: $c^{(2)}$',weight='bold')
    ax[2].set_title('Coherence: $c^{(3)}$',weight='bold')
    plt.suptitle(yearstring+' - '+constraint3, fontsize = 20, weight = 'bold',y=0.92)
    mappable = im.get_children()[0]
    cbar_ax = fig.add_axes([0.05, -0.05, 0.95, 0.05])
    plt.colorbar(mappable, ax =[ax[0],ax[1],ax[2]],cax=cbar_ax,orientation='horizontal',shrink=0.5)    
    fig.tight_layout()
    return fig

def CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years):
    # Input:
        # Coherence matrices for different year
        # Years: vector with years sorted from cheapest to most expensive
    # Output:
        # three vector contaiting PC1, PC2, and PC3 coherence for different years, respectively
    
    # 1998
    C1_1998 = CohMat1998[0][0]
    C2_1998 = CohMat1998[1][1]
    C3_1998 = CohMat1998[2][2]
    
    # 1999
    C1_1999 = CohMat1999[0][0]
    C2_1999 = CohMat1999[1][1]
    C3_1999 = CohMat1999[2][2]
    
    # 2000
    C1_2000 = CohMat2000[0][0]
    C2_2000 = CohMat2000[1][1]
    C3_2000 = CohMat2000[2][2]
    
    # 1987
    C1_1987 = CohMat1987[0][0]
    C2_1987 = CohMat1987[1][1]
    C3_1987 = CohMat1987[2][2]
    
    # 2013
    C1_2013 = CohMat2013[0][0]
    C2_2013 = CohMat2013[1][1]
    C3_2013 = CohMat2013[2][2]
    
    # 2014
    C1_2014 = CohMat2014[0][0]
    C2_2014 = CohMat2014[1][1]
    C3_2014 = CohMat2014[2][2]
    
    # Assembling constants
    if Years == [1998,1999,2000,1987,2013,2014]:
        # PC1 ones coherence
        C1 = [C1_1998,C1_1999,C1_2000,C1_1987,C1_2013,C1_2014]
        C2 = [C2_1998,C2_1999,C2_2000,C2_1987,C2_2013,C2_2014]
        C3 = [C3_1998,C3_1999,C3_2000,C3_1987,C3_2013,C3_2014]
    else:
        C1 = [C1_1998,C1_1999,C1_2000,C1_2013,C1_1987,C1_2014]
        C2 = [C2_1998,C2_1999,C2_2000,C2_2013,C2_1987,C2_2014]
        C3 = [C3_1998,C3_1999,C3_2000,C3_2013,C3_1987,C3_2014]
    return (C1,C2,C3)

def CoherenceYearsPlot(C1_eig,C2_eig,C3_eig,C1_rel,C2_rel,C3_rel,C1_amp,C2_amp,C3_amp,constraint3,Years):
    # Input: COherence vectors i = j
    # For different cases
    # Output: 3 figure subplots
    fig, ax = plt.subplots(1,3, figsize = (10,3))
    x_ax = [0,1,2,3,4,5]
    ax[0].plot(x_ax,C1_eig,'-o',color = 'b',label = 'k = 1')
    ax[0].plot(x_ax,C2_eig,'-o',color = 'orange',label = 'k = 2')
    ax[0].plot(x_ax,C3_eig,'-o',color = 'green',label = 'k = 3')
    ax[1].plot(x_ax,C1_rel,'-o',color = 'b',label = 'k = 1')
    ax[1].plot(x_ax,C2_rel,'-o',color = 'orange',label = 'k = 2')
    ax[1].plot(x_ax,C3_rel,'-o',color = 'green',label = 'k = 3')
    ax[2].plot(x_ax,C1_amp,'-o',color = 'b',label = 'k = 1')
    ax[2].plot(x_ax,C2_amp,'-o',color = 'orange',label = 'k = 2')
    ax[2].plot(x_ax,C3_amp,'-o',color = 'green',label = 'k = 3')
    
    if Years == [1998,1999,2000,1987,2013,2014]:
        labels = ['0','1998','1999','2000','1987','2013','2014']
        ax[0].set_xticklabels(labels)
        ax[1].set_xticklabels(labels)
        ax[2].set_xticklabels(labels)
    else:
        labels = ['0','1998','1999','2000','2013','1987','2014']
        ax[0].set_xticklabels(labels)
        ax[1].set_xticklabels(labels)
        ax[2].set_xticklabels(labels)
    ax[2].legend(loc = 'upper right',framealpha=0.3)
    ax[0].set_title('Coherence: $c^{(1)}$',weight='bold')
    ax[1].set_title('Coherence: $c^{(2)}$',weight='bold')
    ax[2].set_title('Coherence: $c^{(3)}$',weight='bold')
    ax[0].set_ylabel('Coherence (i = j)',fontsize = 12,weight = 'bold')
    plt.suptitle(constraint3, fontsize = 16, weight = 'bold',y=0.92)
    fig.tight_layout()
    return fig