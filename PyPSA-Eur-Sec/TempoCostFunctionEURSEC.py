# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:09:03 2023

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
import warnings
from datetime import date
warnings.filterwarnings("ignore")
#%%
path = r"C:\Users\laur1\Downloads\resolved_n37_3h_dy2013_wy1960_hydro-solar-wind-heat.nc"
n = pypsa.Network(path)


#%%
# capacity
generators = n.generators.p_nom_opt
links = n.links.p_nom_opt
lines = n.lines.s_nom_opt
# loads = n.loads
stores = n.stores.e_nom_opt
storage = n.storage_units.p_nom_opt
buses = n.buses



# time series
generators_t = n.generators_t.p
links_t0 = n.links_t.p0
links_t1 = n.links_t.p1
lines_t0 = n.lines_t.p0
lines_t1 = n.lines_t.p1
loads_t = n.loads_t.p
stores_t = n.stores_t.e
storage_t = n.storage_units_t.p

# COST
totsystemcost = n.objective
syscost = n.objective/n.loads_t.p.sum().sum()
price = n.buses_t.marginal_price

load_shedding = generators_t.filter(regex='load shedding')
#NAMES = n.generators.carrier
#primary[tech] = n.generators_t.p[n.generators.index[n.generators.carrier == tech]].sum(axis=0).rename(lambda x : x[:2]).fillna(0.).groupby(level=0).sum()
#%%
opt_name = {
    "Store": "e",
    "Line": "s",
    "Transformer": "s"
}
def calculate_costs(n, label, costs):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items, "marginal_cost"] = -20.

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index,label] = marginal_costs_grouped

        # add back in all hydro
        #costs.loc[("storage_units", "capital", "hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro", "p_nom"].sum()
        #costs.loc[("storage_units", "capital", "PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS", "p_nom"].sum()
        #costs.loc[("generators", "capital", "ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror", "p_nom"].sum()

    return costs

def load_ALL2(year,pathstart,pathend):
    #Columns = list(map(str,year))
    generators = pd.DataFrame()
    storage = pd.DataFrame()
    stores = pd.DataFrame()
    links = pd.DataFrame()
    lines = pd.DataFrame()
    generators_tY = pd.DataFrame()
    storage_tY = pd.DataFrame()
    stores_tY = pd.DataFrame()
    links_tp0Y = pd.DataFrame()
    links_tp1Y = pd.DataFrame()
    lines_tp0Y = pd.DataFrame()
    lines_tp1Y = pd.DataFrame()
    loads_tY = pd.DataFrame()
    totsystemcost = pd.Series()
    syscostY = pd.Series()
    price = pd.DataFrame()
    loads_t = pd.DataFrame()
    price_t = pd.DataFrame()
    scen_costs_cap = pd.DataFrame() # System cost from function # cap. cost
    scen_costs_mar = pd.DataFrame() # Marginal cost
    scen_costs_all = pd.DataFrame() # all cost grouped
    #scen_costs_level2 = pd.DataFrame()
    for i in range(len(year)):
        path = pathstart+str(year[i])+pathend
        n = pypsa.Network(path)
        generators[i] = n.generators.p_nom_opt # Optimal installed generator capacity
        generators.rename(columns={i:year[i]},inplace=True)
        generators_t = n.generators_t.p # dispatch generators
        generators_tY = generators_tY.append(generators_t)
        storage[i] = n.storage_units.p_nom_opt # Optimal storage nominal power
        storage.rename(columns={i:year[i]},inplace=True)
        storage_t = n.storage_units_t.p # Dispatch or charge storage
        storage_tY = storage_tY.append(storage_t)
        stores[i] = n.stores.e_nom_opt # Optimal nominal energy capacity
        stores.rename(columns={i:year[i]},inplace=True)
        stores_t = n.stores_t.e # Dispatch or charge stores
        stores_tY = stores_tY.append(stores_t)
        links[i] = n.links.p_nom_opt # Optimal installed link capacity
        links.rename(columns={i:year[i]},inplace=True)
        links_tp0 = n.links_t.p0
        links_tp0Y = links_tp0Y.append(links_tp0)
        links_tp1 = n.links_t.p1
        links_tp1Y = links_tp1Y.append(links_tp1)
        lines[i] = n.lines.s_nom_opt # Optimal installed link capacity
        lines.rename(columns={i:year[i]},inplace=True)
        lines_tp0 = n.lines_t.p0
        lines_tp0Y = lines_tp0Y.append(lines_tp0)
        lines_tp1 = n.lines_t.p1
        lines_tp1Y = lines_tp1Y.append(lines_tp1)
        buses = n.buses # Location with more
        loads_t = n.loads_t.p #ALL loads
        loads_tY = loads_tY.append(loads_t) 
        syscost = pd.Series(n.objective,index = [year[i]]) #Euro
        syscost = syscost/loads_t.sum().sum() # EUR/MWh
        syscostY = syscostY.append(syscost)
        costs_ =pd.DataFrame({})
        costs_ = calculate_costs(n,'capital+marginal',costs_).groupby(level=1).sum().div(1e9)  #in billiion Euros
        scen_costs_cap[year[i]] = costs_.loc['capital'] # Billion EUR Capital cost
        scen_costs_mar[year[i]] = costs_.loc['marginal'] # marginal cost
        costs_2 =pd.DataFrame({})
        costs_2 = calculate_costs(n,'capital+marginal',costs_2).groupby(level=2).sum().div(1e9)  #in billiion Euros
        scen_costs_all[year[i]] = costs_2
        #scen_costs_2 = calculate_costs(n,'capital+marginal',costs_).groupby(level=2).sum().div(1e9)
        #scen_costs_level2[year[i]] = scen_costs_2[scen_costs_2['capital+marginal']>0].sum()
        price = n.buses_t.marginal_price #EUR/MWh
        price_t = price_t.append(price)
        price_t.rename(columns={i:year[i]},inplace=True)
        totsyscost = pd.Series(n.objective,index = [year[i]]) #Euro
        totsystemcost = totsystemcost.append(totsyscost)
        
    timeindex = pd.DataFrame(index=pd.date_range(str(year[0])+"-01-01", str(year[-1])+'-12-31-23:00:00', freq="3h"))
    timeindex =timeindex[~((timeindex.index.month == 2) & (timeindex.index.day == 29))]  # SKIP leap days
    generators_tY.index=timeindex.index
    storage_tY.index=timeindex.index
    stores_tY.index=timeindex.index
    links_tp0Y.index = timeindex.index
    links_tp1Y.index = timeindex.index
    loads_tY.index = timeindex.index
    price_t.index = timeindex.index
    syscostY.rename('SystemCost',inplace = True)
    totsystemcost.rename('SystemCost',inplace = True)
    loads = n.loads_t.p_set
    return n,generators,generators_tY,storage, storage_tY,stores,stores_tY, links,links_tp0Y,links_tp1Y,lines,lines_tp0Y,lines_tp1Y,buses, totsystemcost,syscostY,price,price_t,loads,loads_tY,scen_costs_cap, scen_costs_mar, scen_costs_all

def savefigure(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = "D:\Master\Pictures\Loadshedding"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')
#%%
year = np.arange(1960,2022,1) # 62 years

pathstart = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy"
pathend = "_hydro-solar-wind-heat.nc"
n,generators,generators_tY,storage, storage_tY,stores,stores_tY, links,links_tp0Y,links_tp1Y,lines,lines_tp0Y,lines_tp1Y,buses, totsystemcost,syscostY,price,price_t,loads,loads_tY,scen_costs_cap, scen_costs_mar, scen_costs_all = load_ALL2(year,pathstart,pathend)
#%% Load shedding
# 3 hourly resolution
load_shedding_all = generators_tY.filter(regex='load shedding')
low_voltage_shedding = load_shedding_all.filter(regex='low voltage load shedding') # MWh price for load shedding 10.000 EUR
res_rural_heat_shedding = load_shedding_all.filter(regex='residential rural heat load shedding')
res_urban_dec_heat_shedding = load_shedding_all.filter(regex='residential urban decentral heat load shedding')
ser_rural_heat_shedding = load_shedding_all.filter(regex='services rural heat load shedding')
ser_urban_dec_heat_shedding = load_shedding_all.filter(regex='services urban decentral heat load shedding')
urban_cen_heat_shedding = load_shedding_all.filter(regex='urban central heat load shedding')
heat_load_shedding = load_shedding_all.filter(regex='heat').groupby(lambda x : x[:3],axis=1).sum() # Only heat no electricity
#%% Daily sum
# OBBBBS MAYBE MULTIPLY by 3 remember only once
load_shedding_allD = load_shedding_all.groupby([pd.Grouper( freq='d')]).sum()*3
low_voltage_sheddingD = low_voltage_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
res_rural_heat_sheddingD = res_rural_heat_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
res_urban_dec_heat_sheddingD = res_urban_dec_heat_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
ser_rural_heat_sheddingD = ser_rural_heat_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
ser_urban_dec_heat_sheddingD = ser_urban_dec_heat_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
urban_cen_heat_sheddingD = urban_cen_heat_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
heat_load_sheddingD = heat_load_shedding.groupby([pd.Grouper( freq='d')]).sum()*3
# system
load_shedding_allDsys = load_shedding_allD.sum(axis=1)
low_voltage_sheddingDsys = low_voltage_sheddingD.sum(axis=1)
heat_load_sheddingDsys = heat_load_sheddingD.sum(axis=1)
#%% Prices and load - Maybe this should be different
price_columns = price_t.columns
loads_columns = loads_tY.columns
electricity_load_induagri_columns = loads_tY.filter(regex='electricity').columns # Industry and agriculture
electricity_load_ev_columns = loads_tY.filter(regex='EV').columns
electricity_load_columns = price_t.filter(regex='low voltage').columns #price_columns[0:37]
electricity_load_heatpump_columns = links_tp0Y.filter(regex='heat pump').columns
electricity_load_resistive_heater_columns =  links_tp0Y.filter(regex='resistive').columns
# HEat
heat_load_indu_columns = loads_tY.filter(regex='low-temperature').columns # Heat industry
heat_load_agri_columns = loads_tY.filter(regex='agriculture heat').columns # Heat agriculture
heat_load_columns = price_t.filter(regex='heat').columns
#%%
# Elec
electricity_price = price_t[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWh
#%% Cost electricity
electricity_load_not_all = loads_tY[price_columns[0:37]] # MWh_e
electricity_load_ev = loads_tY[electricity_load_ev_columns]
electricity_load_induagri = loads_tY[electricity_load_induagri_columns]
electricity_load_not_heat =pd.concat([electricity_load_not_all,electricity_load_ev,electricity_load_induagri],axis=1).groupby(lambda x : x[:3],axis=1).sum()

cost_electricity_not_heat = electricity_price*electricity_load_not_heat # EUR
cost_electricitysys_not_heat = cost_electricity_not_heat.sum(axis=1) # EUR
#Including Heat pumps and resistive heaters
electricity_load_heatpump =  (links_tp0Y[electricity_load_heatpump_columns])
electricity_load_resistive_heater = links_tp0Y[electricity_load_resistive_heater_columns]

electricity_load = pd.concat([electricity_load_not_heat,electricity_load_heatpump,electricity_load_resistive_heater],axis=1).groupby(lambda x : x[:3],axis=1).sum()
cost_electricity = electricity_price*electricity_load
cost_electricitysys = cost_electricity.sum(axis=1)

#%%
# Heat
heat_price = price_t[heat_load_columns] #EUR/MWh
heat_load_not_all = loads_tY[heat_load_columns] # MWh_th
heat_load_indu = loads_tY[heat_load_indu_columns]
heat_load_agri = loads_tY[heat_load_agri_columns]
heat_load = (heat_load_agri.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load
#%% Cost heat
cost_heat_agri = (heat_load_agri.groupby(lambda x : x[:3],axis=1).sum())*(heat_price.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu = (heat_load_indu.groupby(lambda x : x[:3],axis=1).sum())*(heat_price.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all = (heat_price*heat_load_not_all).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat = cost_heat_not_all+cost_heat_agri+cost_heat_indu
cost_heatsys = cost_heat.sum(axis=1) # EUR
#%%
# Combined cost
cost_heatelec = cost_electricity_not_heat.add(cost_heat.values) # EUR
cost_heatelecsys = cost_heatelec.sum(axis=1) # EUR


#%% For mapplotting
low_voltage_sheddingDCountry = low_voltage_sheddingD.groupby(lambda x : x[:2],axis=1).sum()
low_voltage_sheddingYCountry=low_voltage_sheddingDCountry.groupby([pd.Grouper( freq='y')]).sum()
low_voltage_sheddingYCountrymean = low_voltage_sheddingYCountry.mean()/1000 #MWh-->GWh
low_voltage_sheddingYCountrystd = low_voltage_sheddingYCountry.std()/1000 #MWh-->GWh
low_voltage_sheddingYCountryCV = low_voltage_sheddingYCountrystd/low_voltage_sheddingYCountrymean

heat_load_sheddingDCountry = heat_load_sheddingD.groupby(lambda x : x[:2],axis=1).sum()
heat_load_sheddingYCountry = heat_load_sheddingDCountry.groupby([pd.Grouper( freq='y')]).sum()
heat_load_shedding_YCountrymean = heat_load_sheddingYCountry.mean()/1000 #MWh-->GWh
heat_load_shedding_YCountrystd = heat_load_sheddingYCountry.std()/1000 #MWh-->GWh
heat_load_shedding_YCountryCV = heat_load_shedding_YCountrystd/heat_load_shedding_YCountrymean
#%% Mapplot load shedding electricity
fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=0.7)
ax.coastlines(resolution='110m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.6)
ax.set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
europe_not_included = {'AD','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                       'MT','RU','SM','UA','VA','XK'}

shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_1 = reader.records()
name_loop = 'start'
df = low_voltage_sheddingYCountry
for country in countries_1:
    if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'                
        else:
            name_loop = country.attributes['ISO_A2']
        for country_CF in low_voltage_sheddingYCountrymean.index.values:
            if country_CF == name_loop:
                color_value = low_voltage_sheddingYCountrymean.loc[country_CF] #[PC_NO-1]
                if color_value >= 0:
                    color_value = np.absolute(color_value)/low_voltage_sheddingYCountrymean.max()
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 1, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(low_voltage_sheddingYCountrystd.loc[country_CF])),fontsize=13,weight='bold')
                else:
                    color_value = np.absolute(color_value)/low_voltage_sheddingYCountrymean.max()
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 0, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(low_voltage_sheddingYCountrystd.loc[country_CF])),fontsize=13,weight='bold')
    else:
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=0.7, 
                             edgecolor="black", label=country.attributes['ADM0_A3'])

cmap = LinearSegmentedColormap.from_list('mycmap', ['white',(1,1,0.666),(1,1,0.333),(1,1,0)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=0, vmax=low_voltage_sheddingYCountrymean.max())
cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_xlabel('Annual Average (1960-2021) [GWh]',fontsize=18,weight='bold')
ax.set_title('Low voltage load shedding - 1960-2021',fontsize=22, weight='bold')
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGESHEDDING', fig)
# Mean sum 62 years and standard deviation in GWh
# Annual average af absuotue load shedding 

#%% Load shedding heat
fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1, linewidth=0.7)
ax.coastlines(resolution='110m')
ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.78,0.8,0.78), alpha=0.6)
ax.set_extent ((-9.5, 30.5, 35, 71), cartopy.crs.PlateCarree())
europe_not_included = {'AD','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                       'MT','RU','SM','UA','VA','XK'}

shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries_1 = reader.records()
name_loop = 'start'
#df = low_voltage_sheddingYCountry
for country in countries_1:
    if country.attributes['REGION_UN'] == 'Europe' and country.attributes['ISO_A2'] not in europe_not_included:
        if country.attributes['NAME'] == 'Norway':
            name_loop = 'NO'
        elif country.attributes['NAME'] == 'France':
            name_loop = 'FR'                
        else:
            name_loop = country.attributes['ISO_A2']
        for country_CF in heat_load_shedding_YCountrymean.index.values:
            if country_CF == name_loop:
                color_value = heat_load_shedding_YCountrymean.loc[country_CF] #[PC_NO-1]
                if color_value >= 0:
                    color_value = np.absolute(color_value)/heat_load_shedding_YCountrymean.max()
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(1, 0, 0), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(heat_load_shedding_YCountrystd.loc[country_CF])),fontsize=13,weight='bold')
                else:
                    color_value = np.absolute(color_value)/heat_load_shedding_YCountrymean.max()
                    ax.add_geometries([country.geometry], ccrs.PlateCarree(), linewidth=0.7, facecolor=(0, 0, 1), 
                                         alpha=(np.min([color_value, 1])), edgecolor="black", label=country.attributes['ADM0_A3'])
                    ax.text(country.attributes['LABEL_X']-0.5,country.attributes['LABEL_Y']-0.5,str(round(heat_load_shedding_YCountrystd.loc[country_CF])),fontsize=13,weight='bold')
    else:
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=(.7,.7,.7), alpha=1, linewidth=0.7, 
                             edgecolor="black", label=country.attributes['ADM0_A3'])

cmap = LinearSegmentedColormap.from_list('mycmap', ['white',(1,0.666,0.666),(1,0.333,0.333),(1,0,0)])
shrink = 0.08
ax1 = fig.add_axes([0.125+shrink, 0.105, 0.775-shrink*2, 0.02])
norm = matplotlib.colors.Normalize(vmin=0, vmax=heat_load_shedding_YCountrymean.max())
cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap), cax=ax1, orientation='horizontal')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_xlabel('Annual Average (1960-2021) [GWh]',fontsize=18,weight='bold')
ax.set_title('Heat load shedding - 1960-2021',fontsize=22, weight='bold')
savefigure('System', 'NEWNETWORK', 'MAPHEATLOADSHEDDING', fig)

# Mean sum 62 years and standard deviation in GWh

# primary[tech] = n.generators_t.p[n.generators.index[n.generators.carrier == tech]].sum(axis=0).rename(lambda x : x[:2]).fillna(0.).groupby(level=0).sum()
#n.stores_t.p[n.stores.index[n.stores.index.str[3:] == "gas Store"]].sum().rename(lambda x : x[:2])
# %% System cost plot

total_cost = scen_costs_cap.T+scen_costs_mar.T
total_cost_sort = total_cost.sort_values(by='capital+marginal') # Sorted
fig = plt.figure(figsize=(12,6))
fig, plt.plot((total_cost), color='red', label='Cap.+ Mar. cost', lw= 2)
fig, plt.hlines(scen_costs_cap.T,xmin=1960,xmax=2021,linestyles = '--', lw = 2, label='Capital cost')
#fig, plt.hlines(syscostY.std()+syscostY.mean(),xmin=1960,xmax=2021,linestyles = '-.', lw = 2, label='STD', color='grey')
# fig, plt.hlines(-syscostY.std()+syscostY.mean(),xmin=1960,xmax=2021,linestyles = '-.', lw = 2, label='STD', color='grey')
fig, plt.xticks(fontsize=12, weight='bold')
fig, plt.yticks(fontsize=12, weight='bold')
fig, plt.xlabel('Years', fontsize = 14, weight='bold')
fig, plt.ylabel('System Cost [bn EUR]',fontsize = 14, weight='bold')
fig, plt.title('System Cost - 1960-2021', fontsize = 18, weight = 'bold')
fig, plt.legend(framealpha=0.8, fontsize=15)
fig, plt.xlim([1960,2021])
fig, plt.ylim([0,7000])
#fig, plt.annotate(('$\overline{x} = $'+str(round(syscostY.mean()))),(1958,900),fontsize=15, weight='bold')
# fig, plt.annotate(('$\sigma = \pm $'+str(round(syscostY.std()))),(1958,820),fontsize=15, weight='bold')
fig, plt.annotate("1965", xy=(total_cost_sort.index[0], total_cost_sort.values[0]), 
                   xytext=(1962, 100), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("1962", xy=(total_cost_sort.index[1], total_cost_sort.values[1]), 
                   xytext=(1962, 2000), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("1981", xy=(total_cost_sort.index[2], total_cost_sort.values[2]), 
                   xytext=(1981, 100), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2014", xy=(total_cost_sort.index[-1], total_cost_sort.values[-1]), 
                   xytext=(2014, 6700), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2017", xy=(total_cost_sort.index[-2], total_cost_sort.values[-2]), 
                   xytext=(2017, 6000), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
fig, plt.annotate("2006", xy=(total_cost_sort.index[-3], total_cost_sort.values[-3]), 
                   xytext=(2006, 6000), bbox = dict(facecolor = 'grey', alpha = 0.2), fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})

fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'Systemcost', fig)

#%%
services = scen_costs_all.filter(regex='services',axis=0)
residential = scen_costs_all.filter(regex='residential',axis=0)
services_columns = services.index
residential_columns = residential.index
servicesSum = services.sum()
residentialSum = residential.sum()
home = scen_costs_all.filter(regex='home',axis=0)
home_columns = home.index
homeSum = home.sum()
scen_cost_allfewer = scen_costs_all.drop(services_columns,axis=0)
scen_cost_allfewer = scen_cost_allfewer.drop(residential_columns,axis=0)
scen_cost_allfewer = scen_cost_allfewer.drop(home_columns,axis=0)


urban = scen_cost_allfewer.filter(regex='urban',axis=0)
urban_columns = urban.index
urbanSum = urban.sum()
scen_cost_allfewer = scen_cost_allfewer.drop(urban_columns,axis=0)

#%% add to data frame
scen_cost_allfewerall = scen_cost_allfewer.append(pd.Series(servicesSum,name='services',index=scen_cost_allfewer.columns))
scen_cost_allfewerall = scen_cost_allfewerall.append(pd.Series(residentialSum,name='residential',index=scen_cost_allfewer.columns))
scen_cost_allfewerall = scen_cost_allfewerall.append(pd.Series(homeSum,name='home',index=scen_cost_allfewer.columns))
scen_cost_allfewerall = scen_cost_allfewerall.append(pd.Series(urbanSum,name='urban',index=scen_cost_allfewer.columns))



#%% SYSTEM cost bar plot
from matplotlib import colors as mcolors1

mcolors = list((mcolors1.CSS4_COLORS))
mcolors.remove('white')
mcolors.remove('lightgrey')
mcolors.remove('lightgray')
mcolors.remove('whitesmoke')
mcolors.remove('snow')
mcolors.remove('gainsboro')
mcolors.remove('seashell')
mcolors.remove('linen')
mcolors.remove('mintcream')
mcolors.remove('floralwhite')
mcolors.remove('honeydew')
mcolors.remove('ivory')
mcolors.remove('aliceblue')
mcolors.remove('ghostwhite')
mcolors.remove('azure')
mcolors.remove('lavenderblush')
mcolors.remove('oldlace')

fig, ax = plt.subplots(1,1,figsize=(15,10))
fig, scen_cost_allfewerall.T.plot.bar(stacked=True, legend = False, ax=ax,color=mcolors)
ax.tick_params(axis='both',labelsize=16)
ax.set_ylabel('System Cost [bn EUR]',fontsize = 16, weight='bold')
ax.set_xlabel('Years',fontsize = 16, weight='bold')
ax.set_title('System Cost - 1960-2021',fontsize= 20, weight='bold')
ax.axhline(y=scen_costs_cap[1960].values, color='r', linestyle='--',lw=3,label='Capital Cost')
#ax.hlines(754,xmin=1960,xmax=2025,color='red',ls='dashed',lw=20)
plt.legend(ncol=6,loc='upper left', framealpha=0.3, fontsize=10.5,bbox_to_anchor=(-0.02,-0.15))
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'SystemcostBarplot', fig)

#%% bar plot
# Load voltage load shedding
from matplotlib.dates import DateFormatter
low_voltage_sheddingDsysYearly = low_voltage_sheddingDsys.groupby(pd.Grouper(freq='y')).sum()/1e+3
low_voltage_sheddingDsysYearly.index = year 
fig, ax = plt.subplots(figsize = (12,6))
fig, low_voltage_sheddingDsysYearly.plot.bar()
ax.tick_params(axis='both',labelsize = 13)
ax.set_ylabel('Low voltage load shedding [GWh]',fontsize = 14, weight='bold')
ax.set_xlabel('Years',fontsize = 14, weight='bold')
ax.set_title('Low voltage load shedding - 1960-2021',fontsize= 18, weight='bold')
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'BarplotlowvoltageSHEDDING', fig)

#%% bar plot
# Total load shedding aggregated
#low_voltage_sheddingDsys = low_voltage_shedding.groupby([pd.Grouper( freq='d')]).sum().sum(axis=1)
res_rural_heat_sheddingDsys = res_rural_heat_shedding.groupby([pd.Grouper( freq='d')]).sum().sum(axis=1)*3
res_urban_dec_heat_sheddingDsys = res_urban_dec_heat_shedding.groupby([pd.Grouper( freq='d')]).sum().sum(axis=1)*3
ser_rural_heat_sheddingDsys = ser_rural_heat_shedding.groupby([pd.Grouper( freq='d')]).sum().sum(axis=1)*3
ser_urban_dec_heat_sheddingDsys = ser_urban_dec_heat_shedding.groupby([pd.Grouper( freq='d')]).sum().sum(axis=1)*3
urban_cen_heat_sheddingDsys = urban_cen_heat_shedding.groupby([pd.Grouper( freq='d')]).sum().sum(axis=1)*3

dfloadsheddingsys = pd.DataFrame({'Low Voltage':low_voltage_sheddingDsys.groupby(pd.Grouper(freq='y')).sum(),'res. rural heat':res_urban_dec_heat_sheddingDsys.groupby(pd.Grouper(freq='y')).sum()
                                 ,'res. urb. dec. heat':res_urban_dec_heat_sheddingDsys.groupby(pd.Grouper(freq='y')).sum(),'ser. rural heat':ser_rural_heat_sheddingDsys.groupby(pd.Grouper(freq='y')).sum()
                                 ,'ser. urb. dec. heat': ser_urban_dec_heat_sheddingDsys.groupby(pd.Grouper(freq='y')).sum(),'urb. Cen. heat':urban_cen_heat_sheddingDsys.groupby(pd.Grouper(freq='y')).sum()})/1e+3
dfloadsheddingsys.index = year
#%% bar plot
#fig, ax = plt.subplots(figsize = (12,6))
ax = dfloadsheddingsys.plot.bar(figsize = (12,6),stacked=True)
ax.tick_params(axis='both',labelsize = 13)
ax.set_ylabel('Accumulated load shedding [GWh]',fontsize = 14, weight='bold')
ax.set_xlabel('Years',fontsize = 14, weight='bold')
ax.set_title('Load shedding - 1960-2021',fontsize= 18, weight='bold')
fig = ax.get_figure()
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'BarplotALLloadSHEDDING', fig)

#%%
fig, ax = plt.subplots(2,1,figsize = (12,6))
dfloadsheddingsys[:31].plot.bar(figsize = (12,6),stacked=False,ax=ax[0],legend=False,width=0.8)
dfloadsheddingsys[31:].plot.bar(figsize = (12,6),stacked=False,ax=ax[1],legend=False,width=0.8)
ax[0].tick_params(axis='both',labelsize = 13)
ax[1].tick_params(axis='both',labelsize = 13)
ax[0].set_ylim([0,15000])
ax[1].set_ylim([0,15000])
fig.text(-0.01, 0.5, 'Load shedding [GWh]', va='center', rotation='vertical',fontsize=14,weight='bold')
#ax[0].set_ylabel('Load shedding [GWh]',fontsize = 14, weight='bold')
ax[1].set_xlabel('Years',fontsize = 14, weight='bold')
ax[0].set_title('Load shedding - 1960-2021',fontsize= 18, weight='bold')
ax[0].legend(framealpha=0.2,ncol=3,fontsize=11,loc='upper left')
#fig = ax.get_figure()
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'BarplotALLloadSHEDDINGNOTstacked', fig)


#%% sorted electricity plot function
def calculate_max_cost_by_period(df, period_lengths):
    max_costs = []
    optimal_periods = []

    for period_length in period_lengths:
        total_costs = []

        for i in range(len(df) - period_length + 1):
            subset = df.iloc[i:i + period_length]
            total_cost = subset['cost'].sum()
            total_costs.append(total_cost)

        max_period_cost = max(total_costs)
        max_costs.append(max_period_cost)
        optimal_period = (df.iloc[total_costs.index(max_period_cost)]['date'], df.iloc[total_costs.index(max_period_cost) + period_length - 1]['date'])
        optimal_periods.append(optimal_period)

    return max_costs, optimal_periods
#%% electricity and total price(heatpluselec)
e_price_systemY = pd.DataFrame(columns=year,index=loads.index) 
price_systemY = pd.DataFrame(columns=year,index=loads.index) 

for i in year:
    e_price_systemY[i] =  cost_electricitysys[str(i)].values
    price_systemY[i] = cost_heatelecsys[str(i)].values

#%% electricity costs
daily_costs = e_price_systemY.groupby([pd.Grouper( freq='d')]).sum()*3
date_range = daily_costs.index
max_costsALLelec = pd.DataFrame(columns=year)
optimal_periods_allelec = pd.DataFrame(columns=year)
for i in year:
    data = {
        'date': date_range,
        'cost': daily_costs[i]
        }

    df = pd.DataFrame(data)

    # Specify the period lengths you want to consider (in days)
    period_lengths = range(1, 366)  # Example: consider periods from 1 hour to 24 hours

    max_costs_elec, optimal_periods_elec = calculate_max_cost_by_period(df, period_lengths)
    max_costsALLelec[i] = max_costs_elec
    optimal_periods_allelec[i] = optimal_periods_elec

#%% electricity + heat costs
daily_costs = price_systemY.groupby([pd.Grouper( freq='d')]).sum()*3
date_range = daily_costs.index
max_costsALLelecheat = pd.DataFrame(columns=year)
optimal_periods_allelecheat = pd.DataFrame(columns=year)
for i in year:
    data = {
        'date': date_range,
        'cost': daily_costs[i]
        }

    df = pd.DataFrame(data)

    # Specify the period lengths you want to consider (in hours)
    period_lengths = range(1, 366)  # Example: consider periods from 1 hour to 24 hours

    max_costs_elecheat, optimal_periods_elecheat = calculate_max_cost_by_period(df, period_lengths)
    max_costsALLelecheat[i] = max_costs_elecheat
    optimal_periods_allelecheat[i] = optimal_periods_elecheat


#%%
optimal_periods_allelec.to_csv(r'C:\Users\laur1\OneDrive\4. Civil - semester\new_data\optimal_elec2.csv')
#%%
optimal_periods_allelecheat.to_csv(r'C:\Users\laur1\OneDrive\4. Civil - semester\new_data\optimal_elecheat2.csv')

#%%
optimal_periods_allelecheat.loc[13]
#%%
"""
If threshold changes all plots should be generated again
"""
#%% Electricity periods
T_elec = (max_costsALLelec.mean(axis=1).loc[13])*0.8 #3506 bn EUR Threshold %80% of the mean value at 14 days
period_length = range(0,14) # max period 14 days
df_cost_elec = pd.DataFrame(columns=['Cost','Period','Start date','End date','Period length','Start date - N'],index=year)
#df_period_elec = pd.DataFrame(columns=['Cost', 'Period'],index=year)
for i in year[:]:
    YY = max_costsALLelec[i][period_length]
    XX = optimal_periods_allelec[i][period_length]
    for j in range(len(YY)):
        if YY[j] > T_elec:
            df_cost_elec['Cost'].loc[i] = YY[j]
            
            df_cost_elec['Period'].loc[i] = tuple(XX.loc[YY.loc[YY==YY[j]].index])[0]
            df_cost_elec['Start date'].loc[i] = df_cost_elec['Period'].loc[i][0].date()
            df_cost_elec['Start date - N'].loc[i] = (df_cost_elec['Start date'].loc[i]-date(df_cost_elec['Start date'].loc[i].year, 1, 1)).days+1
            df_cost_elec['End date'].loc[i] = df_cost_elec['Period'].loc[i][1].date()
            df_cost_elec['Period length'].loc[i] = (df_cost_elec['End date'].loc[i]-df_cost_elec['Start date'].loc[i]).days
            break
#%% electriity cost plot OBSS maybe change for years of expensive and cheap
fig = plt.figure(figsize=(12,6))
for i in year:
    if (i == 2014):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Exp.)',color='red')
    elif (i==2017):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Exp.)',color='darkred')
    elif (i==2006):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Exp.)',color='magenta') 
    elif (i==2019 or i==2012 or i==2007 or i ==2020 or i==2015 or i==2018 or i==2011):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Exp.)',color='green') 
    elif (i==1965):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Cheap)',color='blue') 
    elif (i==1962):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Cheap)',color='lightblue')
    elif (i==1981):
        plt.plot((max_costsALLelec[i]/10**9),label=str(i)+' (Cheap)',color='dodgerblue') 
    else:
        plt.plot((max_costsALLelec[i]/10**9),color='grey',alpha=0.4)
plt.plot((max_costsALLelec.mean(axis=1)/10**9),label='Mean (1960-2021)',lw=4,color='k')
plt.hlines(T_elec//1e+9,0,14,color='orange',lw=5,label=('Threshold'))
plt.legend(ncol=4,framealpha=0.4,fontsize='11',loc='upper center')
plt.axvline(14, ymin=0,ymax=1,color='k',ls='--')
plt.annotate("2 Weeks", xy=(14,-1500), xytext=(14, -5000), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.tick_params(axis='both',labelsize=16)
plt.ylabel('Cost of most expensive period [bn EUR]',fontsize=16,weight='bold')
plt.xlabel('Length of period [days]',fontsize=16, weight='bold')
plt.xlim([0,365])
plt.title('Cost of Electricity - (1960-2021)',fontsize=18,weight='bold')
#plt.ylim([0,270])
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'Costofelectricity', fig)
#%% heat+Electricity periods
T_elecheat = max_costsALLelecheat.mean(axis=1).loc[13]*0.8 #6221 bn EUR Threshold around 80% mean
period_length = range(0,14) # max period 14 days
df_cost_elecheat = pd.DataFrame(columns=['Cost','Period','Start date','End date','Period length','Start date - N'],index=year)
#df_period_elec = pd.DataFrame(columns=['Cost', 'Period'],index=year)
for i in year[:]:
    YY = max_costsALLelecheat[i][period_length]
    XX = optimal_periods_allelecheat[i][period_length]
    for j in range(len(YY)):
        if YY[j] > T_elecheat:
            df_cost_elecheat['Cost'].loc[i] = YY[j]
            
            df_cost_elecheat['Period'].loc[i] = tuple(XX.loc[YY.loc[YY==YY[j]].index])[0]
            df_cost_elecheat['Start date'].loc[i] = df_cost_elecheat['Period'].loc[i][0].date()
            df_cost_elecheat['Start date - N'].loc[i] = (df_cost_elecheat['Start date'].loc[i]-date(df_cost_elecheat['Start date'].loc[i].year, 1, 1)).days+1
            df_cost_elecheat['End date'].loc[i] = df_cost_elecheat['Period'].loc[i][1].date()
            df_cost_elecheat['Period length'].loc[i] = (df_cost_elecheat['End date'].loc[i]-df_cost_elecheat['Start date'].loc[i]).days
            break
#%% electriity+heat  cost plot OBSS maybe change for years of expensive and cheap
fig = plt.figure(figsize=(12,6))
for i in year:
    if (i == 2014):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Exp.)',color='red')
    elif (i==2017):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Exp.)',color='darkred')
    elif (i==2006):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Exp.)',color='magenta')
    elif (i==2019 or i==2012 or i==2007 or i ==2020 or i==2015 or i==2018 or i==2011):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Exp.)',color='green') 
    elif (i==1965):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Cheap)',color='blue') 
    elif (i==1962):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Cheap)',color='lightblue')
    elif (i==1981):
        plt.plot((max_costsALLelecheat[i]/10**9),label=str(i)+' (Cheap)',color='dodgerblue') 
    else:
        plt.plot((max_costsALLelecheat[i]/10**9),color='grey',alpha=0.4)
plt.plot((max_costsALLelecheat.mean(axis=1)/10**9),label='Mean (1960-2021)',lw=4,color='k')
plt.hlines(T_elecheat/(1e+9),0,14,color='orange',lw=5,label=('Threshold'))
plt.legend(ncol=4,framealpha=0.4,fontsize='11',loc='upper center')
plt.axvline(14, ymin=0,ymax=1,color='k',ls='--')
plt.annotate("2 Weeks", xy=(14,-50), xytext=(14, -9000), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.tick_params(axis='both',labelsize=16)
plt.ylabel('Cost of most expensive period [bn EUR]',fontsize=16,weight='bold')
plt.xlabel('Length of period [days]',fontsize=16, weight='bold')
plt.xlim([0,365])
plt.title('Cost of Electricity and Heat - (1960-2021)',fontsize=18,weight='bold')
#plt.ylim([0,270])
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'Costofelectricityandheat', fig)
#%% calendar plots
# Low_voltage_shredded
from matplotlib import colors
Country = 'DK'
average = 'd' # Frequency
gamma = 0.3 # For power values in color bar
#epriceDaily = ((e_price_systemY).groupby(pd.Grouper(freq=average))).sum()

#epriceDaily.dropna()
epriceDailySystem = low_voltage_sheddingDsys/1e+3 # System sum daily
epriceDailyX = low_voltage_sheddingDsys/1e+3
X = epriceDailyX.index.strftime('%m-%d')
fig, ax = plt.subplots(figsize=(18,6))
j = 0
Y2 = 0
for i in range(1960,2022):
    XX = low_voltage_sheddingDsys.index.year == i
    XX = XX.astype(int)
    #print(XX)
    Y = sum(XX)
    Y2 +=Y
    Ylen = np.full((1,Y),i)
    Xlen = X[(Y2-Y):Y2]
    Xlen2 = np.linspace(1,Y,Y)
    #print(Xlen)
    C = epriceDailySystem.loc[str(i)]
    #print(i)
    #print(C)
    j +=1
    sct = plt.scatter(Xlen2,Ylen,s=30,c=C,cmap='YlOrRd',alpha=1, norm=colors.PowerNorm(gamma=gamma,vmin=0, vmax=(550000*3)/1e+3))
    hor = plt.hlines(i-0.3,df_cost_elec['Start date - N'][i],(df_cost_elec['Start date - N'][i])+(df_cost_elec['Period length'][i]),color='k',lw=2)
    hor2 = plt.hlines(i-0.5,df_cost_elecheat['Start date - N'][i],(df_cost_elecheat['Start date - N'][i])+(df_cost_elecheat['Period length'][i]),color='cyan',lw=2)
#ax.xaxis.set_major_locator(MonthLocator(interval=1))
#ax.xaxis.set_major_formatter(DateFormatter('%m'))
fig, plt.vlines(92,1960,2021,lw=1.5,linestyles='dashed',color='black')
fig, plt.vlines(275,1960,2021,lw=1.5,linestyles='dashed',color='black')
plt.annotate("Primo Apr.", xy=(92, 1960), xytext=(92-10, 1950), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.annotate("Primo Oct.", xy=(275, 1960), xytext=(275-5, 1950), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})

fig, plt.xlabel('days', fontsize = 18, weight='bold')
fig, plt.xticks(fontsize=14, weight='bold')
fig, plt.ylabel('Years', fontsize=18, weight='bold')
fig, plt.yticks(fontsize=14, weight='bold')
fig, plt.title(' Low voltage load shedding - 1960-2021 ', fontsize=20, weight='bold')
cbar = fig.colorbar(sct,pad = 0.01)
cbar.set_label('GWh', size=15, weight='bold')
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'CalendarplotLOWVOLTAGESHEDDING', fig)

#%%
# ALL heat load shredded
from matplotlib import colors
Country = 'DK'
average = 'd' # Frequency
gamma = 0.3 # For power values in color bar
#epriceDaily = ((e_price_systemY).groupby(pd.Grouper(freq=average))).sum()

#epriceDaily.dropna()
epriceDailySystem = heat_load_sheddingDsys/1e+3 # System sum Daily
epriceDailyX = heat_load_sheddingDsys/1e+3
X = epriceDailyX.index.strftime('%m-%d')
fig, ax = plt.subplots(figsize=(18,6))
j = 0
Y2 = 0
for i in range(1960,2022):
    XX = load_shedding_allDsys.index.year == i
    XX = XX.astype(int)
    #print(XX)
    Y = sum(XX)
    Y2 +=Y
    Ylen = np.full((1,Y),i)
    Xlen = X[(Y2-Y):Y2]
    Xlen2 = np.linspace(1,Y,Y)
    #print(Xlen)
    C = epriceDailySystem.loc[str(i)]
    #print(i)
    #print(C)
    j +=1
    sct = plt.scatter(Xlen2,Ylen,s=30,c=C,cmap='YlOrRd',alpha=1,norm=colors.PowerNorm(gamma=gamma,vmin=0,vmax=(1330000*3)/1e+3))
    
    hor2 = plt.hlines(i-0.5,df_cost_elec['Start date - N'][i],(df_cost_elec['Start date - N'][i])+(df_cost_elec['Period length'][i]),color='k',lw=2)
    hor = plt.hlines(i-0.3,df_cost_elecheat['Start date - N'][i],(df_cost_elecheat['Start date - N'][i])+(df_cost_elecheat['Period length'][i]),color='cyan',lw=2)
#ax.xaxis.set_major_locator(MonthLocator(interval=1))
#ax.xaxis.set_major_formatter(DateFormatter('%m'))
fig, plt.vlines(92,1960,2021,lw=1.5,linestyles='dashed',color='black')
fig, plt.vlines(275,1960,2021,lw=1.5,linestyles='dashed',color='black')
plt.annotate("Primo Apr.", xy=(92, 1960), xytext=(92-10, 1950), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.annotate("Primo Oct.", xy=(275, 1960), xytext=(275-5, 1950), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})

fig, plt.xlabel('days', fontsize = 18, weight='bold')
fig, plt.xticks(fontsize=14, weight='bold')
fig, plt.ylabel('Years', fontsize=18, weight='bold')
fig, plt.yticks(fontsize=14, weight='bold')
fig, plt.title('Heat load shedding - 1960-2021 ', fontsize=20, weight='bold')
cbar = fig.colorbar(sct,pad = 0.01)
cbar.set_label('GWh', size=15, weight='bold')
cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'CalendarplotHeatloadSHEDDING', fig)
#%% period calendar plot
# ALL heat load shredded
from matplotlib import colors
Country = 'DK'
average = 'd' # Frequency
gamma = 0.3 # For power values in color bar
#epriceDaily = ((e_price_systemY).groupby(pd.Grouper(freq=average))).sum()

#epriceDaily.dropna()
epriceDailySystem = heat_load_sheddingDsys/1e+3 # System sum Daily
epriceDailyX = heat_load_sheddingDsys/1e+3
X = epriceDailyX.index.strftime('%m-%d')
fig, ax = plt.subplots(figsize=(18,6))
j = 0
Y2 = 0
for i in range(1960,2022):
    XX = load_shedding_allDsys.index.year == i
    XX = XX.astype(int)
    #print(XX)
    Y = sum(XX)
    Y2 +=Y
    Ylen = np.full((1,Y),i)
    Xlen = X[(Y2-Y):Y2]
    Xlen2 = np.linspace(1,Y,Y)
    #print(Xlen)
    C = epriceDailySystem.loc[str(i)]
    #print(i)
    #print(C)
    j +=1
    #sct = plt.scatter(Xlen2,Ylen,s=30,c=C,cmap='YlOrRd',alpha=1,norm=colors.PowerNorm(gamma=gamma,vmin=0,vmax=1330000/1e+3))
    hor = plt.hlines(i,df_cost_elecheat['Start date - N'][i],(df_cost_elecheat['Start date - N'][i])+(df_cost_elecheat['Period length'][i]),color='cyan',lw=3)
    hor2 = plt.hlines(i-0.3,df_cost_elec['Start date - N'][i],(df_cost_elec['Start date - N'][i])+(df_cost_elec['Period length'][i]),color='k',lw=3)
#ax.xaxis.set_major_locator(MonthLocator(interval=1))
#ax.xaxis.set_major_formatter(DateFormatter('%m'))
fig, plt.vlines(92,1960,2021,lw=1.5,linestyles='dashed',color='black')
fig, plt.vlines(275,1960,2021,lw=1.5,linestyles='dashed',color='black')
plt.annotate("Primo Apr.", xy=(92, 1960), xytext=(92-25, 1950), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})
plt.annotate("Primo Oct.", xy=(275, 1960), xytext=(275-5, 1950), bbox = dict(facecolor = 'grey', alpha = 0.2),
              fontsize=14,arrowprops={"arrowstyle":"->", "color":"black"})

fig, plt.xlabel('days', fontsize = 18, weight='bold')
fig, plt.xticks(fontsize=14, weight='bold')
fig, plt.ylabel('Years', fontsize=18, weight='bold')
fig, plt.yticks(fontsize=14, weight='bold')
fig, plt.title('Extreme Periods - 1960-2021 ', fontsize=20, weight='bold')
hor = plt.hlines(i,df_cost_elecheat['Start date - N'][i],(df_cost_elecheat['Start date - N'][i])+(df_cost_elecheat['Period length'][i]),color='cyan',lw=3,label='Electricity and Heat')
hor2 = plt.hlines(i-0.5,df_cost_elec['Start date - N'][i],(df_cost_elec['Start date - N'][i])+(df_cost_elec['Period length'][i]),color='k',lw=3,label='Electricity')
fig, plt.vlines(range(0,365),1958,2021,ls='dashed',color='grey',alpha=0.3,lw=1.5)
fig, plt.legend(loc='upper center',fontsize=18)

# cbar = fig.colorbar(sct,pad = 0.01)
# cbar.set_label('GWh', size=15, weight='bold')
# cbar.ax.tick_params(labelsize=15)
fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'CalendarplotExtremePeriods', fig)
#%%
# TRMS
TRMSlinkcolumns = links.index[:37]
TRMSlink = abs(links_tp0Y[TRMSlinkcolumns])
TRMSlines = abs(lines_tp0Y)
TRMSlines.index=TRMSlink.index
TRMSlinksys = TRMSlink.sum(axis=1)
TRMSlinessys = TRMSlines.sum(axis=1)
TRMSsys = TRMSlinessys+TRMSlinksys
TRMSsysmean = TRMSsys.mean()

# H2 factor
H2elec = links_tp0Y.filter(regex='H2 Electrolysis').groupby(lambda x : x[:3],axis=1).sum()
H2fuel = -links_tp1Y.filter(regex='H2 Fuel').groupby(lambda x : x[:3],axis=1).sum()
H2factor = H2elec.subtract(H2fuel,axis='columns')
H2factorsys = H2factor.sum(axis=1)
H2factorsysmean = H2factorsys.mean()
# Battery factor
Batcharge = links_tp0Y.filter(regex='battery charger').groupby(lambda x : x[:3],axis=1).sum()
Batdischarge = -(links_tp1Y.filter(regex='battery discharger')).groupby(lambda x : x[:3],axis=1).sum()
Batfactor = Batcharge-Batdischarge
Batfactorsys = Batfactor.sum(axis=1)
Batfactorsysmean = Batfactorsys.mean()

#%% Only electricity cost periods - For deviation
onWindsys = generators_tY.filter(regex = 'onwind').sum(axis=1)
onWindsysmean = onWindsys.mean() # MWh
offWindsys = generators_tY.filter(regex = 'offwind').sum(axis=1)
offWindsysmean = offWindsys.mean()
SolarPV = generators_tY.filter(regex = 'solar')
SolarPV = SolarPV.drop(columns =SolarPV.filter(regex='thermal') )
SolarPVsys = SolarPV.sum(axis=1)
SolarPVsysmean = SolarPVsys.mean()
rorsys = generators_tY.filter(regex = 'ror').sum(axis=1)
rorsysmean = rorsys.mean()
Hydrosys = storage_tY.filter(regex='hydro').sum(axis=1)
Hydrosysmean = Hydrosys.mean()
PHSsys = storage_tY.filter(regex='PHS').sum(axis=1)
PHSsysmean = PHSsys.mean()
# TRMS
TRMSlinkcolumns = links.index[:37]
TRMSlink = abs(links_tp0Y[TRMSlinkcolumns])
TRMSlines = abs(lines_tp0Y)
TRMSlines.index=TRMSlink.index
TRMSlinksys = TRMSlink.sum(axis=1)
TRMSlinessys = TRMSlines.sum(axis=1)
TRMSsys = TRMSlinessys+TRMSlinksys
TRMSsysmean = TRMSsys.mean()

# H2 factor
H2elec = links_tp0Y.filter(regex='H2 Electrolysis').groupby(lambda x : x[:3],axis=1).sum()
H2fuel = -links_tp1Y.filter(regex='H2 Fuel').groupby(lambda x : x[:3],axis=1).sum()
H2factor = H2elec.subtract(H2fuel,axis='columns')
H2factorsys = H2factor.sum(axis=1)
H2factorsysmean = H2factorsys.mean()
# Battery factor
Batcharge = links_tp0Y.filter(regex='battery charger').groupby(lambda x : x[:3],axis=1).sum()
Batdischarge = -(links_tp1Y.filter(regex='battery discharger')).groupby(lambda x : x[:3],axis=1).sum()
Batfactor = Batcharge-Batdischarge
Batfactorsys = Batfactor.sum(axis=1)
Batfactorsysmean = Batfactorsys.mean()

ElecLoadsys = electricity_load_not_heat.sum(axis=1)
ElecLoadsysmean = ElecLoadsys.mean()
HeatLoadsys = heat_load.sum(axis=1)
HeatLoadsysmean = HeatLoadsys.mean()
df_means_elec = pd.DataFrame(columns=['Onwind','Offwind', 'Solar PV','RoR','hydro','PHS','TRMS','H2 factor','Bat factor','E load', 'H load'],index=year)


for i in year:
    
    if pd.isna(df_cost_elec['Start date'][i]):
        #print(i)
        df_means_elec['Onwind'][i] = np.nan
        df_means_elec['Offwind'][i] = np.nan
        df_means_elec['Solar PV'][i] = np.nan
        df_means_elec['Offwind'][i] = np.nan
        df_means_elec['RoR'][i] = np.nan
        df_means_elec['hydro'][i] = np.nan
        df_means_elec['PHS'][i] = np.nan
        df_means_elec['TRMS'][i] = np.nan
        df_means_elec['H2 factor'][i] = np.nan
        df_means_elec['Bat factor'][i] = np.nan
        df_means_elec['E load'][i] = np.nan
        df_means_elec['H load'][i] = np.nan
    else:
        date_start = df_cost_elec['Start date'][i] # Here and Here (Change to heat +elec)
        date_end = df_cost_elec['End date'][i] # Here and Here (Change to heat+elec)
        date_start = date_start.replace(year=i)
        date_end = date_end.replace(year=i)
        df_means_elec['Onwind'][i] = (onWindsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['Offwind'][i] = (offWindsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['Solar PV'][i] = (SolarPVsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['E load'][i] = (ElecLoadsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['H load'][i] = (HeatLoadsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['RoR'][i] = (rorsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['hydro'][i] = (Hydrosys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['PHS'][i] = (PHSsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['TRMS'][i] = (TRMSsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['H2 factor'][i] = (H2factorsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elec['Bat factor'][i] = (Batfactorsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()



#%%
from matplotlib import colors as mcolors1

mcolors = list((mcolors1.CSS4_COLORS))
mcolors.remove('white')
mcolors.remove('lightgrey')
mcolors.remove('lightgray')
mcolors.remove('whitesmoke')
mcolors.remove('snow')
mcolors.remove('gainsboro')
mcolors.remove('seashell')
mcolors.remove('linen')
mcolors.remove('mintcream')
mcolors.remove('floralwhite')
mcolors.remove('honeydew')
mcolors.remove('ivory')
mcolors.remove('aliceblue')
mcolors.remove('ghostwhite')
mcolors.remove('azure')
mcolors.remove('lavenderblush')
mcolors.remove('oldlace')
#%% Max min for y limits elec
onwindelec_max = (df_means_elec['Onwind']/1000-onWindsysmean/1000).max()
onwindelec_min = (df_means_elec['Onwind']/1000-onWindsysmean/1000).min()
offwindelec_max = (df_means_elec['Offwind']/1000-offWindsysmean/1000).max()
offwindelec_min = (df_means_elec['Offwind']/1000-offWindsysmean/1000).min()
SolarPV_max = (df_means_elec['Solar PV']/1000-SolarPVsysmean/1000).max()
SolarPV_min = (df_means_elec['Solar PV']/1000-SolarPVsysmean/1000).min()
RoR_max = ((df_means_elec['RoR'])/1000-rorsysmean/1000).max()
RoR_min = ((df_means_elec['RoR'])/1000-rorsysmean/1000).min()
Hydro_max = (df_means_elec['hydro']/1000-Hydrosysmean/1000).max() # MAX
Hydro_min = (df_means_elec['hydro']/1000-Hydrosysmean/1000).min()
PHS_max = (df_means_elec['PHS']/1000-PHSsysmean/1000).max() # Max
PHS_min = (df_means_elec['PHS']/1000-PHSsysmean/1000).min()
H2_max = (df_means_elec['H2 factor']/1000-H2factorsysmean/1000).max()
H2_min = (df_means_elec['H2 factor']/1000-H2factorsysmean/1000).min() # Min
Bat_max = (df_means_elec['Bat factor']/1000-Batfactorsysmean/1000).max()
Bat_min = (df_means_elec['Bat factor']/1000-Batfactorsysmean/1000).min() # min
TRMS_max =  (df_means_elec['TRMS']/1000-TRMSsysmean/1000).max()
TRMS_min=  (df_means_elec['TRMS']/1000-TRMSsysmean/1000).min()

E_load_max = (df_means_elec['E load']/1000-ElecLoadsysmean/1000).max()
E_load_min = (df_means_elec['E load']/1000-ElecLoadsysmean/1000).min()

H_load_max = (df_means_elec['H load']/1000-HeatLoadsysmean/1000).max()
H_load_min = (df_means_elec['H load']/1000-HeatLoadsysmean/1000).min()

SolarThermalsys = generators_tY.filter(regex = 'solar thermal').sum(axis=1)
SolarThermalsysmean = SolarThermalsys.mean()

elec_only_mean = pd.DataFrame(columns=['Onwind','Offwind', 'Solar PV','RoR','hydro','PHS','TRMS','H2 factor','Bat factor','Solar th','E load', 'H load'],index=['mean'])
elec_only_mean['Onwind'] = onWindsysmean/1000
elec_only_mean['Offwind'] = offWindsysmean/1000
elec_only_mean['Solar PV'] = SolarPVsysmean/1000
elec_only_mean['RoR'] = rorsysmean/1000
elec_only_mean['hydro'] = Hydrosysmean/1000
elec_only_mean['PHS'] =PHSsysmean/1000
elec_only_mean['H2 factor'] = H2factorsysmean/1000
elec_only_mean['Bat factor'] = Batfactorsysmean/1000
elec_only_mean['TRMS'] = TRMSsysmean/1000
elec_only_mean['Solar th'] = SolarThermalsysmean/1000
elec_only_mean['E load'] = ElecLoadsysmean/1000
elec_only_mean['H load'] = HeatLoadsysmean/1000

#%%
fig, ax = plt.subplots(1,11,figsize = (16,6))
x_axonwind = np.arange(-0.5,0.5,1/62)
j = 0
for i in year:
    ax[0].plot(x_axonwind[j],(df_means_elec['Onwind'][i])/1000-onWindsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[1].plot(x_axonwind[j],(df_means_elec['Offwind'][i])/1000-offWindsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[2].plot(x_axonwind[j],(df_means_elec['Solar PV'][i])/1000-SolarPVsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[3].plot(x_axonwind[j],(df_means_elec['RoR'][i])/1000-rorsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[4].plot(x_axonwind[j],(df_means_elec['hydro'][i])/1000-Hydrosysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[5].plot(x_axonwind[j],(df_means_elec['PHS'][i])/1000-PHSsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[6].plot(x_axonwind[j],(df_means_elec['H2 factor'][i])/1000-H2factorsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[7].plot(x_axonwind[j],(df_means_elec['Bat factor'][i])/1000-Batfactorsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[8].plot(x_axonwind[j],(df_means_elec['TRMS'][i])/1000-TRMSsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[9].plot(x_axonwind[j],(df_means_elec['E load'][i])/1000-ElecLoadsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[10].plot(x_axonwind[j],(df_means_elec['H load'][i])/1000-HeatLoadsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])

    j+=1
axisYlim = [-400,100]  # For wind and solar  
axisYlim2 = [-100,100] # For RoR, Hydro, PHS
axisYlim3 = [-10,50] # For electricity demand
axisYlim4 = [-100,720] # For heat demand
axisYlim5 = [-450,10] # H2 factor
axisYlim6 = [-13,5] # Bat factor
axisYlim7 = [-20,10] # TRMS
axisYlim8 = [-0.25,0.25] # Ror
elecfactor = 1.05
ax[0].set_xlim([-1,1])
ax[0].set_ylim([-abs(onwindelec_min*elecfactor),abs(onwindelec_min*elecfactor)])
ax[1].set_xlim([-1,1])
ax[1].set_ylim([-abs(offwindelec_min*elecfactor),abs(offwindelec_min*elecfactor)])
ax[2].set_xlim([-1,1])
ax[2].set_ylim([-abs(SolarPV_min*elecfactor),abs(SolarPV_min*elecfactor)])
ax[3].set_xlim([-1,1])
ax[3].set_ylim([-abs(RoR_min*elecfactor),abs(RoR_min*elecfactor)])
ax[4].set_xlim([-1,1])
ax[4].set_ylim([-abs(Hydro_max*elecfactor),abs(Hydro_max*elecfactor)])
ax[5].set_xlim([-1,1])
ax[5].set_ylim([-abs(PHS_max*elecfactor),abs(PHS_max*elecfactor)])
ax[6].set_xlim([-1,1])
ax[6].set_ylim([-abs(H2_min*elecfactor),abs(H2_min*elecfactor)])
ax[7].set_xlim([-1,1])
ax[7].set_ylim([-abs(Bat_min*elecfactor),abs(Bat_min*elecfactor)])
ax[8].set_xlim([-1,1])
ax[8].set_ylim([-abs(TRMS_min*elecfactor),abs(TRMS_min*elecfactor)])
ax[9].set_xlim([-1,1])
ax[9].set_ylim([-abs(E_load_max*elecfactor),abs(E_load_max*elecfactor)])
ax[10].set_xlim([-1,1])
ax[10].set_ylim([-abs(H_load_max*elecfactor),abs(H_load_max*elecfactor)])


ax[0].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[1].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[2].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[3].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[4].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[5].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[6].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[7].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[8].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[9].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[10].hlines(0,-1,1,ls='dashed',color='black',label='mean')

# y axis
ax[0].set_ylabel('Generator Production Deviation [GWh]',fontsize = 14,weight='bold')
ax[4].set_ylabel('Storage Production Deviation [GWh]',fontsize = 14,weight='bold')
ax[8].set_ylabel('TRMS Deviation [GWh]',fontsize = 14,weight='bold')
ax[9].set_ylabel('Electricity Demand Deviation [GWh]',fontsize=14,weight='bold')
ax[10].set_ylabel('Heat Demand Deviation [GWh]',fontsize=14,weight='bold')
# Label size
ax[0].yaxis.set_tick_params(labelsize=14)
ax[1].yaxis.set_tick_params(labelsize=14)
ax[2].yaxis.set_tick_params(labelsize=14)
ax[3].yaxis.set_tick_params(labelsize=14)
ax[3].yaxis.set_tick_params(labelsize=14)
ax[4].yaxis.set_tick_params(labelsize=14)
ax[5].yaxis.set_tick_params(labelsize=14)
ax[6].yaxis.set_tick_params(labelsize=14)
ax[7].yaxis.set_tick_params(labelsize=14)
ax[8].yaxis.set_tick_params(labelsize=14)
ax[9].yaxis.set_tick_params(labelsize=14)
ax[10].yaxis.set_tick_params(labelsize=14)

ax[1].yaxis.set_tick_params(labelleft=True)
ax[2].yaxis.set_tick_params(labelleft=True)
ax[4].yaxis.set_tick_params(labelleft=True)
ax[5].yaxis.set_tick_params(labelleft=True)
# x axis 
ax[0].xaxis.set_tick_params(labelbottom=False)
ax[1].xaxis.set_tick_params(labelbottom=False)
ax[2].xaxis.set_tick_params(labelbottom=False)
ax[3].xaxis.set_tick_params(labelbottom=False)
ax[4].xaxis.set_tick_params(labelbottom=False)
ax[5].xaxis.set_tick_params(labelbottom=False)
ax[6].xaxis.set_tick_params(labelbottom=False)
ax[7].xaxis.set_tick_params(labelbottom=False)
ax[8].xaxis.set_tick_params(labelbottom=False)
ax[9].xaxis.set_tick_params(labelbottom=False)
ax[10].xaxis.set_tick_params(labelbottom=False)
# titles
ax[0].set_title('Onwind (a)',fontsize=14,weight='bold',y=1.02)
ax[1].set_title('Offwind (b)',fontsize=14,weight='bold',y=1.02)
ax[2].set_title('Solar PV (c)',fontsize=14,weight='bold',y=1.02)
ax[3].set_title('RoR (d)',fontsize=14,weight='bold',y=1.02)
ax[4].set_title('Hydro (e)',fontsize=14,weight='bold',y=1.02)
ax[5].set_title('PHS (f)',fontsize=14,weight='bold',y=1.02)
ax[6].set_title('H2 F. (g)',fontsize=14,weight='bold',y=1.02)
ax[7].set_title('Bat F. (h)',fontsize=14,weight='bold',y=1.02)
ax[8].set_title('TRMS (i)',fontsize=14,weight='bold',y=1.02)
ax[9].set_title('Elec. (j)',fontsize=14,weight='bold',y=1.02)
ax[10].set_title('Heat (k)',fontsize=14,weight='bold',y=1.02)
fig.tight_layout()
plt.legend(ncol=10,bbox_to_anchor=(-1, -0.05),fontsize=13)
savefigure('System', 'NEWNETWORK', 'DeviationElecPeriods', fig)

#%% heat + electricity cost periods - For deviation
# onWindsys = generators_tY.filter(regex = 'onwind').sum(axis=1)
# onWindsysmean = onWindsys.mean() # MWh
# offWindsys = generators_tY.filter(regex = 'offwind').sum(axis=1)
# offWindsysmean = offWindsys.mean()
# SolarPV = generators_tY.filter(regex = 'solar')
# SolarPV = SolarPV.drop(columns =SolarPV.filter(regex='thermal') )
# SolarPVsys = SolarPV.sum(axis=1)
# SolarPVsysmean = SolarPVsys.mean()
# rorsys = generators_tY.filter(regex = 'ror').sum(axis=1)
# rorsysmean = rorsys.mean()
# Hydrosys = storage_tY.filter(regex='hydro').sum(axis=1)
# Hydrosysmean = Hydrosys.mean()
# PHSsys = storage_tY.filter(regex='PHS').sum(axis=1)
# PHSsysmean = PHSsys.mean()
# ElecLoadsys = electricity_load.sum(axis=1)
# ElecLoadsysmean = ElecLoadsys.mean()
# HeatLoadsys = heat_load.sum(axis=1)
# HeatLoadsysmean = HeatLoadsys.mean()
SolarThermalsys = generators_tY.filter(regex = 'solar thermal').sum(axis=1)
SolarThermalsysmean = SolarThermalsys.mean()
df_means_elecheat = pd.DataFrame(columns=['Onwind','Offwind', 'Solar PV','RoR','hydro','PHS','H2 factor','Bat factor','TRMS','Solar Th','E load', 'H load'],index=year)


for i in year:
    
    if pd.isna(df_cost_elecheat['Start date'][i]):
        #print(i)
        df_means_elecheat['Onwind'][i] = np.nan
        df_means_elecheat['Offwind'][i] = np.nan
        df_means_elecheat['Solar PV'][i] = np.nan
        df_means_elecheat['Offwind'][i] = np.nan
        df_means_elecheat['RoR'][i] = np.nan
        df_means_elecheat['hydro'][i] = np.nan
        df_means_elecheat['PHS'][i] = np.nan
        df_means_elecheat['TRMS'][i] = np.nan
        df_means_elecheat['H2 factor'][i] = np.nan
        df_means_elecheat['Bat factor'][i] = np.nan
        df_means_elecheat['Solar Th'][i] = np.nan
        df_means_elecheat['E load'][i] = np.nan
        df_means_elecheat['H load'][i] = np.nan
    else:
        date_start = df_cost_elecheat['Start date'][i] # Here and Here (Change to heat +elec)
        date_end = df_cost_elecheat['End date'][i] # Here and Here (Change to heat+elec)
        date_start = date_start.replace(year=i)
        date_end = date_end.replace(year=i)
        df_means_elecheat['Onwind'][i] = (onWindsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['Offwind'][i] = (offWindsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['Solar PV'][i] = (SolarPVsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['E load'][i] = (ElecLoadsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['H load'][i] = (HeatLoadsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['RoR'][i] = (rorsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['hydro'][i] = (Hydrosys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['PHS'][i] = (PHSsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['Solar Th'][i] = (SolarThermalsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['TRMS'][i] = (TRMSsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['H2 factor'][i] = (H2factorsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
        df_means_elecheat['Bat factor'][i] = (Batfactorsys.loc[pd.to_datetime(date_start):pd.to_datetime(date_end)]).mean()
#%%
#%% Max min for y limits elec
onwindelec_max = (df_means_elecheat['Onwind']/1000-onWindsysmean/1000).max()
onwindelec_min = (df_means_elecheat['Onwind']/1000-onWindsysmean/1000).min()
offwindelec_max = (df_means_elecheat['Offwind']/1000-offWindsysmean/1000).max()
offwindelec_min = (df_means_elecheat['Offwind']/1000-offWindsysmean/1000).min()
SolarPV_max = (df_means_elecheat['Solar PV']/1000-SolarPVsysmean/1000).max()
SolarPV_min = (df_means_elecheat['Solar PV']/1000-SolarPVsysmean/1000).min()
RoR_max = ((df_means_elecheat['RoR'])/1000-rorsysmean/1000).max()
RoR_min = ((df_means_elecheat['RoR'])/1000-rorsysmean/1000).min()
Hydro_max = (df_means_elecheat['hydro']/1000-Hydrosysmean/1000).max() # MAX
Hydro_min = (df_means_elecheat['hydro']/1000-Hydrosysmean/1000).min()
PHS_max = (df_means_elecheat['PHS']/1000-PHSsysmean/1000).max() # Max
PHS_min = (df_means_elecheat['PHS']/1000-PHSsysmean/1000).min()
H2_max = (df_means_elecheat['H2 factor']/1000-H2factorsysmean/1000).max()
H2_min = (df_means_elecheat['H2 factor']/1000-H2factorsysmean/1000).min() # Min
Bat_max = (df_means_elecheat['Bat factor']/1000-Batfactorsysmean/1000).max()
Bat_min = (df_means_elecheat['Bat factor']/1000-Batfactorsysmean/1000).min() # min
TRMS_max =  (df_means_elecheat['TRMS']/1000-TRMSsysmean/1000).max()
TRMS_min=  (df_means_elecheat['TRMS']/1000-TRMSsysmean/1000).min()
Solarth_max = (df_means_elecheat['Solar Th']/1000-SolarThermalsysmean/1000).max()
Solarth_min = (df_means_elecheat['Solar Th']/1000-SolarThermalsysmean/1000).min()

E_load_max = (df_means_elecheat['E load']/1000-ElecLoadsysmean/1000).max()
E_load_min = (df_means_elecheat['E load']/1000-ElecLoadsysmean/1000).min()

H_load_max = (df_means_elecheat['H load']/1000-HeatLoadsysmean/1000).max()
H_load_min = (df_means_elecheat['H load']/1000-HeatLoadsysmean/1000).min()
#%%
fig, ax = plt.subplots(1,12,figsize = (17,6))
x_axonwind = np.arange(-0.5,0.5,1/62)
j = 0
for i in year:
    ax[0].plot(x_axonwind[j],(df_means_elecheat['Onwind'][i])/1000-onWindsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[1].plot(x_axonwind[j],(df_means_elecheat['Offwind'][i])/1000-offWindsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[2].plot(x_axonwind[j],(df_means_elecheat['Solar PV'][i])/1000-SolarPVsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[3].plot(x_axonwind[j],(df_means_elecheat['RoR'][i])/1000-rorsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[4].plot(x_axonwind[j],(df_means_elecheat['hydro'][i])/1000-Hydrosysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[5].plot(x_axonwind[j],(df_means_elecheat['PHS'][i])/1000-PHSsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[6].plot(x_axonwind[j],(df_means_elec['H2 factor'][i])/1000-H2factorsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[7].plot(x_axonwind[j],(df_means_elec['Bat factor'][i])/1000-Batfactorsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[8].plot(x_axonwind[j],(df_means_elec['TRMS'][i])/1000-TRMSsysmean/1000,'o',color=mcolors[j],label=df_means_elec.index[j])
    ax[9].plot(x_axonwind[j],(df_means_elecheat['Solar Th'][i])/1000-SolarThermalsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[10].plot(x_axonwind[j],(df_means_elecheat['E load'][i])/1000-ElecLoadsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])
    ax[11].plot(x_axonwind[j],(df_means_elecheat['H load'][i])/1000-HeatLoadsysmean/1000,'o',color=mcolors[j],label=df_means_elecheat.index[j])

    j+=1
axisYlim = [-400,200]  # For wind and solar  
axisYlim2 = [-10,60] # For RoR, Hydro, PHS
axisYlim3 = [-50,50] # For electricity demand
axisYlim4 = [-10,850] # For heat demand
axisYlim5 = [-7,1] # Solar thermal
axisYlim9 = [-450,10] # H2 factor
axisYlim6 = [-13,5] # Bat factor
axisYlim7 = [-20,10] # TRMS
axisYlim8 = [-0.25,0.25] # Ror
ax[0].set_xlim([-1,1])
ax[0].set_ylim([-abs(onwindelec_min*elecfactor),abs(onwindelec_min*elecfactor)])
ax[1].set_xlim([-1,1])
ax[1].set_ylim([-abs(offwindelec_min*elecfactor),abs(offwindelec_min*elecfactor)])
ax[2].set_xlim([-1,1])
ax[2].set_ylim([-abs(SolarPV_min*elecfactor),abs(SolarPV_min*elecfactor)])
ax[3].set_xlim([-1,1])
ax[3].set_ylim([-abs(RoR_min*elecfactor),abs(RoR_min*elecfactor)])
ax[4].set_xlim([-1,1])
ax[4].set_ylim([-abs(Hydro_max*elecfactor),abs(Hydro_max*elecfactor)])
ax[5].set_xlim([-1,1])
ax[5].set_ylim([-abs(PHS_max*elecfactor),abs(PHS_max*elecfactor)])
ax[6].set_xlim([-1,1])
ax[6].set_ylim([-abs(H2_min*elecfactor),abs(H2_min*elecfactor)])
ax[7].set_xlim([-1,1])
ax[7].set_ylim([-abs(Bat_min*elecfactor),abs(Bat_min*elecfactor)])
ax[8].set_xlim([-1,1])
ax[8].set_ylim([-abs(TRMS_min*elecfactor),abs(TRMS_min*elecfactor)])
ax[9].set_xlim([-1,1])
ax[9].set_ylim([-abs(Solarth_min*elecfactor),abs(Solarth_min*elecfactor)])
ax[10].set_xlim([-1,1])
ax[10].set_ylim([-abs(E_load_max*elecfactor),abs(E_load_max*elecfactor)])
ax[11].set_xlim([-1,1])
ax[11].set_ylim([-abs(H_load_max*elecfactor),abs(H_load_max*elecfactor)])


ax[0].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[1].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[2].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[3].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[4].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[5].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[6].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[7].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[8].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[9].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[10].hlines(0,-1,1,ls='dashed',color='black',label='mean')
ax[11].hlines(0,-1,1,ls='dashed',color='black',label='mean')

# y axis
ax[0].set_ylabel('Generator Production Deviation [GWh]',fontsize = 15,weight='bold')
ax[4].set_ylabel('Storage Production Deviation [GWh]',fontsize = 15,weight='bold')
ax[8].set_ylabel('TRMS Deviation [GWh]',fontsize = 15,weight='bold')
ax[9].set_ylabel('Thermal Production Deviation [GWh]',fontsize = 15,weight='bold')
ax[10].set_ylabel('Electricity Demand Deviation [GWh]',fontsize=15,weight='bold')
ax[11].set_ylabel('Heat Demand Deviation [GWh]',fontsize=15,weight='bold')
# Label size
ax[0].yaxis.set_tick_params(labelsize=15)
ax[1].yaxis.set_tick_params(labelsize=15)
ax[2].yaxis.set_tick_params(labelsize=15)
ax[3].yaxis.set_tick_params(labelsize=15)
ax[4].yaxis.set_tick_params(labelsize=15)
ax[5].yaxis.set_tick_params(labelsize=15)
ax[6].yaxis.set_tick_params(labelsize=15)
ax[7].yaxis.set_tick_params(labelsize=15)
ax[8].yaxis.set_tick_params(labelsize=15)
ax[9].yaxis.set_tick_params(labelsize=15)
ax[10].yaxis.set_tick_params(labelsize=15)
ax[11].yaxis.set_tick_params(labelsize=15)

ax[1].yaxis.set_tick_params(labelleft=True)
ax[2].yaxis.set_tick_params(labelleft=True)
ax[4].yaxis.set_tick_params(labelleft=True)
ax[5].yaxis.set_tick_params(labelleft=True)
# x axis 
ax[0].xaxis.set_tick_params(labelbottom=False)
ax[1].xaxis.set_tick_params(labelbottom=False)
ax[2].xaxis.set_tick_params(labelbottom=False)
ax[3].xaxis.set_tick_params(labelbottom=False)
ax[4].xaxis.set_tick_params(labelbottom=False)
ax[5].xaxis.set_tick_params(labelbottom=False)
ax[6].xaxis.set_tick_params(labelbottom=False)
ax[7].xaxis.set_tick_params(labelbottom=False)
ax[8].xaxis.set_tick_params(labelbottom=False)
ax[9].xaxis.set_tick_params(labelbottom=False)
ax[10].xaxis.set_tick_params(labelbottom=False)
ax[11].xaxis.set_tick_params(labelbottom=False)
# titles
ax[0].set_title('Onwind (a)',fontsize=15,weight='bold',y=1.02)
ax[1].set_title('Offwind (b)',fontsize=15,weight='bold',y=1.02)
ax[2].set_title('Solar PV (c)',fontsize=15,weight='bold',y=1.02)
ax[3].set_title('RoR (d)',fontsize=15,weight='bold',y=1.02)
ax[4].set_title('Hydro (e)',fontsize=15,weight='bold',y=1.02)
ax[5].set_title('PHS (f)',fontsize=15,weight='bold',y=1.02)
ax[6].set_title('H2 F. (g)',fontsize=15,weight='bold',y=1.02)
ax[7].set_title('Bat F. (h)',fontsize=15,weight='bold',y=1.02)
ax[8].set_title('TRMS (i)',fontsize=15,weight='bold',y=1.02)
ax[9].set_title('Solar Th (j)',fontsize=15,weight='bold',y=1.02)
ax[10].set_title('Elec. (k)',fontsize=15,weight='bold',y=1.02)
ax[11].set_title('Heat (l)',fontsize=15,weight='bold',y=1.02)
#plt.legend(ncol=10,bbox_to_anchor=(1.1, -0.05))
fig.tight_layout()
plt.legend(ncol=10,bbox_to_anchor=(-1, -0.05),fontsize=14)
savefigure('System', 'NEWNETWORK', 'DeviationElecHeatPeriods', fig)

#%% Slope calculations up to 14 days

capmarcost = scen_costs_cap+scen_costs_mar
slope_length = 14

Cost_elec0days = max_costsALLelec.loc[0]/1e+9 # bn EUR
Cost_elecheat0days = max_costsALLelecheat.loc[0]/1e+9
Cost_elec14days = max_costsALLelec.loc[13]/1e+9
Cost_elecheat14days = max_costsALLelecheat.loc[13]/1e+9

slope_elec = (Cost_elec14days-Cost_elec0days)/(14-0) #bn EUR/pr day
slope_elecheat = (Cost_elecheat14days-Cost_elecheat0days)/(14-0)

fig, ax = plt.subplots(1,2,figsize=(12,6),sharex=False)
j = 0
for i in year:
    ax[0].scatter(slope_elec.loc[i],capmarcost[i],label=i,color=mcolors[j],alpha=1)
    ax[1].scatter(slope_elecheat.loc[i],capmarcost[i],label=i,color=mcolors[j],alpha=1)
    j += 1
ax[0].set_ylabel('System Cost [bn EUR]',fontsize=15,weight='bold')
ax[0].yaxis.set_tick_params(labelsize=12)
ax[0].xaxis.set_tick_params(labelsize=12)
ax[1].xaxis.set_tick_params(labelsize=12)
ax[1].yaxis.set_tick_params(labelleft=False)
fig.text(0.52, 0, 'Slope of first 2 Weeks of Continuous Maximum Cost [bn EUR/day]', ha='center',fontsize=15,weight='bold')
x1 = slope_elec
x2 = slope_elecheat
y = capmarcost.iloc[0]
X1 = sm.add_constant(x1)
model1 = sm.OLS(list(y),X1).fit()
predictions1 = model1.predict(X1) 
print_model1 = model1.summary()
coef1 = model1.params
p_value1 = model1.summary2().tables[1]['P>|t|'].loc[0]
R21 = model1.rsquared
NormSE1 = model1.bse/(y).mean()
ax[0].plot(x1,predictions1,color='darkblue',label='Reg.',alpha=0.5)
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R21, ),
     r'$P=%.3f$' % (p_value1, ),
     r'$\bar{SE}=%.3f$' % (NormSE1.iloc[1], )))
ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=13,
             verticalalignment='top')
# right regression
X1 = sm.add_constant(x2)
model1 = sm.OLS(list(y),X1).fit()
predictions1 = model1.predict(X1) 
print_model1 = model1.summary()
coef1 = model1.params
p_value1 = model1.summary2().tables[1]['P>|t|'].loc[0]
R21 = model1.rsquared
NormSE1 = model1.bse/(y).mean()
ax[1].plot(x2,predictions1,color='darkblue',label='Reg.',alpha=0.5)
textstr = '\n'.join((
     r'$R^2=%.3f$' % (R21, ),
     r'$P=%.3f$' % (p_value1, ),
     r'$\bar{SE}=%.3f$' % (NormSE1.iloc[1], )))
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=13,
             verticalalignment='top')
ax[0].set_title('Electricity Cost',fontsize=18,weight='bold')
ax[1].set_title('Electricity and Heat Cost',fontsize=18,weight='bold')

fig.tight_layout()
plt.legend(ncol=10,bbox_to_anchor=(1, -0.12),fontsize=11)
savefigure('System', 'NEWNETWORK', 'RegressionCostVSslope', fig)



#%% Amounts of periods and max periods length
print('Max elec periods length')
print(df_cost_elec['Period length'].max())
print('Min elec periods length')
print(df_cost_elec['Period length'].min())
print('amount of periods - elec')
print(len(year)-df_means_elec['Onwind'].isna().sum())

print('Max elec+heat periods length')
print(df_cost_elecheat['Period length'].max())
print('Min elec+heat periods length')
print(df_cost_elecheat['Period length'].min())
print('amount of periods - elec+heat')
print(len(year)-df_cost_elecheat['Period length'].isna().sum())
#%% onWindsys.loc[pd.to_datetime(df_cost_elec['Start date'][1960])]
# daily_costs = e_price_systemY.groupby([pd.Grouper( freq='d')]).sum()
# i = 1960
# date_range = daily_costs.index
# data = {
#     'date': date_range,
#     'cost': daily_costs[i]
#     }

# df = pd.DataFrame(data)
# period_lengths = range(1,15)
# periods = []
# T=300*1e+9
# max_costs = []
# optimal_periods = []
# for period_length in period_lengths:
#     total_costs = []

#     for i in range(len(df) - period_length + 1):
#         subset = df.iloc[i:i + period_length]
#         total_cost = subset['cost'].sum()
#         total_costs.append(total_cost)
        
#     max_period_cost = [num for num in total_costs if num > T]
#     max_costs.append(max_period_cost)
#     optimal_period = (df[df.isin(max_period_cost).any(axis=1)]['date'], df[df.isin(max_period_cost).any(axis=1)]['date']+pd.DateOffset(days=period_length - 1))
#     optimal_periods.append(optimal_period)
# #%%
# YYY = [x > T for x in total_costs]
# YY = [num for num in total_costs if num > T]
# #df.index[total_costs.isin(max_period_cost)]['date']
# df[df.isin(max_period_cost).any(axis=1)]['date']+pd.DateOffset(days=period_length - 1)
