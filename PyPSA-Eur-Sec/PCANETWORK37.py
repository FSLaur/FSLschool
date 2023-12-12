# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:30:29 2023

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
warnings.filterwarnings("ignore")
#%% Functions
def savefigure(NAMES,constraint,figurename,fig):
    import os
    NAMES = NAMES
    root_path = "D:\Master\Pictures\Loadshedding\PCA"
    path = os.path.join(root_path,figurename+constraint+'.png')
    #path = os.path.join(root_path,NAMES+constraint,figurename+constraint+'.png')
    fig.savefig(path,bbox_inches='tight')

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
        europe_not_included = {'AD','AX','BY','FO','GG','GI','IM','IS','JE','LI','MC','MD',
                               'MT','RU','SM','UA','VA','XK'}
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
#%% Import 3 cheapest and 3 most expensive files
# Most expensive
path14 = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy2014_hydro-solar-wind-heat.nc"
path17 = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy2017_hydro-solar-wind-heat.nc"
path06 = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy2006_hydro-solar-wind-heat.nc"
n14 = pypsa.Network(path14)
n17 = pypsa.Network(path17)
n06 = pypsa.Network(path06)
# Cheapest
path65 = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy1965_hydro-solar-wind-heat.nc"
path62 = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy1962_hydro-solar-wind-heat.nc"
path81 = r"D:\Master\Postnetworks\resolved_n37_3h_dy2013_wy1981_hydro-solar-wind-heat.nc"
n65 = pypsa.Network(path65)
n62 = pypsa.Network(path62)
n81 = pypsa.Network(path81)

#%%
time_index = n14.loads_t.p.index
#%% Low voltage load shedding
low_voltage_shedding14 = n14.generators_t.p.filter(regex='low voltage load shedding').groupby(lambda x : x[:2],axis=1).sum() # MWh price for load shedding 10.000 EUR
low_voltage_shedding17 = n17.generators_t.p.filter(regex='low voltage load shedding').groupby(lambda x : x[:2],axis=1).sum() # MWh price for load shedding 10.000 EUR
low_voltage_shedding06 = n06.generators_t.p.filter(regex='low voltage load shedding').groupby(lambda x : x[:2],axis=1).sum() # MWh price for load shedding 10.000 EUR
low_voltage_shedding65 = n65.generators_t.p.filter(regex='low voltage load shedding').groupby(lambda x : x[:2],axis=1).sum() # MWh price for load shedding 10.000 EUR
low_voltage_shedding62 = n62.generators_t.p.filter(regex='low voltage load shedding').groupby(lambda x : x[:2],axis=1).sum() # MWh price for load shedding 10.000 EUR
low_voltage_shedding81 = n81.generators_t.p.filter(regex='low voltage load shedding').groupby(lambda x : x[:2],axis=1).sum() # MWh price for load shedding 10.000 EUR

#%% Heat load shedding
heat_load_shedding14 = n14.generators_t.p.filter(regex='heat').groupby(lambda x : x[:2],axis=1).sum() # Only heat no electricity 
heat_load_shedding17 = n17.generators_t.p.filter(regex='heat').groupby(lambda x : x[:2],axis=1).sum() # Only heat no electricity 
heat_load_shedding06 = n06.generators_t.p.filter(regex='heat').groupby(lambda x : x[:2],axis=1).sum() # Only heat no electricity 

heat_load_shedding65 = n65.generators_t.p.filter(regex='heat').groupby(lambda x : x[:2],axis=1).sum() # Only heat no electricity 
heat_load_shedding62 = n62.generators_t.p.filter(regex='heat').groupby(lambda x : x[:2],axis=1).sum() # Only heat no electricity 
heat_load_shedding81 = n81.generators_t.p.filter(regex='heat').groupby(lambda x : x[:2],axis=1).sum() # Only heat no electricity 
#%% Cost of heat and electricity
# price_columns = (n14.buses_t.marginal_price).columns
# loads_columns = n14.loads_t.p.columns
# electricity_load_columns = price_columns[0:37]
# heat_load_columns = n14.buses_t.marginal_price.filter(regex='heat').columns
price_columns = (n14.buses_t.marginal_price).columns
loads_columns = n14.loads_t.p.columns
electricity_load_induagri_columns = n14.loads_t.p.filter(regex='electricity').columns # Industry and agriculture
electricity_load_ev_columns = n14.loads_t.p.filter(regex='EV').columns
electricity_load_columns = (n14.buses_t.marginal_price).filter(regex='low voltage').columns #price_columns[0:37]
# HEat
heat_load_indu_columns = n14.loads_t.p.filter(regex='low-temperature').columns # Heat industry
heat_load_agri_columns = n14.loads_t.p.filter(regex='agriculture heat').columns # Heat agriculture
heat_load_columns = n14.buses_t.marginal_price.filter(regex='heat').columns

#%%
# 2014
# electricity
eprice_14 = n14.buses_t.marginal_price[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWhe
electricity_load_not_all_14 = n14.loads_t.p[price_columns[0:37]] # MWh_e
electricity_load_ev_14 = n14.loads_t.p[electricity_load_ev_columns]
electricity_load_induagri_14 = n14.loads_t.p[electricity_load_induagri_columns]
electricity_load_14 =pd.concat([electricity_load_not_all_14,electricity_load_ev_14,electricity_load_induagri_14],axis=1).groupby(lambda x : x[:3],axis=1).sum()
ecost_14 = eprice_14*electricity_load_14 # EUR

# Heat
heat_price_14 = n14.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
heat_load_not_all_14 = n14.loads_t.p[heat_load_columns] # MWh_th
heat_load_indu_14 = n14.loads_t.p[heat_load_indu_columns]
heat_load_agri_14 = n14.loads_t.p[heat_load_agri_columns]
heat_load_14 = (heat_load_agri_14.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu_14.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all_14.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load

cost_heat_agri_14 = (heat_load_agri_14.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_14.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu_14 = (heat_load_indu_14.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_14.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all_14 = (heat_price_14*heat_load_not_all_14).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat_14 = cost_heat_not_all_14+cost_heat_agri_14+cost_heat_indu_14
# Combined
costheatelec_14 = ecost_14.add(cost_heat_14.values).groupby(lambda x : x[:2],axis=1).sum()

#%% 2017
# eprice_17 = n17.buses_t.marginal_price[electricity_load_columns] #EUR/MWhe
# eload_17 = n17.loads_t.p[electricity_load_columns] # MWh_e
# ecost_17 = (eprice_17*eload_17).groupby(lambda x : x[:2],axis=1).sum() # EUR
# heatprice_17 = n17.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
# heatload_17 = n17.loads_t.p[heat_load_columns] # MWh_th
# heatcost_17 = (heatprice_17*heatload_17).groupby(lambda x : x[:2],axis=1).sum() # EUR
# costheatelec_17 = ecost_17.add(heatcost_17)
eprice_17 = n17.buses_t.marginal_price[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWhe
electricity_load_not_all_17 = n17.loads_t.p[price_columns[0:37]] # MWh_e
electricity_load_ev_17 = n17.loads_t.p[electricity_load_ev_columns]
electricity_load_induagri_17 = n17.loads_t.p[electricity_load_induagri_columns]
electricity_load_17 =pd.concat([electricity_load_not_all_17,electricity_load_ev_17,electricity_load_induagri_17],axis=1).groupby(lambda x : x[:3],axis=1).sum()
ecost_17 = eprice_17*electricity_load_17 # EUR

# Heat
heat_price_17 = n17.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
heat_load_not_all_17 = n17.loads_t.p[heat_load_columns] # MWh_th
heat_load_indu_17 = n17.loads_t.p[heat_load_indu_columns]
heat_load_agri_17 = n17.loads_t.p[heat_load_agri_columns]
heat_load_17 = (heat_load_agri_17.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu_17.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all_17.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load

cost_heat_agri_17 = (heat_load_agri_17.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_17.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu_17 = (heat_load_indu_17.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_17.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all_17 = (heat_price_17*heat_load_not_all_17).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat_17 = cost_heat_not_all_17+cost_heat_agri_17+cost_heat_indu_17
# Combined
costheatelec_17 = ecost_17.add(cost_heat_17.values).groupby(lambda x : x[:2],axis=1).sum()




#%% 2006 
# eprice_06 = n06.buses_t.marginal_price[electricity_load_columns] #EUR/MWhe
# eload_06 = n06.loads_t.p[electricity_load_columns] # MWh_e
# ecost_06 = (eprice_06*eload_06).groupby(lambda x : x[:2],axis=1).sum() # EUR
# heatprice_06 = n06.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
# heatload_06 = n06.loads_t.p[heat_load_columns] # MWh_th
# heatcost_06 = (heatprice_06*heatload_06).groupby(lambda x : x[:2],axis=1).sum() # EUR
# costheatelec_06 = ecost_06.add(heatcost_06)

eprice_06 = n06.buses_t.marginal_price[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWhe
electricity_load_not_all_06 = n06.loads_t.p[price_columns[0:37]] # MWh_e
electricity_load_ev_06 = n06.loads_t.p[electricity_load_ev_columns]
electricity_load_induagri_06 = n06.loads_t.p[electricity_load_induagri_columns]
electricity_load_06 =pd.concat([electricity_load_not_all_06,electricity_load_ev_06,electricity_load_induagri_06],axis=1).groupby(lambda x : x[:3],axis=1).sum()
ecost_06 = eprice_06*electricity_load_06 # EUR

# Heat
heat_price_06 = n06.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
heat_load_not_all_06 = n06.loads_t.p[heat_load_columns] # MWh_th
heat_load_indu_06 = n06.loads_t.p[heat_load_indu_columns]
heat_load_agri_06 = n06.loads_t.p[heat_load_agri_columns]
heat_load_06 = (heat_load_agri_06.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu_06.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all_06.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load

cost_heat_agri_06 = (heat_load_agri_06.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_06.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu_06 = (heat_load_indu_06.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_06.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all_06 = (heat_price_06*heat_load_not_all_06).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat_06 = cost_heat_not_all_06+cost_heat_agri_06+cost_heat_indu_06
# Combined
costheatelec_06 = ecost_06.add(cost_heat_06.values).groupby(lambda x : x[:2],axis=1).sum()

#%% 1965
# eprice_65 = n65.buses_t.marginal_price[electricity_load_columns] #EUR/MWhe
# eload_65 = n65.loads_t.p[electricity_load_columns] # MWh_e
# ecost_65 = (eprice_65*eload_65).groupby(lambda x : x[:2],axis=1).sum() # EUR
# heatprice_65 = n65.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
# heatload_65 = n65.loads_t.p[heat_load_columns] # MWh_th
# heatcost_65 = (heatprice_65*heatload_65).groupby(lambda x : x[:2],axis=1).sum() # EUR
# costheatelec_65 = ecost_65.add(heatcost_65)

eprice_65 = n65.buses_t.marginal_price[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWhe
electricity_load_not_all_65 = n65.loads_t.p[price_columns[0:37]] # MWh_e
electricity_load_ev_65 = n65.loads_t.p[electricity_load_ev_columns]
electricity_load_induagri_65 = n65.loads_t.p[electricity_load_induagri_columns]
electricity_load_65 =pd.concat([electricity_load_not_all_65,electricity_load_ev_65,electricity_load_induagri_65],axis=1).groupby(lambda x : x[:3],axis=1).sum()
ecost_65 = eprice_65*electricity_load_65 # EUR

# Heat
heat_price_65 = n65.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
heat_load_not_all_65 = n65.loads_t.p[heat_load_columns] # MWh_th
heat_load_indu_65 = n65.loads_t.p[heat_load_indu_columns]
heat_load_agri_65 = n65.loads_t.p[heat_load_agri_columns]
heat_load_65 = (heat_load_agri_65.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu_65.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all_65.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load

cost_heat_agri_65 = (heat_load_agri_65.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_65.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu_65 = (heat_load_indu_65.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_65.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all_65 = (heat_price_65*heat_load_not_all_65).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat_65 = cost_heat_not_all_65+cost_heat_agri_65+cost_heat_indu_65
# Combined
costheatelec_65 = ecost_65.add(cost_heat_65.values).groupby(lambda x : x[:2],axis=1).sum()


#%% 1962 
# eprice_62 = n62.buses_t.marginal_price[electricity_load_columns] #EUR/MWhe
# eload_62 = n62.loads_t.p[electricity_load_columns] # MWh_e
# ecost_62 = (eprice_62*eload_62).groupby(lambda x : x[:2],axis=1).sum() # EUR
# heatprice_62 = n62.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
# heatload_62 = n62.loads_t.p[heat_load_columns] # MWh_th
# heatcost_62 = (heatprice_62*heatload_62).groupby(lambda x : x[:2],axis=1).sum() # EUR
# costheatelec_62 = ecost_62.add(heatcost_62)
eprice_62 = n62.buses_t.marginal_price[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWhe
electricity_load_not_all_62 = n62.loads_t.p[price_columns[0:37]] # MWh_e
electricity_load_ev_62 = n62.loads_t.p[electricity_load_ev_columns]
electricity_load_induagri_62 = n62.loads_t.p[electricity_load_induagri_columns]
electricity_load_62 =pd.concat([electricity_load_not_all_62,electricity_load_ev_62,electricity_load_induagri_62],axis=1).groupby(lambda x : x[:3],axis=1).sum()
ecost_62 = eprice_62*electricity_load_62 # EUR

# Heat
heat_price_62 = n62.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
heat_load_not_all_62 = n62.loads_t.p[heat_load_columns] # MWh_th
heat_load_indu_62 = n62.loads_t.p[heat_load_indu_columns]
heat_load_agri_62 = n62.loads_t.p[heat_load_agri_columns]
heat_load_62 = (heat_load_agri_62.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu_62.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all_62.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load

cost_heat_agri_62 = (heat_load_agri_62.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_62.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu_62 = (heat_load_indu_62.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_62.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all_62 = (heat_price_62*heat_load_not_all_62).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat_62 = cost_heat_not_all_62+cost_heat_agri_62+cost_heat_indu_62
# Combined
costheatelec_62 = ecost_62.add(cost_heat_62.values).groupby(lambda x : x[:2],axis=1).sum()


#%% 1981
# eprice_81 = n81.buses_t.marginal_price[electricity_load_columns] #EUR/MWhe
# eload_81 = n81.loads_t.p[electricity_load_columns] # MWh_e
# ecost_81 = (eprice_81*eload_81).groupby(lambda x : x[:2],axis=1).sum() # EUR
# heatprice_81 = n81.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
# heatload_81 = n81.loads_t.p[heat_load_columns] # MWh_th
# heatcost_81 = (heatprice_81*heatload_81).groupby(lambda x : x[:2],axis=1).sum() # EUR
# costheatelec_81 = ecost_81.add(heatcost_81)

eprice_81 = n81.buses_t.marginal_price[electricity_load_columns].groupby(lambda x : x[:3],axis=1).sum() #EUR/MWhe
electricity_load_not_all_81 = n81.loads_t.p[price_columns[0:37]] # MWh_e
electricity_load_ev_81 = n81.loads_t.p[electricity_load_ev_columns]
electricity_load_induagri_81 = n81.loads_t.p[electricity_load_induagri_columns]
electricity_load_81 =pd.concat([electricity_load_not_all_81,electricity_load_ev_81,electricity_load_induagri_81],axis=1).groupby(lambda x : x[:3],axis=1).sum()
ecost_81 = eprice_81*electricity_load_81 # EUR

# Heat
heat_price_81 = n81.buses_t.marginal_price[heat_load_columns] #EUR/MWh_th
heat_load_not_all_81 = n81.loads_t.p[heat_load_columns] # MWh_th
heat_load_indu_81 = n81.loads_t.p[heat_load_indu_columns]
heat_load_agri_81 = n81.loads_t.p[heat_load_agri_columns]
heat_load_81 = (heat_load_agri_81.groupby(lambda x : x[:3],axis=1).sum())+(heat_load_indu_81.groupby(lambda x : x[:3],axis=1).sum())+heat_load_not_all_81.groupby(lambda x : x[:3],axis=1).sum() # ALL heat load

cost_heat_agri_81 = (heat_load_agri_81.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_81.filter(regex='services rural heat').groupby(lambda x : x[:3],axis=1).sum())
cost_heat_indu_81 = (heat_load_indu_81.groupby(lambda x : x[:3],axis=1).sum())*(heat_price_81.filter(regex='urban central heat').groupby(lambda x : x[:3],axis=1).sum()) # https://pypsa-eur-sec.readthedocs.io/en/latest/supply_demand.html#heat-demand

cost_heat_not_all_81 = (heat_price_81*heat_load_not_all_81).groupby(lambda x : x[:3],axis=1).sum() # EUR # ALL different heats
cost_heat_81 = cost_heat_not_all_81+cost_heat_agri_81+cost_heat_indu_81
# Combined
costheatelec_81 = ecost_81.add(cost_heat_81.values).groupby(lambda x : x[:2],axis=1).sum()


#%% PCA low voltage shedding
(eigen_valueslow14, eigen_vectorslow14, Cov_matlow14, clow14, variance_explainedlow14, VTlow14, X_normlow14) = PCA(low_voltage_shedding14,heat_load_shedding14.columns)
(eigen_valueslow17, eigen_vectorslow17, Cov_matlow17, clow17, variance_explainedlow17, VTlow17, X_normlow17) = PCA(low_voltage_shedding17,heat_load_shedding17.columns)
(eigen_valueslow06, eigen_vectorslow06, Cov_matlow06, clow06, variance_explainedlow06, VTlow06, X_normlow06) = PCA(low_voltage_shedding06,heat_load_shedding06.columns)

(eigen_valueslow65, eigen_vectorslow65, Cov_matlow65, clow65, variance_explainedlow65, VTlow65, X_normlow65) = PCA(low_voltage_shedding65,heat_load_shedding65.columns)
(eigen_valueslow62, eigen_vectorslow62, Cov_matlow65, clow62, variance_explainedlow62, VTlow62, X_normlow62) = PCA(low_voltage_shedding62,heat_load_shedding62.columns)
(eigen_valueslow81, eigen_vectorslow81, Cov_matlow81, clow81, variance_explainedlow81, VTlow81, X_normlow81) = PCA(low_voltage_shedding81,heat_load_shedding81.columns)

VTlow14[[1,3]] = -1*VTlow14[[1,3]]
VTlow17[[0,1,3]] = -1*VTlow17[[0,1,3]]
VTlow06[[0,1]] =-1*VTlow06[[0,1]] 
VTlow65[[0,1,3]] = -1*VTlow65[[0,1,3]]
VTlow62[[3]] = -1*VTlow62[[3]]
VTlow81[[0,1,3]] = -1*VTlow81[[0,1,3]]


a_klow14 = np.dot(X_normlow14,VTlow14)
a_klow17 = np.dot(X_normlow17,VTlow17)
a_klow06 = np.dot(X_normlow06,VTlow06)
a_klow65 = np.dot(X_normlow65,VTlow65)
a_klow62 = np.dot(X_normlow62,VTlow62)
a_klow81 = np.dot(X_normlow81,VTlow81)


# data frames
a_klow14df = pd.DataFrame(data=a_klow14,index=time_index)
a_klow17df = pd.DataFrame(data=a_klow17,index=time_index)
a_klow06df = pd.DataFrame(data=a_klow06,index=time_index)
a_klow65df = pd.DataFrame(data=a_klow65,index=time_index)
a_klow62df = pd.DataFrame(data=a_klow62,index=time_index)
a_klow81df = pd.DataFrame(data=a_klow81,index=time_index)







#%% Spatial plots - low voltage
fig = MAPPLOT(VTlow14,'Low voltage load shedding',variance_explainedlow14,'2014') # 2014
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGE2014', fig)
#%% 
fig = MAPPLOT(VTlow17,'Low voltage load shedding',variance_explainedlow17,'2017') # 2017
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGE2017', fig)
#%% 
fig = MAPPLOT(VTlow06,'Low voltage load shedding',variance_explainedlow06,'2006') # 2006
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGE2006', fig)
#%%
fig = MAPPLOT(VTlow65,'Low voltage load shedding',variance_explainedlow65,'1965') # 1965
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGE1965', fig)
#%%
fig = MAPPLOT(VTlow62,'Low voltage load shedding',variance_explainedlow62,'1962') # 1962
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGE1962', fig)

#%%
fig = MAPPLOT(VTlow81,'Low voltage load shedding',variance_explainedlow81,'1981') # 1981
savefigure('System', 'NEWNETWORK', 'MAPLOWVOLTAGE1981', fig)

#%% Season plot . low voltage
# 2014
Tlow14 = pd.DataFrame(data=a_klow14,index=time_index)
T_avg_hourlow14 = Tlow14.groupby(time_index.hour).mean() # Hour
T_avg_daylow14 = Tlow14.groupby([time_index.month,time_index.day]).mean() # Day
# 2017
Tlow17 = pd.DataFrame(data=a_klow17,index=time_index)
T_avg_hourlow17 = Tlow17.groupby(time_index.hour).mean() # Hour
T_avg_daylow17 = Tlow17.groupby([time_index.month,time_index.day]).mean() # Day
#2006
Tlow06 = pd.DataFrame(data=a_klow06,index=time_index)
T_avg_hourlow06 = Tlow06.groupby(time_index.hour).mean() # Hour
T_avg_daylow06 = Tlow06.groupby([time_index.month,time_index.day]).mean() # Day
# 1965
Tlow65 = pd.DataFrame(data=a_klow65,index=time_index)
T_avg_hourlow65 = Tlow65.groupby(time_index.hour).mean() # Hour
T_avg_daylow65 = Tlow65.groupby([time_index.month,time_index.day]).mean() # Day
# 1962
Tlow62 = pd.DataFrame(data=a_klow62,index=time_index)
T_avg_hourlow62 = Tlow62.groupby(time_index.hour).mean() # Hour
T_avg_daylow62 = Tlow62.groupby([time_index.month,time_index.day]).mean() # Day
# 1981
Tlow81 = pd.DataFrame(data=a_klow81,index=time_index)
T_avg_hourlow81 = Tlow81.groupby(time_index.hour).mean() # Hour
T_avg_daylow81 = Tlow81.groupby([time_index.month,time_index.day]).mean() # Day


period14min = 336
period14max = (336+3)
period17min = 21
period17max  = 21+2
period06min = 33
period06max = (33+3)


period65min14 = 18
period65max14 = (18+14)
period62min14 = 0
period62max14  = 0+14
period81min14 = 342
period81max14 = (342+14)

period14min14 = 18
period14max14 = (18+14)
period17min14 = 12
period17max14  = 12+14
period06min14 = 23
period06max14 = (23+14)

fig, ax = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios': [1,1.5]})
ax[0].plot(T_avg_hourlow14[0],label='2014',color='red')
ax[0].plot(T_avg_hourlow17[0],label='2017',color='darkred')
ax[0].plot(T_avg_hourlow06[0],label='2006',color='magenta')
ax[0].plot(T_avg_hourlow65[0],label='1965',color='blue')
ax[0].plot(T_avg_hourlow62[0],label='1962',color='lightblue')
ax[0].plot(T_avg_hourlow81[0],label='1981',color='darkblue')
ax[0].legend()
ax[0].set_xlabel('Hours', weight='bold',fontsize=14)
ax[0].set_ylabel("PC1 amplitude diurnal", weight='bold',fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
ax[0].set_title('Hourly average',fontsize=16, weight='bold')
x_ax = range(0,365)
ax[1].plot(x_ax,T_avg_daylow14[0],label='2014',color='red')
ax[1].hlines(-5.5,period14min,period14max,color='red',lw=8)
ax[1].hlines(-6.4,period14min14,period14max14,color='red',lw=8)
ax[1].plot(x_ax,T_avg_daylow17[0],label='2017',color='darkred')
ax[1].hlines(-5.5,period17min,period17max,color='darkred',lw=8)
ax[1].hlines(-6.5,period17min14,period17max14,color='darkred',lw=8)
ax[1].plot(x_ax,T_avg_daylow06[0],label='2006',color='magenta')
ax[1].hlines(-5.5,period06min,period06max,color='magenta',lw=8)
ax[1].hlines(-6.6,period06min14,period06max14,color='magenta',lw=8)

ax[1].hlines(-6.8,period65min14,period65max14,color='blue',lw=8,alpha=0.5)
ax[1].hlines(-6.8,period62min14,period62max14,color='lightblue',lw=8,alpha=0.5)
ax[1].hlines(-6.8,period81min14,period81max14,color='darkblue',lw=8,alpha=0.5)


ax[1].plot(x_ax,T_avg_daylow65[0],label='1965',color='blue',alpha=0.5)
ax[1].plot(x_ax,T_avg_daylow62[0],label='1962',color='lightblue',alpha=0.5)
ax[1].plot(x_ax,T_avg_daylow81[0],label='1981',color='darkblue',alpha=0.5)
ax[1].hlines(-5.5,0,365,color='k',lw=1,ls='--',label='Periods')
ax[1].hlines(-6.5,0,365,color='k',lw=1,ls='--')

ax[1].legend()
ax[1].set_xlabel('Days', weight='bold',fontsize=14)
ax[1].set_ylabel("PC1 amplitude Seasonal", weight='bold',fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
ax[1].set_title('Daily average',fontsize=16, weight='bold')

fig, plt.suptitle('Low voltage load shedding - PC1',fontsize=18, weight='bold')
savefigure('System', 'NEWNETWORK', 'SeasonplotLowvoltagePC1', fig)
#%% PCA heat voltage shedding
(eigen_valuesheat14, eigen_vectorsheat14, Cov_matheat14, cheat14, variance_explainedheat14, VTheat14, X_normheat14) = PCA(heat_load_shedding14,heat_load_shedding14.columns)
(eigen_valuesheat17, eigen_vectorsheat17, Cov_matheat17, cheat17, variance_explainedheat17, VTheat17, X_normheat17) = PCA(heat_load_shedding17,heat_load_shedding17.columns)
(eigen_valuesheat06, eigen_vectorsheat06, Cov_matheat06, cheat06, variance_explainedheat06, VTheat06, X_normheat06) = PCA(heat_load_shedding06,heat_load_shedding06.columns)

(eigen_valuesheat65, eigen_vectorsheat65, Cov_matheat65, cheat65, variance_explainedheat65, VTheat65, X_normheat65) = PCA(heat_load_shedding65,heat_load_shedding65.columns)
(eigen_valuesheat62, eigen_vectorsheat62, Cov_matheat65, cheat62, variance_explainedheat62, VTheat62, X_normheat62) = PCA(heat_load_shedding62,heat_load_shedding62.columns)
(eigen_valuesheat81, eigen_vectorsheat81, Cov_matheat81, cheat81, variance_explainedheat81, VTheat81, X_normheat81) = PCA(heat_load_shedding81,heat_load_shedding81.columns)

VTheat14[[1,3]] = -1*VTheat14[[1,3]]
VTheat17[[0,3]] = -1*VTheat17[[0,3]]
VTheat06[[1,3]] = -1*VTheat06[[1,3]]
VTheat65[[0,2,3]] = -1*VTheat65[[0,2,3]]
VTheat62[[0,2]] = -1*VTheat62[[0,2]]
VTheat81[[0,1]] = -1*VTheat81[[0,1]]

a_kheat14 = np.dot(X_normheat14,VTheat14)
a_kheat17 = np.dot(X_normheat17,VTheat17)
a_kheat06 = np.dot(X_normheat06,VTheat06)
a_kheat65 = np.dot(X_normheat65,VTheat65)
a_kheat62 = np.dot(X_normheat62,VTheat62)
a_kheat81 = np.dot(X_normheat81,VTheat81)


# data frames
a_kheat14df = pd.DataFrame(data=a_kheat14,index=time_index)
a_kheat17df = pd.DataFrame(data=a_kheat17,index=time_index)
a_kheat06df = pd.DataFrame(data=a_kheat06,index=time_index)
a_kheat65df = pd.DataFrame(data=a_kheat65,index=time_index)
a_kheat62df = pd.DataFrame(data=a_kheat62,index=time_index)
a_kheat81df = pd.DataFrame(data=a_kheat81,index=time_index)

#%% Spatial plots - heat
fig = MAPPLOT(VTheat14,'Heat load shedding',variance_explainedheat14,'2014') # 2014
savefigure('System', 'NEWNETWORK', 'MAPHEAT2014', fig)
#%%
fig = MAPPLOT(VTheat17,'Heat load shedding',variance_explainedheat17,'2017') # 2017
savefigure('System', 'NEWNETWORK', 'MAPHEAT2017', fig)

#%%
fig = MAPPLOT(VTheat06,'Heat load shedding',variance_explainedheat06,'2006') # 2006
savefigure('System', 'NEWNETWORK', 'MAPHEAT2006', fig)

#%%
fig = MAPPLOT(VTheat65,'Heat load shedding',variance_explainedheat65,'1965') # 1965
savefigure('System', 'NEWNETWORK', 'MAPHEAT1965', fig)

#%%
fig = MAPPLOT(VTheat62,'Heat load shedding',variance_explainedheat62,'1962') # 1962
savefigure('System', 'NEWNETWORK', 'MAPHEAT1962', fig)

#%%
fig = MAPPLOT(VTheat81,'Heat load shedding',variance_explainedheat81,'1981') # 1981
savefigure('System', 'NEWNETWORK', 'MAPHEAT1981', fig)


#%% Season plot . heat shedding
# 2014
Theat14 = pd.DataFrame(data=a_kheat14,index=time_index)
T_avg_hourheat14 = Theat14.groupby(time_index.hour).mean() # Hour
T_avg_dayheat14 = Theat14.groupby([time_index.month,time_index.day]).mean() # Day
# 2017
Theat17 = pd.DataFrame(data=a_kheat17,index=time_index)
T_avg_hourheat17 = Theat17.groupby(time_index.hour).mean() # Hour
T_avg_dayheat17 = Theat17.groupby([time_index.month,time_index.day]).mean() # Day
#2006
Theat06 = pd.DataFrame(data=a_kheat06,index=time_index)
T_avg_hourheat06 = Theat06.groupby(time_index.hour).mean() # Hour
T_avg_dayheat06 = Theat06.groupby([time_index.month,time_index.day]).mean() # Day
# 1965
Theat65 = pd.DataFrame(data=a_kheat65,index=time_index)
T_avg_hourheat65 = Theat65.groupby(time_index.hour).mean() # Hour
T_avg_dayheat65 = Theat65.groupby([time_index.month,time_index.day]).mean() # Day
# 1962
Theat62 = pd.DataFrame(data=a_kheat62,index=time_index)
T_avg_hourheat62 = Theat62.groupby(time_index.hour).mean() # Hour
T_avg_dayheat62 = Theat62.groupby([time_index.month,time_index.day]).mean() # Day
# 1981
Theat81 = pd.DataFrame(data=a_kheat81,index=time_index)
T_avg_hourheat81 = Theat81.groupby(time_index.hour).mean() # Hour
T_avg_dayheat81 = Theat81.groupby([time_index.month,time_index.day]).mean() # Day

period14min1 = 21
period14max1 = (21+3)
period17min1 = 21
period17max1  = 21+2
period06min1 = 23
period06max1 = (23+2)

period65min141 = 18
period65max141 = (18+14)
period62min141 = 0
period62max141  = 0+14
period81min141 = 342
period81max141 = (342+14)

period14min141 = 19
period14max141 = (19+14)
period17min141 = 13
period17max141  = 13+14
period06min141 = 22
period06max141 = (22+14)


fig, ax = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios': [1,1.5]})
ax[0].plot(T_avg_hourheat14[0],label='2014',color='red')
ax[0].plot(T_avg_hourheat17[0],label='2017',color='darkred')
ax[0].plot(T_avg_hourheat06[0],label='2006',color='magenta')
ax[0].plot(T_avg_hourheat65[0],label='1965',color='blue')
ax[0].plot(T_avg_hourheat62[0],label='1962',color='lightblue')
ax[0].plot(T_avg_hourheat81[0],label='1981',color='darkblue')
ax[0].legend()
ax[0].set_xlabel('Hours', weight='bold',fontsize=14)
ax[0].set_ylabel("PC1 amplitude diurnal", weight='bold',fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
ax[0].set_title('Hourly average',fontsize=16, weight='bold')
x_ax = range(0,365)
ax[1].plot(x_ax,T_avg_dayheat14[0],label='2014',color='red')
ax[1].plot(x_ax,T_avg_dayheat17[0],label='2017',color='darkred')
ax[1].plot(x_ax,T_avg_dayheat06[0],label='2006',color='magenta')
ax[1].hlines(-6.5,period14min1,period14max1,color='red',lw=8)
ax[1].hlines(-6.5,period17min1,period17max1,color='darkred',lw=8)
ax[1].hlines(-6.5,period06min1,period06max1,color='magenta',lw=8)
ax[1].plot(x_ax,T_avg_dayheat65[0],label='1965',color='blue',alpha=0.5)
ax[1].plot(x_ax,T_avg_dayheat62[0],label='1962',color='lightblue',alpha=0.5)
ax[1].plot(x_ax,T_avg_dayheat81[0],label='1981',color='darkblue',alpha=0.5)
ax[1].hlines(-7.7,period14min141,period14max141,color='red',lw=8)
ax[1].hlines(-7.9,period17min141,period17max141,color='darkred',lw=8)
ax[1].hlines(-8,period06min141,period06max141,color='magenta',lw=8)
ax[1].hlines(-8.2,period65min141,period65max141,color='blue',lw=8,alpha=0.5)
ax[1].hlines(-8.4,period62min141,period62max141,color='lightblue',lw=8,alpha=0.5)
ax[1].hlines(-8,period81min141,period81max141,color='darkblue',lw=8,alpha=0.5)
ax[1].hlines(-6.5,0,365,color='k',lw=1,ls='--',label='Periods')
ax[1].hlines(-7.8,0,365,color='k',lw=1,ls='--')

ax[1].legend()
ax[1].set_xlabel('Days', weight='bold',fontsize=14)
ax[1].set_ylabel("PC1 amplitude Seasonal", weight='bold',fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
ax[1].set_title('Daily average',fontsize=16, weight='bold')
fig, plt.suptitle('Heat load shedding - PC1',fontsize=18, weight='bold')
savefigure('System', 'NEWNETWORK', 'SeasonplotHeatLoadshedding', fig)

#%% PCA heat and electricity cost EUR
(eigen_valueslow14, eigen_vectorscost14, Cov_matcost14, ccost14, variance_explainedcost14, VTcost14, X_normcost14) = PCA(costheatelec_14,heat_load_shedding14.columns)
(eigen_valuescost17, eigen_vectorscost17, Cov_matcost17, ccost17, variance_explainedcost17, VTcost17, X_normcost17) = PCA(costheatelec_17,heat_load_shedding17.columns)
(eigen_valuescost06, eigen_vectorscost06, Cov_matcost06, ccost06, variance_explainedcost06, VTcost06, X_normcost06) = PCA(costheatelec_06,heat_load_shedding06.columns)

(eigen_valuescost65, eigen_vectorscost65, Cov_matcost65, ccost65, variance_explainedcost65, VTcost65, X_normcost65) = PCA(costheatelec_65,heat_load_shedding65.columns)
(eigen_valuescost62, eigen_vectorscost62, Cov_matcost65, ccost62, variance_explainedcost62, VTcost62, X_normcost62) = PCA(costheatelec_62,heat_load_shedding62.columns)
(eigen_valuescost81, eigen_vectorscost81, Cov_matcost81, ccost81, variance_explainedcost81, VTcost81, X_normcost81) = PCA(costheatelec_81,heat_load_shedding81.columns)

VTcost14[[1,2,3]] = -1*VTcost14[[1,2,3]]
VTcost17[[2]] = -1*VTcost17[[2]]
VTcost65[[1,2]] = -1*VTcost65[[1,2]]
VTcost62[[1,3]] = -1*VTcost62[[1,3]]
VTcost81[[1,2,3]] =-1*VTcost81[[1,2,3]]

a_kcost14 = np.dot(X_normcost14,VTcost14)
a_kcost17 = np.dot(X_normcost17,VTcost17)
a_kcost06 = np.dot(X_normcost06,VTcost06)
a_kcost65 = np.dot(X_normcost65,VTcost65)
a_kcost62 = np.dot(X_normcost62,VTcost62)
a_kcost81 = np.dot(X_normcost81,VTcost81)


# data frames
a_kcost14df = pd.DataFrame(data=a_kcost14,index=time_index)
a_kcost17df = pd.DataFrame(data=a_kcost17,index=time_index)
a_kcost06df = pd.DataFrame(data=a_kcost06,index=time_index)
a_kcost65df = pd.DataFrame(data=a_kcost65,index=time_index)
a_kcost62df = pd.DataFrame(data=a_kcost62,index=time_index)
a_kcost81df = pd.DataFrame(data=a_kcost81,index=time_index)
#%% Spatial heat and electricity cost
#2014
fig = MAPPLOT(VTcost14,'Heat & electricity cost',variance_explainedcost14,'2014') # 2014
savefigure('System', 'NEWNETWORK', 'MAPCost2014', fig)

#%% 
fig = MAPPLOT(VTcost17,'Heat & electricity cost',variance_explainedcost17,'2017') # 2017
savefigure('System', 'NEWNETWORK', 'MAPCost2017', fig)

#%%
fig = MAPPLOT(VTcost06,'Heat & electricity cost',variance_explainedcost06,'2006') # 2006
savefigure('System', 'NEWNETWORK', 'MAPCost2006', fig)
#%% 
fig = MAPPLOT(VTcost65,'Heat & electricity cost',variance_explainedcost65,'1965') # 1965
savefigure('System', 'NEWNETWORK', 'MAPCost1965', fig)
#%%
fig = MAPPLOT(VTcost62,'Heat & electricity cost',variance_explainedcost62,'1962') # 1962
savefigure('System', 'NEWNETWORK', 'MAPCost1962', fig)
#%%
fig = MAPPLOT(VTcost81,'Heat & electricity cost',variance_explainedcost81,'1981') # 1981
savefigure('System', 'NEWNETWORK', 'MAPCost1981', fig)
#%% Season plot . heat and electricity cost
# 2014
Tcost14 = pd.DataFrame(data=a_kcost14,index=time_index)
T_avg_hourcost14 = Tcost14.groupby(time_index.hour).mean() # Hour
T_avg_daycost14 = Tcost14.groupby([time_index.month,time_index.day]).mean() # Day
# 2017
Tcost17 = pd.DataFrame(data=a_kcost17,index=time_index)
T_avg_hourcost17 = Tcost17.groupby(time_index.hour).mean() # Hour
T_avg_daycost17 = Tcost17.groupby([time_index.month,time_index.day]).mean() # Day
#2006
Tcost06 = pd.DataFrame(data=a_kcost06,index=time_index)
T_avg_hourcost06 = Tcost06.groupby(time_index.hour).mean() # Hour
T_avg_daycost06 = Tcost06.groupby([time_index.month,time_index.day]).mean() # Day
# 1965
Tcost65 = pd.DataFrame(data=a_kcost65,index=time_index)
T_avg_hourcost65 = Tcost65.groupby(time_index.hour).mean() # Hour
T_avg_daycost65 = Tcost65.groupby([time_index.month,time_index.day]).mean() # Day
# 1962
Tcost62 = pd.DataFrame(data=a_kcost62,index=time_index)
T_avg_hourcost62 = Tcost62.groupby(time_index.hour).mean() # Hour
T_avg_daycost62 = Tcost62.groupby([time_index.month,time_index.day]).mean() # Day
# 1981
Tcost81 = pd.DataFrame(data=a_kcost81,index=time_index)
T_avg_hourcost81 = Tcost81.groupby(time_index.hour).mean() # Hour
T_avg_daycost81 = Tcost81.groupby([time_index.month,time_index.day]).mean() # Day

fig, ax = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios': [1,1.5]})
ax[0].plot(T_avg_hourcost14[0],label='2014',color='red')
ax[0].plot(T_avg_hourcost17[0],label='2017',color='darkred')
ax[0].plot(T_avg_hourcost06[0],label='2006',color='magenta')
ax[0].plot(T_avg_hourcost65[0],label='1965',color='blue')
ax[0].plot(T_avg_hourcost62[0],label='1962',color='lightblue')
ax[0].plot(T_avg_hourcost81[0],label='1981',color='darkblue')
ax[0].legend()
ax[0].set_xlabel('Hours', weight='bold',fontsize=14)
ax[0].set_ylabel("PC1 amplitude diurnal", weight='bold',fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
ax[0].set_title('Hourly average',fontsize=16, weight='bold')
x_ax = range(0,365)
ax[1].plot(x_ax,T_avg_daycost14[0],label='2014',color='red')
ax[1].plot(x_ax,T_avg_daycost17[0],label='2017',color='darkred')
ax[1].plot(x_ax,T_avg_daycost06[0],label='2006',color='magenta')
ax[1].hlines(-6.5,period14min1,period14max1,color='red',lw=8)
ax[1].hlines(-6.5,period17min1,period17max1,color='darkred',lw=8)
ax[1].hlines(-6.5,period06min1,period06max1,color='magenta',lw=8)
ax[1].plot(x_ax,T_avg_daycost65[0],label='1965',color='blue',alpha=0.3)
ax[1].plot(x_ax,T_avg_daycost62[0],label='1962',color='lightblue',alpha=0.3)
ax[1].plot(x_ax,T_avg_daycost81[0],label='1981',color='darkblue',alpha=0.3)
ax[1].hlines(-7.7,period14min141,period14max141,color='red',lw=8)
ax[1].hlines(-7.9,period17min141,period17max141,color='darkred',lw=8)
ax[1].hlines(-8,period06min141,period06max141,color='magenta',lw=8)
ax[1].hlines(-8.2,period65min141,period65max141,color='blue',lw=8,alpha=0.5)
ax[1].hlines(-8.4,period62min141,period62max141,color='lightblue',lw=8,alpha=0.5)
ax[1].hlines(-8,period81min141,period81max141,color='darkblue',lw=8,alpha=0.5)
ax[1].hlines(-6.5,0,365,color='k',lw=1,ls='--',label='Periods')
ax[1].hlines(-7.8,0,365,color='k',lw=1,ls='--')

ax[1].legend()
ax[1].set_xlabel('Days', weight='bold',fontsize=14)
ax[1].set_ylabel("PC1 amplitude Seasonal", weight='bold',fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
ax[1].set_title('Daily average',fontsize=16, weight='bold')
fig, plt.suptitle('Heat and electricity cost - PC1',fontsize=18, weight='bold')
savefigure('System', 'NEWNETWORK', 'SeasonplotHeatElecCost', fig)

#%% Coherence
#%% Coherence 2014
# Low Voltage Shedding vs Cost
df1string = 'Low voltage shedding'
df2string = 'Cost'
yearstring = '2014'
df1 = VTlow14
df2 = VTcost14
a_k1 = a_klow14df
a_k2 = a_kcost14df
lambda1 = variance_explainedlow14
lambda2 = variance_explainedcost14
CMlowcostEig14 = CoherenceMatrixEigen(df1,df2)
CMlowcostRel14 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowcostAMP14 = CoherenceMatrixAmplitude(a_k1,a_k2)

# heat load shedding vs cost
df1string = 'Heat load shedding'
df2string = 'Cost'
yearstring = '2014'
df1 = VTheat14
df2 = VTcost14
a_k1 = a_kheat14df
a_k2 = a_kcost14df
lambda1 = variance_explainedheat14
lambda2 = variance_explainedcost14
CMheatcostEig14 = CoherenceMatrixEigen(df1,df2)
CMheatcostRel14 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMheatcostAMP14 = CoherenceMatrixAmplitude(a_k1,a_k2)


# low voltage shedding vs heat load shedding
df1string = 'Low voltage shedding'
df2string = 'Heat load shedding'
yearstring = '2014'
df1 = VTlow14
df2 = VTheat14
a_k1 = a_klow14df
a_k2 = a_kheat14df
lambda1 = variance_explainedlow14
lambda2 = variance_explainedheat14
CMlowheatEig14 = CoherenceMatrixEigen(df1,df2)
CMlowheatRel14 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowheatAMP14 = CoherenceMatrixAmplitude(a_k1,a_k2)
#%% Coherence 2017
# Low Voltage Shedding vs Cost
df1string = 'Low voltage shedding'
df2string = 'Cost'
yearstring = '2017'
df1 = VTlow17
df2 = VTcost17
a_k1 = a_klow17df
a_k2 = a_kcost17df
lambda1 = variance_explainedlow17
lambda2 = variance_explainedcost17
CMlowcostEig17 = CoherenceMatrixEigen(df1,df2)
CMlowcostRel17 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowcostAMP17 = CoherenceMatrixAmplitude(a_k1,a_k2)

# heat load shedding vs cost
df1string = 'Heat load shedding'
df2string = 'Cost'
yearstring = '2017'
df1 = VTheat17
df2 = VTcost17
a_k1 = a_kheat17df
a_k2 = a_kcost17df
lambda1 = variance_explainedheat17
lambda2 = variance_explainedcost17
CMheatcostEig17 = CoherenceMatrixEigen(df1,df2)
CMheatcostRel17 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMheatcostAMP17 = CoherenceMatrixAmplitude(a_k1,a_k2)


# low voltage shedding vs heat load shedding
df1string = 'Low voltage shedding'
df2string = 'Heat load shedding'
yearstring = '2017'
df1 = VTlow17
df2 = VTheat17
a_k1 = a_klow17df
a_k2 = a_kheat17df
lambda1 = variance_explainedlow17
lambda2 = variance_explainedheat17
CMlowheatEig17 = CoherenceMatrixEigen(df1,df2)
CMlowheatRel17 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowheatAMP17 = CoherenceMatrixAmplitude(a_k1,a_k2)

#%% Coherence 2006
# Low Voltage Shedding vs Cost
df1string = 'Low voltage shedding'
df2string = 'Cost'
yearstring = '2006'
df1 = VTlow06
df2 = VTcost06
a_k1 = a_klow06df
a_k2 = a_kcost06df
lambda1 = variance_explainedlow06
lambda2 = variance_explainedcost06
CMlowcostEig06 = CoherenceMatrixEigen(df1,df2)
CMlowcostRel06 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowcostAMP06 = CoherenceMatrixAmplitude(a_k1,a_k2)

# heat load shedding vs cost
df1string = 'Heat load shedding'
df2string = 'Cost'
yearstring = '2006'
df1 = VTheat06
df2 = VTcost06
a_k1 = a_kheat06df
a_k2 = a_kcost06df
lambda1 = variance_explainedheat06
lambda2 = variance_explainedcost06
CMheatcostEig06 = CoherenceMatrixEigen(df1,df2)
CMheatcostRel06 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMheatcostAMP06 = CoherenceMatrixAmplitude(a_k1,a_k2)


# low voltage shedding vs heat load shedding
df1string = 'Low voltage shedding'
df2string = 'Heat load shedding'
yearstring = '2006'
df1 = VTlow06
df2 = VTheat06
a_k1 = a_klow06df
a_k2 = a_kheat06df
lambda1 = variance_explainedlow06
lambda2 = variance_explainedheat06
CMlowheatEig06 = CoherenceMatrixEigen(df1,df2)
CMlowheatRel06 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowheatAMP06 = CoherenceMatrixAmplitude(a_k1,a_k2)

#%% Coherence 1965
# Low Voltage Shedding vs Cost
df1string = 'Low voltage shedding'
df2string = 'Cost'
yearstring = '1965'
df1 = VTlow65
df2 = VTcost65
a_k1 = a_klow65df
a_k2 = a_kcost65df
lambda1 = variance_explainedlow65
lambda2 = variance_explainedcost65
CMlowcostEig65 = CoherenceMatrixEigen(df1,df2)
CMlowcostRel65 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowcostAMP65 = CoherenceMatrixAmplitude(a_k1,a_k2)

# heat load shedding vs cost
df1string = 'Heat load shedding'
df2string = 'Cost'
yearstring = '1965'
df1 = VTheat65
df2 = VTcost65
a_k1 = a_kheat65df
a_k2 = a_kcost65df
lambda1 = variance_explainedheat65
lambda2 = variance_explainedcost65
CMheatcostEig65 = CoherenceMatrixEigen(df1,df2)
CMheatcostRel65 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMheatcostAMP65 = CoherenceMatrixAmplitude(a_k1,a_k2)


# low voltage shedding vs heat load shedding
df1string = 'Low voltage shedding'
df2string = 'Heat load shedding'
yearstring = '1965'
df1 = VTlow65
df2 = VTheat65
a_k1 = a_klow65df
a_k2 = a_kheat65df
lambda1 = variance_explainedlow65
lambda2 = variance_explainedheat65
CMlowheatEig65 = CoherenceMatrixEigen(df1,df2)
CMlowheatRel65 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowheatAMP65 = CoherenceMatrixAmplitude(a_k1,a_k2)
#%% Coherence 1962
# Low Voltage Shedding vs Cost
df1string = 'Low voltage shedding'
df2string = 'Cost'
yearstring = '1962'
df1 = VTlow62
df2 = VTcost62
a_k1 = a_klow62df
a_k2 = a_kcost62df
lambda1 = variance_explainedlow62
lambda2 = variance_explainedcost62
CMlowcostEig62 = CoherenceMatrixEigen(df1,df2)
CMlowcostRel62 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowcostAMP62 = CoherenceMatrixAmplitude(a_k1,a_k2)

# heat load shedding vs cost
df1string = 'Heat load shedding'
df2string = 'Cost'
yearstring = '1962'
df1 = VTheat62
df2 = VTcost62
a_k1 = a_kheat62df
a_k2 = a_kcost62df
lambda1 = variance_explainedheat62
lambda2 = variance_explainedcost62
CMheatcostEig62 = CoherenceMatrixEigen(df1,df2)
CMheatcostRel62 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMheatcostAMP62 = CoherenceMatrixAmplitude(a_k1,a_k2)


# low voltage shedding vs heat load shedding
df1string = 'Low voltage shedding'
df2string = 'Heat load shedding'
yearstring = '1962'
df1 = VTlow62
df2 = VTheat62
a_k1 = a_klow62df
a_k2 = a_kheat62df
lambda1 = variance_explainedlow62
lambda2 = variance_explainedheat62
CMlowheatEig62 = CoherenceMatrixEigen(df1,df2)
CMlowheatRel62 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowheatAMP62 = CoherenceMatrixAmplitude(a_k1,a_k2)
#%% Coherence 1981
# Low Voltage Shedding vs Cost
df1string = 'Low voltage shedding'
df2string = 'Cost'
yearstring = '1981'
df1 = VTlow81
df2 = VTcost81
a_k1 = a_klow81df
a_k2 = a_kcost81df
lambda1 = variance_explainedlow81
lambda2 = variance_explainedcost81
CMlowcostEig81 = CoherenceMatrixEigen(df1,df2)
CMlowcostRel81 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowcostAMP81 = CoherenceMatrixAmplitude(a_k1,a_k2)

# heat load shedding vs cost
df1string = 'Heat load shedding'
df2string = 'Cost'
yearstring = '1981'
df1 = VTheat81
df2 = VTcost81
a_k1 = a_kheat81df
a_k2 = a_kcost81df
lambda1 = variance_explainedheat81
lambda2 = variance_explainedcost81
CMheatcostEig81 = CoherenceMatrixEigen(df1,df2)
CMheatcostRel81 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMheatcostAMP81 = CoherenceMatrixAmplitude(a_k1,a_k2)


# low voltage shedding vs heat load shedding
df1string = 'Low voltage shedding'
df2string = 'Heat load shedding'
yearstring = '1981'
df1 = VTlow81
df2 = VTheat81
a_k1 = a_klow81df
a_k2 = a_kheat81df
lambda1 = variance_explainedlow81
lambda2 = variance_explainedheat81
CMlowheatEig81 = CoherenceMatrixEigen(df1,df2)
CMlowheatRel81 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMlowheatAMP81 = CoherenceMatrixAmplitude(a_k1,a_k2)
#%%
def CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014):
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
    
   
    # PC1 ones coherence
    C1 = [C1_1998,C1_1999,C1_2000,C1_1987,C1_2013,C1_2014]
    C2 = [C2_1998,C2_1999,C2_2000,C2_1987,C2_2013,C2_2014]
    C3 = [C3_1998,C3_1999,C3_2000,C3_1987,C3_2013,C3_2014]
    
    return (C1,C2,C3)
#%% Spatial coherence
# Low voltage and cost
C1_lowcostEig, C2_lowcostEig, C3_lowcostEig = CoherenceVector(CMlowcostEig65,CMlowcostEig62,CMlowcostEig81,CMlowcostEig06,CMlowcostEig17,CMlowcostEig14)

# Heat and cost
C1_heatcostEig, C2_heatcostEig, C3_heatcostEig = CoherenceVector(CMheatcostEig65,CMheatcostEig62,CMheatcostEig81,CMheatcostEig06,CMheatcostEig17,CMheatcostEig14)

# low voltage and heat load shedding
C1_lowheatEig, C2_lowheatEig, C3_lowheatEig = CoherenceVector(CMlowheatEig65,CMlowheatEig62,CMlowheatEig81,CMlowheatEig06,CMlowheatEig17,CMlowheatEig14)

#%% Spatial coherence relative
# Low voltage and cost
C1_lowcostEigRel, C2_lowcostEigRel, C3_lowcostEigRel = CoherenceVector(CMlowcostRel65,CMlowcostRel62,CMlowcostRel81,CMlowcostRel06,CMlowcostRel17,CMlowcostRel14)

# Heat and cost
C1_heatcostEigRel, C2_heatcostEigRel, C3_heatcostEigRel = CoherenceVector(CMheatcostRel65,CMheatcostRel62,CMheatcostRel81,CMheatcostRel06,CMheatcostRel17,CMheatcostRel14)

# low voltage and heat load shedding
C1_lowheatEigRel, C2_lowheatEigRel, C3_lowheatEigRel = CoherenceVector(CMlowheatRel65,CMlowheatRel62,CMlowheatRel81,CMlowheatRel06,CMlowheatRel17,CMlowheatRel14)
#%% Temporal Coherence
# Low voltage and cost
C1_lowcostEigAMP, C2_lowcostEigAMP, C3_lowcostEigAMP = CoherenceVector(CMlowcostAMP65,CMlowcostAMP62,CMlowcostAMP81,CMlowcostAMP06,CMlowcostAMP17,CMlowcostAMP14)

# Heat and cost
C1_heatcostEigAMP, C2_heatcostEigAMP, C3_heatcostEigAMP = CoherenceVector(CMheatcostAMP65,CMheatcostAMP62,CMheatcostAMP81,CMheatcostAMP06,CMheatcostAMP17,CMheatcostAMP14)

# low voltage and heat load shedding
C1_lowheatEigAMP, C2_lowheatEigAMP, C3_lowheatEigAMP = CoherenceVector(CMlowheatAMP65,CMlowheatAMP62,CMlowheatAMP81,CMlowheatAMP06,CMlowheatAMP17,CMlowheatAMP14)

#%% Coherence plot
fig, ax = plt.subplots(3,1,figsize=(12,6))
x_ax = ['1965','1962','1981','2006','2017','2014']
ax[0].plot(x_ax,C1_lowcostEig,'o-',color='darkblue',label='Low Voltage Shedding vs. Cost')
ax[0].plot(x_ax,C1_heatcostEig,'o-',color='dodgerblue',label='Heat Load Shedding vs. Cost')
ax[0].plot(x_ax,C1_lowheatEig,'o-',color='darkgreen',label='Low Voltage Shedding vs. Heat Load Shedding')
ax[1].plot(x_ax,C1_lowcostEigRel,'o-',color='darkblue',label='Low Voltage Shedding vs. Cost')
ax[1].plot(x_ax,C1_heatcostEigRel,'o-',color='dodgerblue',label='Heat Load Shedding vs. Cost')
ax[1].plot(x_ax,C1_lowheatEigRel,'o-',color='darkgreen',label='Low Voltage Shedding vs. Heat Load Shedding')
ax[2].plot(x_ax,C1_lowcostEigAMP,'o-',color='darkblue',label='Low Voltage Shedding vs. Cost')
ax[2].plot(x_ax,C1_heatcostEigAMP,'o-',color='dodgerblue',label='Heat Load Shedding vs. Cost')
ax[2].plot(x_ax,C1_lowheatEigAMP,'o-',color='darkgreen',label='Low Voltage Shedding vs. Heat Load Shedding')
ax[0].set_title('Coherence: $c^{(1)}$',weight='bold')
ax[1].set_title('Coherence: $c^{(2)}$',weight='bold')
ax[2].set_title('Coherence: $c^{(3)}$',weight='bold')
ax[0].set_ylabel('Coherence PC1',fontsize = 12,weight = 'bold')
ax[1].set_ylabel('Coherence PC1',fontsize = 12,weight = 'bold')
ax[2].set_ylabel('Coherence PC1',fontsize = 12,weight = 'bold')
ax[2].set_xlabel('Years - [Cheapest to most expensive]',fontsize = 12,weight = 'bold')
ax[0].tick_params(axis='both',labelsize=12)
ax[1].tick_params(axis='both',labelsize=12)
ax[2].tick_params(axis='both',labelsize=12)
ax[0].legend(loc='lower left',framealpha=0.4)
plt.suptitle('Coherence for PC1', fontsize = 16, weight = 'bold',y=0.95)


fig.tight_layout()
savefigure('System', 'NEWNETWORK', 'Coherence', fig)

