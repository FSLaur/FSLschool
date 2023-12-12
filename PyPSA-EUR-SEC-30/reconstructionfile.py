# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:32:12 2023

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
from FunctionsPM import AmplitudeAreaHour
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
#%%
time_index = network1998.loads_t.p.index
#%%
OCGTY1998 = network1998.links_t.p0.filter(regex='OCGT')
eigen_values1998, eigen_vectors1998, Cov_mat1998, c1998, variance_explained1998, VT1998, X_norm1998 = PCA(OCGTY1998,NAMES)
#%% AMP
# Z
a_bar = np.dot(X_norm1998,eigen_vectors1998)
#%% Reconstruction
OCGTY_rec = np.dot(a_bar,eigen_vectors1998.T)*1/c1998.values+np.mean(OCGTY1998,axis=0).values
#%% Use only 10 strongest eigenvectors truncation
# amplitude
Z = np.dot(X_norm1998,eigen_vectors1998[:,:15])

OCGTY_rec2_0 = np.dot(Z,eigen_vectors1998[:,:15].T)*1/c1998.values+np.mean(OCGTY1998,axis=0).values

#%%
plt.figure()
plt.plot(range(0,8760),OCGTY1998.mean(axis=1),color='red')
plt.plot(range(0,8760),OCGTY_rec.mean(axis=1),color='green')
plt.plot(range(0,8760),OCGTY_rec2_0.mean(axis=1),color='blue',alpha=0.3)
#%% eigenvalues summed
sum(eigen_values1998/30*100)
sum(eigen_values1998/30)

#%%
T = pd.DataFrame(data=a_bar,index=time_index)
#T_avg_hour = T.groupby(time_index.hour).mean() # Hour
T_avg_day = T.groupby([time_index.month,time_index.day]).mean() 
#%%
#OCGTY1998_daily = OCGTY1998.groupby([pd.Grouper( freq='d')]).mean()
#Amp_real_size = np.dot(T_avg_day[0].values,eigen_vectors1998[:,0].T)*np.mean(1/c1998.values)+np.mean(OCGTY1998_daily,axis=1).values
