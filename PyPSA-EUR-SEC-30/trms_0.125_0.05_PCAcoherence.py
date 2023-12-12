# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:52:39 2023

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

from FunctionsPM import CoherenceMatrixAmplitude
from FunctionsPM import CoherenceMatrixEigen
from FunctionsPM import CoherenceMatrixEigenRel
from FunctionsPM import HeatMapCoherencePlot
from FunctionsPM import CoherenceVector
from FunctionsPM import CoherenceYearsPlot

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

#%% Gas load
OCGTY1998 = network1998.links_t.p0.filter(regex='OCGT')
OCGTY1999 = network1999.links_t.p0.filter(regex='OCGT')
OCGTY2000 = network2000.links_t.p0.filter(regex='OCGT')
OCGTY1987 = network1987.links_t.p0.filter(regex='OCGT')
OCGTY2013 = network2013.links_t.p0.filter(regex='OCGT')
OCGTY2014 = network2014.links_t.p0.filter(regex='OCGT')
#%% Hydrogen load
# 1998
H21998Elec = network1998.links_t.p0.filter(regex = 'Electrolysis')
H21998Elec.columns = NAMES
H21998Fuel = (-network1998.links_t.p1.filter(regex = 'Fuel Cell'))
H21998Fuel.columns = NAMES
H21998 = H21998Elec-H21998Fuel

# 1999
H21999Elec = network1999.links_t.p0.filter(regex = 'Electrolysis')
H21999Elec.columns = NAMES
H21999Fuel = (-network1999.links_t.p1.filter(regex = 'Fuel Cell'))
H21999Fuel.columns = NAMES
H21999 = H21999Elec-H21999Fuel

# 2000
H22000Elec = network2000.links_t.p0.filter(regex = 'Electrolysis')
H22000Elec.columns = NAMES
H22000Fuel = (-network2000.links_t.p1.filter(regex = 'Fuel Cell'))
H22000Fuel.columns = NAMES
H22000 = H22000Elec-H22000Fuel

# 1987
H21987Elec = network1987.links_t.p0.filter(regex = 'Electrolysis')
H21987Elec.columns = NAMES
H21987Fuel = (-network1987.links_t.p1.filter(regex = 'Fuel Cell'))
H21987Fuel.columns = NAMES
H21987 = H21987Elec-H21987Fuel

# 2013
H22013Elec = network2013.links_t.p0.filter(regex = 'Electrolysis')
H22013Elec.columns = NAMES
H22013Fuel = (-network2013.links_t.p1.filter(regex = 'Fuel Cell'))
H22013Fuel.columns = NAMES
H22013 = H22013Elec-H22013Fuel

# 2014
H22014Elec = network2014.links_t.p0.filter(regex = 'Electrolysis')
H22014Elec.columns = NAMES
H22014Fuel = (-network2014.links_t.p1.filter(regex = 'Fuel Cell'))
H22014Fuel.columns = NAMES
H22014 = H22014Elec-H22014Fuel
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

#%% PCA
# PCA GAS
eigen_values1998gas, eigen_vectors1998gas, Cov_mat1998gas, c1998gas, variance_explained1998gas, VT1998gas, X_norm1998gas = PCA(OCGTY1998,NAMES)
eigen_values1999gas, eigen_vectors1999gas, Cov_mat1999gas, c1999gas, variance_explained1999gas, VT1999gas, X_norm1999gas = PCA(OCGTY1999,NAMES)
eigen_values2000gas, eigen_vectors2000gas, Cov_mat2000gas, c2000gas, variance_explained2000gas, VT2000gas, X_norm2000gas = PCA(OCGTY2000,NAMES)
eigen_values1987gas, eigen_vectors1987gas, Cov_mat1987gas, c1987gas, variance_explained1987gas, VT1987gas, X_norm1987gas = PCA(OCGTY1987,NAMES)
eigen_values2013gas, eigen_vectors2013gas, Cov_mat2013gas, c2013gas, variance_explained2013gas, VT2013gas, X_norm2013gas = PCA(OCGTY2013,NAMES)
eigen_values2014gas, eigen_vectors2014gas, Cov_mat2014gas, c2014gas, variance_explained2014gas, VT2014gas, X_norm2014gas = PCA(OCGTY2014,NAMES)


VT1999gas[[1,2]] = -1*VT1999gas[[1,2]]
VT2000gas[[0,2]] = -1*VT2000gas[[0,2]]
VT1987gas[[0,2,3]] = -1*VT1987gas[[0,2,3]]
VT2013gas[[2,3]] = -1*VT2013gas[[2,3]]
VT2014gas[[1,2]] = -1*VT2014gas[[1,2]]

# Calculate Amplitudes of eigenvectors
a_k1998gas = np.dot(X_norm1998gas,VT1998gas)
a_k1999gas = np.dot(X_norm1999gas,VT1999gas)
a_k2000gas = np.dot(X_norm2000gas,VT2000gas)
a_k1987gas = np.dot(X_norm1987gas,VT1987gas)
a_k2013gas = np.dot(X_norm2013gas,VT2013gas)
a_k2014gas = np.dot(X_norm2014gas,VT2014gas)
# data frames
a_k1998dfgas = pd.DataFrame(data=a_k1998gas,index=time_index)
a_k1999dfgas = pd.DataFrame(data=a_k1999gas,index=time_index)
a_k2000dfgas = pd.DataFrame(data=a_k2000gas,index=time_index)
a_k1987dfgas = pd.DataFrame(data=a_k1987gas,index=time_index)
a_k2013dfgas = pd.DataFrame(data=a_k2013gas,index=time_index)
a_k2014dfgas = pd.DataFrame(data=a_k2014gas,index=time_index)

# PCA Hydrogen
eigen_values1998h2, eigen_vectors1998h2, Cov_mat1998h2, c1998h2, variance_explained1998h2, VT1998h2, X_norm1998h2 = PCA(H21998,NAMES)
eigen_values1999h2, eigen_vectors1999h2, Cov_mat1999h2, c1999h2, variance_explained1999h2, VT1999h2, X_norm1999h2 = PCA(H21999,NAMES)
eigen_values2000h2, eigen_vectors2000h2, Cov_mat2000h2, c2000h2, variance_explained2000h2, VT2000h2, X_norm2000h2 = PCA(H22000,NAMES)
eigen_values1987h2, eigen_vectors1987h2, Cov_mat1987h2, c1987h2, variance_explained1987h2, VT1987h2, X_norm1987h2 = PCA(H21987,NAMES)
eigen_values2013h2, eigen_vectors2013h2, Cov_mat2013h2, c2013h2, variance_explained2013h2, VT2013h2, X_norm2013h2 = PCA(H22013,NAMES)
eigen_values2014h2, eigen_vectors2014h2, Cov_mat2014h2, c2014h2, variance_explained2014h2, VT2014h2, X_norm2014h2 = PCA(H22014,NAMES)


VT1998h2[[2]] = -1*VT1998h2[[2]]
VT1999h2[[1]] = -1*VT1999h2[[1]]
VT2000h2[[0,1]] = -1*VT2000h2[[0,1]]
VT1987h2[[0,1,3]] = -1*VT1987h2[[0,1,3]]
# VT2013h2[[2,3]] = -1*VT2013h2[[2,3]]
VT2014h2[[1,3]] = -1*VT2014h2[[1,3]]

# Calculate Amplitudes of eigenvectors
a_k1998h2 = np.dot(X_norm1998h2,VT1998h2)
a_k1999h2 = np.dot(X_norm1999h2,VT1999h2)
a_k2000h2 = np.dot(X_norm2000h2,VT2000h2)
a_k1987h2 = np.dot(X_norm1987h2,VT1987h2)
a_k2013h2 = np.dot(X_norm2013h2,VT2013h2)
a_k2014h2 = np.dot(X_norm2014h2,VT2014h2)
# data frames
a_k1998dfh2 = pd.DataFrame(data=a_k1998h2,index=time_index)
a_k1999dfh2 = pd.DataFrame(data=a_k1999h2,index=time_index)
a_k2000dfh2 = pd.DataFrame(data=a_k2000h2,index=time_index)
a_k1987dfh2 = pd.DataFrame(data=a_k1987h2,index=time_index)
a_k2013dfh2 = pd.DataFrame(data=a_k2013h2,index=time_index)
a_k2014dfh2 = pd.DataFrame(data=a_k2014h2,index=time_index)
# PCA nodel prices or marginal prices
eigen_values1998price, eigen_vectors1998price, Cov_mat1998price, c1998price, variance_explained1998price, VT1998price, X_norm1998price = PCA(eprice1998,NAMES)
eigen_values1999price, eigen_vectors1999price, Cov_mat1999price, c1999price, variance_explained1999price, VT1999price, X_norm1999price = PCA(eprice1999,NAMES)
eigen_values2000price, eigen_vectors2000price, Cov_mat2000price, c2000price, variance_explained2000price, VT2000price, X_norm2000price = PCA(eprice2000,NAMES)
eigen_values1987price, eigen_vectors1987price, Cov_mat1987price, c1987price, variance_explained1987price, VT1987price, X_norm1987price = PCA(eprice1987,NAMES)
eigen_values2013price, eigen_vectors2013price, Cov_mat2013price, c2013price, variance_explained2013price, VT2013price, X_norm2013price = PCA(eprice2013,NAMES)
eigen_values2014price, eigen_vectors2014price, Cov_mat2014price, c2014price, variance_explained2014price, VT2014price, X_norm2014price = PCA(eprice2014,NAMES)

VT1998price[[0,1,2]] = -1*VT1998price[[0,1,2]]
VT1999price[[1,2]] = -1*VT1999price[[1,2]]
VT2000price[[1,3]] = -1*VT2000price[[1,3]]
VT1987price[[0,3]] = -1*VT1987price[[0,3]]
VT2013price[[0,1]] = -1*VT2013price[[0,1]]
VT2014price[[3]] = -1*VT2014price[[3]]

# Calculate Amplitudes of eigenvectors
a_k1998price = np.dot(X_norm1998price,VT1998price)
a_k1999price = np.dot(X_norm1999price,VT1999price)
a_k2000price = np.dot(X_norm2000price,VT2000price)
a_k1987price = np.dot(X_norm1987price,VT1987price)
a_k2013price = np.dot(X_norm2013price,VT2013price)
a_k2014price = np.dot(X_norm2014price,VT2014price)
# data frames
a_k1998dfprice = pd.DataFrame(data=a_k1998price,index=time_index)
a_k1999dfprice = pd.DataFrame(data=a_k1999price,index=time_index)
a_k2000dfprice = pd.DataFrame(data=a_k2000price,index=time_index)
a_k1987dfprice = pd.DataFrame(data=a_k1987price,index=time_index)
a_k2013dfprice = pd.DataFrame(data=a_k2013price,index=time_index)
a_k2014dfprice = pd.DataFrame(data=a_k2014price,index=time_index)

#%%
plt.figure(figsize=(12,6))
plt.scatter(c1998gas.index,c1998gas, color = 'green', label='1998')
plt.scatter(c1999gas.index,c1999gas,color = 'blue', label='1999')
plt.scatter(c1999gas.index,c2000gas,color = 'purple', label='2000')
plt.scatter(c1999gas.index,c2014gas,color = 'red', label='2014')
plt.scatter(c1999gas.index,c2013gas,color = 'darkred', label='2013')
plt.scatter(c1999gas.index,c1987gas,color = 'pink', label='1987')
plt.title('PCA - Gas constants')
plt.legend()
plt.figure(figsize=(12,6))
plt.scatter(c1998gas.index,c1998h2, color = 'green', label='1998')
plt.scatter(c1999gas.index,c1999h2,color = 'blue', label='1999')
plt.scatter(c1999gas.index,c2000h2,color = 'purple', label='2000')
plt.scatter(c1999gas.index,c2014h2,color = 'red', label='2014')
plt.scatter(c1999gas.index,c2013h2,color = 'darkred', label='2013')
plt.scatter(c1999gas.index,c1987h2,color = 'pink', label='1987')
plt.title('PCA - H2 constants')
plt.legend()
#%% Coherence 1998
# Gas vs Eprice
df1string = 'Gas Usage'
df2string = 'Electricity Price'
yearstring = '1998'
df1 = VT1998gas
df2 = VT1998price
a_k1 = a_k1998dfgas
a_k2 = a_k1998dfprice
lambda1 = variance_explained1998gas
lambda2 = variance_explained1998price
CMgaspriceEig1998 = CoherenceMatrixEigen(df1,df2)
CMgaspriceRel1998 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgaspriceAMP1998 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgaspriceEig1998,CMgaspriceRel1998,CMgaspriceAMP1998,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Gas vs h2
df1string = 'Gas Usage'
df2string = 'H2'
yearstring = '1998'
df1 = VT1998gas
df2 = VT1998h2
a_k1 = a_k1998dfgas
a_k2 = a_k1998dfh2
lambda1 = variance_explained1998gas
lambda2 = variance_explained1998h2
CMgash2Eig1998 = CoherenceMatrixEigen(df1,df2)
CMgash2Rel1998 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgash2AMP1998 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgash2Eig1998,CMgash2Rel1998,CMgash2AMP1998,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Price vs h2
df1string = 'Electricity Price'
df2string = 'H2'
yearstring = '1998'
df1 = VT1998price
df2 = VT1998h2
a_k1 = a_k1998dfprice
a_k2 = a_k1998dfh2
lambda1 = variance_explained1998price
lambda2 = variance_explained1998h2
CMpriceh2Eig1998 = CoherenceMatrixEigen(df1,df2)
CMpriceh2Rel1998 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMpriceh2AMP1998 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMpriceh2Eig1998,CMpriceh2Rel1998,CMpriceh2AMP1998,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

#%% Coherence 1999

# Gas vs Eprice
df1string = 'Gas Usage'
df2string = 'Electricity Price'
yearstring = '1999'
df1 = VT1999gas
df2 = VT1999price
a_k1 = a_k1999dfgas
a_k2 = a_k1999dfprice
lambda1 = variance_explained1999gas
lambda2 = variance_explained1999price
CMgaspriceEig1999 = CoherenceMatrixEigen(df1,df2)
CMgaspriceRel1999 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgaspriceAMP1999 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgaspriceEig1999,CMgaspriceRel1999,CMgaspriceAMP1999,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Gas vs h2
df1string = 'Gas Usage'
df2string = 'H2'
yearstring = '1999'
df1 = VT1999gas
df2 = VT1999h2
a_k1 = a_k1999dfgas
a_k2 = a_k1999dfh2
lambda1 = variance_explained1999gas
lambda2 = variance_explained1999h2
CMgash2Eig1999 = CoherenceMatrixEigen(df1,df2)
CMgash2Rel1999 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgash2AMP1999 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgash2Eig1999,CMgash2Rel1999,CMgash2AMP1999,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Price vs h2
df1string = 'Electricity Price'
df2string = 'H2'
yearstring = '1999'
df1 = VT1999price
df2 = VT1999h2
a_k1 = a_k1999dfprice
a_k2 = a_k1999dfh2
lambda1 = variance_explained1999price
lambda2 = variance_explained1999h2
CMpriceh2Eig1999 = CoherenceMatrixEigen(df1,df2)
CMpriceh2Rel1999 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMpriceh2AMP1999 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMpriceh2Eig1999,CMpriceh2Rel1999,CMpriceh2AMP1999,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

#%% Coherence 2000

# Gas vs Eprice
df1string = 'Gas Usage'
df2string = 'Electricity Price'
yearstring = '2000'
df1 = VT2000gas
df2 = VT2000price
a_k1 = a_k2000dfgas
a_k2 = a_k2000dfprice
lambda1 = variance_explained2000gas
lambda2 = variance_explained2000price
CMgaspriceEig2000 = CoherenceMatrixEigen(df1,df2)
CMgaspriceRel2000 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgaspriceAMP2000 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgaspriceEig2000,CMgaspriceRel2000,CMgaspriceAMP2000,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Gas vs h2
df1string = 'Gas Usage'
df2string = 'H2'
yearstring = '2000'
df1 = VT2000gas
df2 = VT2000h2
a_k1 = a_k2000dfgas
a_k2 = a_k2000dfh2
lambda1 = variance_explained2000gas
lambda2 = variance_explained2000h2
CMgash2Eig2000 = CoherenceMatrixEigen(df1,df2)
CMgash2Rel2000 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgash2AMP2000 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgash2Eig2000,CMgash2Rel2000,CMgash2AMP2000,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Price vs h2
df1string = 'Electricity Price'
df2string = 'H2'
yearstring = '2000'
df1 = VT2000price
df2 = VT2000h2
a_k1 = a_k2000dfprice
a_k2 = a_k2000dfh2
lambda1 = variance_explained2000price
lambda2 = variance_explained2000h2
CMpriceh2Eig2000 = CoherenceMatrixEigen(df1,df2)
CMpriceh2Rel2000 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMpriceh2AMP2000 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMpriceh2Eig2000,CMpriceh2Rel2000,CMpriceh2AMP2000,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

#%% Coherence 1987

# Gas vs Eprice
df1string = 'Gas Usage'
df2string = 'Electricity Price'
yearstring = '1987'
df1 = VT1987gas
df2 = VT1987price
a_k1 = a_k1987dfgas
a_k2 = a_k1987dfprice
lambda1 = variance_explained1987gas
lambda2 = variance_explained1987price
CMgaspriceEig1987 = CoherenceMatrixEigen(df1,df2)
CMgaspriceRel1987 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgaspriceAMP1987 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgaspriceEig1987,CMgaspriceRel1987,CMgaspriceAMP1987,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Gas vs h2
df1string = 'Gas Usage'
df2string = 'H2'
yearstring = '1987'
df1 = VT1987gas
df2 = VT1987h2
a_k1 = a_k1987dfgas
a_k2 = a_k1987dfh2
lambda1 = variance_explained1987gas
lambda2 = variance_explained1987h2
CMgash2Eig1987 = CoherenceMatrixEigen(df1,df2)
CMgash2Rel1987 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgash2AMP1987 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgash2Eig1987,CMgash2Rel1987,CMgash2AMP1987,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Price vs h2
df1string = 'Electricity Price'
df2string = 'H2'
yearstring = '1987'
df1 = VT1987price
df2 = VT1987h2
a_k1 = a_k1987dfprice
a_k2 = a_k1987dfh2
lambda1 = variance_explained1987price
lambda2 = variance_explained1987h2
CMpriceh2Eig1987 = CoherenceMatrixEigen(df1,df2)
CMpriceh2Rel1987 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMpriceh2AMP1987 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMpriceh2Eig1987,CMpriceh2Rel1987,CMpriceh2AMP1987,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

#%% Coherence 2013

# Gas vs Eprice
df1string = 'Gas Usage'
df2string = 'Electricity Price'
yearstring = '2013'
df1 = VT2013gas
df2 = VT2013price
a_k1 = a_k2013dfgas
a_k2 = a_k2013dfprice
lambda1 = variance_explained2013gas
lambda2 = variance_explained2013price
CMgaspriceEig2013 = CoherenceMatrixEigen(df1,df2)
CMgaspriceRel2013 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgaspriceAMP2013 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgaspriceEig2013,CMgaspriceRel2013,CMgaspriceAMP2013,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Gas vs h2
df1string = 'Gas Usage'
df2string = 'H2'
yearstring = '2013'
df1 = VT2013gas
df2 = VT2013h2
a_k1 = a_k2013dfgas
a_k2 = a_k2013dfh2
lambda1 = variance_explained2013gas
lambda2 = variance_explained2013h2
CMgash2Eig2013 = CoherenceMatrixEigen(df1,df2)
CMgash2Rel2013 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgash2AMP2013 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgash2Eig2013,CMgash2Rel2013,CMgash2AMP2013,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Price vs h2
df1string = 'Electricity Price'
df2string = 'H2'
yearstring = '2013'
df1 = VT2013price
df2 = VT2013h2
a_k1 = a_k2013dfprice
a_k2 = a_k2013dfh2
lambda1 = variance_explained2013price
lambda2 = variance_explained2013h2
CMpriceh2Eig2013 = CoherenceMatrixEigen(df1,df2)
CMpriceh2Rel2013 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMpriceh2AMP2013 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMpriceh2Eig2013,CMpriceh2Rel2013,CMpriceh2AMP2013,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

#%% Coherence 2014

# Gas vs Eprice
df1string = 'Gas Usage'
df2string = 'Electricity Price'
yearstring = '2014'
df1 = VT2014gas
df2 = VT2014price
a_k1 = a_k2014dfgas
a_k2 = a_k2014dfprice
lambda1 = variance_explained2014gas
lambda2 = variance_explained2014price
CMgaspriceEig2014 = CoherenceMatrixEigen(df1,df2)
CMgaspriceRel2014 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgaspriceAMP2014 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgaspriceEig2014,CMgaspriceRel2014,CMgaspriceAMP2014,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Gas vs h2
df1string = 'Gas Usage'
df2string = 'H2'
yearstring = '2014'
df1 = VT2014gas
df2 = VT2014h2
a_k1 = a_k2014dfgas
a_k2 = a_k2014dfh2
lambda1 = variance_explained2014gas
lambda2 = variance_explained2014h2
CMgash2Eig2014 = CoherenceMatrixEigen(df1,df2)
CMgash2Rel2014 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMgash2AMP2014 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMgash2Eig2014,CMgash2Rel2014,CMgash2AMP2014,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()

# Price vs h2
df1string = 'Electricity Price'
df2string = 'H2'
yearstring = '2014'
df1 = VT2014price
df2 = VT2014h2
a_k1 = a_k2014dfprice
a_k2 = a_k2014dfh2
lambda1 = variance_explained2014price
lambda2 = variance_explained2014h2
CMpriceh2Eig2014 = CoherenceMatrixEigen(df1,df2)
CMpriceh2Rel2014 = CoherenceMatrixEigenRel(df1,df2,lambda1,lambda2)
CMpriceh2AMP2014 = CoherenceMatrixAmplitude(a_k1,a_k2)
fig = HeatMapCoherencePlot(CMpriceh2Eig2014,CMpriceh2Rel2014,CMpriceh2AMP2014,df1string,df2string,yearstring,constraint3)
savefigure1('System', constraint3, 'COHE'+df1string+df2string+yearstring, fig) # OBS remember change savefigure
plt.rcdefaults()


#%%

Years = [1998,1999,2000,1987,2013,2014] # Remember to change when constraint changes

# Gas Price coherence
String1 = 'Gas Usage'
String2 = 'Electricity Price'
# Eigenvectors
CohMat1998 = CMgaspriceEig1998
CohMat1999 = CMgaspriceEig1999
CohMat2000 = CMgaspriceEig2000
CohMat1987 = CMgaspriceEig1987
CohMat2013 = CMgaspriceEig2013
CohMat2014 = CMgaspriceEig2014

C1_gaspriceEig, C2_gaspriceEig, C3_gaspriceEig = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)

# Relative Strength Eigenvectors
CohMat1998 = CMgaspriceRel1998
CohMat1999 = CMgaspriceRel1999
CohMat2000 = CMgaspriceRel2000
CohMat1987 = CMgaspriceRel1987
CohMat2013 = CMgaspriceRel2013
CohMat2014 = CMgaspriceRel2014

C1_gaspriceRel, C2_gaspriceRel, C3_gaspriceRel = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)

# Amplitude
CohMat1998 = CMgaspriceAMP1998
CohMat1999 = CMgaspriceAMP1999
CohMat2000 = CMgaspriceAMP2000
CohMat1987 = CMgaspriceAMP1987
CohMat2013 = CMgaspriceAMP2013
CohMat2014 = CMgaspriceAMP2014

C1_gaspriceAMP, C2_gaspriceAMP, C3_gaspriceAMP = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)


C1_eig = C1_gaspriceEig
C2_eig = C2_gaspriceEig
C3_eig = C3_gaspriceEig

C1_rel = C1_gaspriceRel
C2_rel = C2_gaspriceRel
C3_rel = C3_gaspriceRel

C1_amp = C1_gaspriceAMP
C2_amp = C2_gaspriceAMP
C3_amp = C3_gaspriceAMP

fig = CoherenceYearsPlot(C1_eig,C2_eig,C3_eig,C1_rel,C2_rel,C3_rel,C1_amp,C2_amp,C3_amp,constraint3,Years)
savefigure1('System', constraint3, 'COHyears'+String1+String2, fig) # OBS remember change savefigure

# Gas H2 coherence
String1 = 'Gas Usage'
String2 = 'H2'
# Eigenvectors
CohMat1998 = CMgash2Eig1998
CohMat1999 = CMgash2Eig1999
CohMat2000 = CMgash2Eig2000
CohMat1987 = CMgash2Eig1987
CohMat2013 = CMgash2Eig2013
CohMat2014 = CMgash2Eig2014

C1_gash2Eig, C2_gash2Eig, C3_gash2Eig = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)

# Relative Strength Eigenvectors
CohMat1998 = CMgash2Rel1998
CohMat1999 = CMgash2Rel1999
CohMat2000 = CMgash2Rel2000
CohMat1987 = CMgash2Rel1987
CohMat2013 = CMgash2Rel2013
CohMat2014 = CMgash2Rel2014

C1_gash2Rel, C2_gash2Rel, C3_gash2Rel = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)

# Amplitude
CohMat1998 = CMgash2AMP1998
CohMat1999 = CMgash2AMP1999
CohMat2000 = CMgash2AMP2000
CohMat1987 = CMgash2AMP1987
CohMat2013 = CMgash2AMP2013
CohMat2014 = CMgash2AMP2014

C1_gash2AMP, C2_gash2AMP, C3_gash2AMP = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)


C1_eig = C1_gash2Eig
C2_eig = C2_gash2Eig
C3_eig = C3_gash2Eig

C1_rel = C1_gash2Rel
C2_rel = C2_gash2Rel
C3_rel = C3_gash2Rel

C1_amp = C1_gash2AMP
C2_amp = C2_gash2AMP
C3_amp = C3_gash2AMP

fig = CoherenceYearsPlot(C1_eig,C2_eig,C3_eig,C1_rel,C2_rel,C3_rel,C1_amp,C2_amp,C3_amp,constraint3,Years)
savefigure1('System', constraint3, 'COHyears'+String1+String2, fig) # OBS remember change savefigure

# Price H2 coherence
String1 = 'Price'
String2 = 'H2'

# Eigenvectors
CohMat1998 = CMpriceh2Eig1998
CohMat1999 = CMpriceh2Eig1999
CohMat2000 = CMpriceh2Eig2000
CohMat1987 = CMpriceh2Eig1987
CohMat2013 = CMpriceh2Eig2013
CohMat2014 = CMpriceh2Eig2014

C1_priceh2Eig, C2_priceh2Eig, C3_priceh2Eig = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)

# Relative Strength Eigenvectors
CohMat1998 = CMpriceh2Rel1998
CohMat1999 = CMpriceh2Rel1999
CohMat2000 = CMpriceh2Rel2000
CohMat1987 = CMpriceh2Rel1987
CohMat2013 = CMpriceh2Rel2013
CohMat2014 = CMpriceh2Rel2014

C1_priceh2Rel, C2_priceh2Rel, C3_priceh2Rel = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)

# Amplitude
CohMat1998 = CMpriceh2AMP1998
CohMat1999 = CMpriceh2AMP1999
CohMat2000 = CMpriceh2AMP2000
CohMat1987 = CMpriceh2AMP1987
CohMat2013 = CMpriceh2AMP2013
CohMat2014 = CMpriceh2AMP2014

C1_priceh2AMP, C2_priceh2AMP, C3_priceh2AMP = CoherenceVector(CohMat1998,CohMat1999,CohMat2000,CohMat1987,CohMat2013,CohMat2014, Years)


C1_eig = C1_priceh2Eig
C2_eig = C2_priceh2Eig
C3_eig = C3_priceh2Eig

C1_rel = C1_priceh2Rel
C2_rel = C2_priceh2Rel
C3_rel = C3_priceh2Rel

C1_amp = C1_priceh2AMP
C2_amp = C2_priceh2AMP
C3_amp = C3_priceh2AMP

fig = CoherenceYearsPlot(C1_eig,C2_eig,C3_eig,C1_rel,C2_rel,C3_rel,C1_amp,C2_amp,C3_amp,constraint3,Years)
savefigure1('System', constraint3, 'COHyears'+String1+String2, fig) # OBS remember change savefigure
#%%

    
