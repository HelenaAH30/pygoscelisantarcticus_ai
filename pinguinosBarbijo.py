#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:20:29 2020

@author: Helena
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

import shapely.geometry as sgeom
#%%Configuration
os.chdir('/Volumes/MUSI-HAH/TFM/penguin_data/nombres_unificados/')


#%% Loading data
penguin = pd.read_csv('viaje1_newpeng01_nido94.csv', delim_whitespace=True, lineterminator='\n', header=None)

penguin = penguin.rename(columns= {0:"name", 1:"date",2:"time", 3:"undef1", 4:"undef2",
                         5:"undef3", 6:"active_dry", 7:"depth", 8:"temp",
                         9:"lon", 10:"lat", 11:"undef4",  12:"undef5",
                         13:"undef6", 14:"undef7", 15:"undef8", 16: "volt"})
penguin = penguin [["name", "date", "time", "active_dry", "depth", "temp", "lon", "lat"]]

# df = penguin.drop_duplicates(subset = ["name", "date", "active_dry", "depth", "temp", "lon", "lat"],
#                              keep='first', ignore_index=True)

#%% Parse dates
penguin ['datetime'] = penguin['date'] + ' ' + penguin['time']
penguin ['datetime'] = pd.to_datetime(penguin['datetime'], format='%d/%m/%Y %H:%M:%S.%f')

#%% Calcule delta time and space

penguin['delta_time'] = penguin.datetime.diff()
matrix_cdist = cdist(penguin.loc[0:5000,['lat','lon']].to_numpy(),penguin.loc[1:5001,['lat','lon']].to_numpy())
# TODO: comprobar
diagonal_plus1 = np.diagonal(matrix_cdist, offset = 1) # array[i,i+1]
penguin['delta_space'] = diagonal_plus1

penguin['velocity'] = penguin['delta_space']/penguin['delta_time']
#%% Outliers
sns.boxplot(x=penguin['velocity'])

mean = penguin['velocity'].mean()
sigma = penguin['velocity'].std()
3sigma = 3*sigma




#%% Track

lons = penguin ['lon']
lats = penguin ['lat']

track = sgeom.LineString(zip(lons, lats))

#%% Plot
lonW = -62.9
lonE = -60
latS = -63
latN = -60

fig = plt.figure(figsize=(20,10))
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

#ax1.contourf(lonSLA,latSLA,SLAmean_90, cmap='YlOrRd', extend='both', levels=cflevels)

ax.set_extent([lonW, lonE, latS, latN])
#ax.add_feature(cfeature.GSHHSFeature(levels = [5,6],scale='full',facecolor='silver'))
#ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ne_10m_minor_islands_coastline', scale ='10m')) #ne_10m_minor_islands_coastline
ax.coastlines(resolution ='10m')
ax.set_xticks(np.arange(lonW,lonE,5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(latS,latN,5), crs=ccrs.PlateCarree())
ax.set_title('Penguin track',fontsize=18)
ax.set_ylabel('Latitude',fontsize=16)
ax.set_xlabel('Longitude',fontsize=16)

ax.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='red')