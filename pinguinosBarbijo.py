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

import seaborn as sns

import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

import shapely.geometry as sgeom
#%%Configuration
os.chdir('/Volumes/MUSI-HAH/TFM/penguin_data/nombres_unificados/')


#%% Functions

def load_data(file):
    # Loading data
    penguin = pd.read_csv(file, delim_whitespace=True, lineterminator='\n', header=None)
    # Rename columns
    penguin = penguin.rename(columns= {0:"name", 1:"date",2:"time", 3:"undef1", 4:"undef2",
                             5:"undef3", 6:"active_dry", 7:"depth", 8:"temp",
                             9:"lon", 10:"lat", 11:"undef4",  12:"undef5",
                             13:"undef6", 14:"undef7", 15:"undef8", 16: "volt"})
    # Select useful columns
    penguin = penguin [["name", "date", "time", "depth", "temp", "lon", "lat"]]
    return penguin


def parse_dates(penguin):
    # Parse dates
    penguin ['datetime'] = penguin['date'] + ' ' + penguin['time']
    penguin ['datetime'] = pd.to_datetime(penguin['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
    return penguin


def calcule_velocity (penguin):
    # Calcule of time delta between points
    penguin['delta_time'] = penguin.datetime.diff()
    # Matrix of spatial diferences
    matrix_cdist = cdist(penguin.loc[0:5000,['lat','lon']].to_numpy(),penguin.loc[1:5001,['lat','lon']].to_numpy())
    # Select diagonal of spatial diferences matrix
    diagonal_plus1 = np.diagonal(matrix_cdist, offset = 1) # array[i,i+1]
    penguin['delta_space'] = diagonal_plus1
    #Calcule velotity column
    penguin['velocity'] = penguin['delta_space']/penguin['delta_time']
    return penguin


def extract_trip_number(filename):
    trip_number = int(filename.split('_')[0].split('viaje')[1])
    return trip_number


def extract_peng_number(filename):
    peng_number = int(filename.split('_')[1].split('.')[0].split('newpeng')[1])
    return peng_number


def save_boxplot(penguin_number, penguin_data):
    # plot
    boxplot = sns.boxplot(x=penguin_data['velocity'])
    # create figure
    fig = boxplot.get_figure()
    filename = './figures/' + penguin_number + '.png'
    # save figure
    fig.savefig(filename)
    
    
# def delete_velocity_outliers(penguin):
#     mean = penguin['velocity'].mean()
#     sigma = penguin['velocity'].std()
#     sigma3 = 3*sigma
#     penguin.loc[(penguin['velocity'] <= mean + sigma3) | (penguin['velocity'] >= mean - sigma3),:]

#%%
file = 'viaje2_newpeng03.csv'
# Parse data
penguin = load_data(file)
penguin = parse_dates(file)
penguin = calcule_velocity (penguin)

# Penguin data
trip_number = extract_trip_number(file)
peng_number = extract_peng_number(file)

# Add penguin data to dataframe
penguin['trip'] = trip_number
penguin['peng_number'] = peng_number

# Outlier detection and removal
save_boxplot(peng_number, penguin_data)
# penguin = delete_velocity_outliers(penguin)








# #%% Track

# lons = penguin ['lon']
# lats = penguin ['lat']

# track = sgeom.LineString(zip(lons, lats))

# #%% Plot
# lonW = -62.9
# lonE = -60
# latS = -63
# latN = -60

# fig = plt.figure(figsize=(20,10))
# ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

# #ax1.contourf(lonSLA,latSLA,SLAmean_90, cmap='YlOrRd', extend='both', levels=cflevels)

# ax.set_extent([lonW, lonE, latS, latN])
# #ax.add_feature(cfeature.GSHHSFeature(levels = [5,6],scale='full',facecolor='silver'))
# #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ne_10m_minor_islands_coastline', scale ='10m')) #ne_10m_minor_islands_coastline
# ax.coastlines(resolution ='10m')
# ax.set_xticks(np.arange(lonW,lonE,5), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(latS,latN,5), crs=ccrs.PlateCarree())
# ax.set_title('Penguin track',fontsize=18)
# ax.set_ylabel('Latitude',fontsize=16)
# ax.set_xlabel('Longitude',fontsize=16)

# ax.add_geometries([track], ccrs.PlateCarree(),facecolor='none', edgecolor='red')