#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:20:29 2020

@author: Helena
"""
#%% Libraries
from cmath import nan
from itertools import groupby
import os
import math
import glob
import numpy as np
import pandas as pd
import geopy.distance as gp

import seaborn as sns

import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

import shapely.geometry as sgeom
#%%Configuration
# os.chdir('/Volumes/MUSI-HAH/TFM/penguin_data/nombres_unificados/')
os.chdir('/home/helena/Documents')
_DATA_FOLDER = './nombres_unificados/'
_RESULTS_FOLDER = './results_peng/'
_NEWDATA_FOLDER = './results_peng/new_data/'


#%% Functions

def load_data(file):
    # Loading data
    filename = _DATA_FOLDER + file
    penguin = pd.read_csv(filename, delim_whitespace=True, lineterminator='\n', header=None)
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


def _distance_btwn_lonlatpoints(lon_1, lat_1, lon_2, lat_2):
    coords_1 = (lat_1,lon_1)
    coords_2 = (lat_2,lon_2)
    try:
        dist = gp.distance(coords_1, coords_2).km
        #dist = gp.vincenty(coords_1, coords_2).km
        return dist
    except: 
        return np.nan
    

def _replace_lat_outofrange(penguin):
    """
    There was values == -244.03267
    """
    penguin.lat[(penguin.lat<-90) | (penguin.lat>90)] = np.nan

    return penguin

def calcule_velocity (penguin):
    # Calcule of time delta between points
    penguin['delta_time'] = penguin.datetime.diff()
    penguin['delta_time'] = penguin.apply(lambda row: row.delta_time.total_seconds(), axis=1)

    # Calcule spatial difference between points
    penguin = _replace_lat_outofrange(penguin)
    penguin.dropna(axis=0, how='any', inplace=True)
    penguin[['lon_shift', 'lat_shift']] = penguin[['lon', 'lat']].shift(periods=1)
    penguin['delta_space'] = penguin.apply(lambda row: _distance_btwn_lonlatpoints(row.lon, row.lat, row.lon_shift, row.lat_shift), axis=1)
    #Calcule velotity column
    penguin['velocity'] = penguin['delta_space']/penguin['delta_time']
    return penguin


def extract_trip_number(filename):
    if len(filename) == 27:
        filename = filename [0:-11]
    else:
        filename = filename[21:-11] # TODO: change!!
    trip_number = int(filename.split('_')[0].split('viaje')[1])
    return trip_number


def extract_peng_number(filename):
    if len(filename) == 27:
        filename = filename [0:-11]
    else:
        filename = filename[21:-11] # TODO: change!!
    peng_number = int(filename.split('_')[1].split('.')[0].split('newpeng')[1])
    return peng_number


def save_boxplot(penguin_number, trip_number, penguin_data, string = None):
    # plot
    boxplot = sns.boxplot(x=penguin_data['velocity'])
    boxplot.set_xlabel('Velocity (m/s)')
    boxplot.set(title = f'Penguin {penguin_number} - Trip {trip_number}')
    # create figure
    fig = boxplot.get_figure()
    if string != None:
       filename = _RESULTS_FOLDER +'figures/penguin' + str(penguin_number) + '_boxplot_'+ string + '.png' 
    else:
        filename = _RESULTS_FOLDER +'figures/penguin' + str(penguin_number) + '_boxplot.png'
    # save figure
    fig.savefig(filename)
    plt.close(fig) # close the figure


def compose_statscsv(files_list):
    files_df = pd.DataFrame()
    files_df['files'] = files_list
    files_df['peng_number'] = files_df.apply(lambda row: extract_peng_number(row.files), axis=1)
    files_df['trip'] = files_df.apply(lambda row: extract_trip_number(row.files), axis=1)
    files_df = pd.DataFrame(files_df.groupby(['peng_number']).count())
    files_df['trip'].to_csv(_RESULTS_FOLDER + "files_statisticaldata.csv")
    return files_df


def save_barplot(df, string = None):
    date = pd.to_datetime('today').strftime('%Y%m%d')
    # plot
    barplot = sns.barplot(data = df, y = 'trip', x = df.index)
    barplot.set_xlabel('Penguin number')
    barplot.set_ylabel('Number of trips')
    # create figure
    fig = barplot.get_figure()
    if string != None:
       filename = _RESULTS_FOLDER +'figures/satistics_barplot_' + date +'_'+ string + '.png' 
    else:
        filename = _RESULTS_FOLDER +'figures/satistics_barplot_' + date + '.png'
    # save figure
    fig.savefig(filename)
    plt.close(fig) # close the figure


def write_txt_statistics(files_df):
    file_stats_route = _RESULTS_FOLDER + "files_statisticaldata.txt"
    file_stats = open(file_stats_route, "w")
    file_stats.write("**********************" + os.linesep)
    file_stats.write(f"Número de archivos analizados: {len(files_df)}" + os.linesep)
    file_stats.write(f"Media de viajes por pinguino: {files_df.trip.mean()}" + os.linesep)
    file_stats.write("**********************")
    file_stats.close()
   

def detect_velocity_outliers(penguin):
    mean = penguin['velocity'].mean()
    sigma = penguin['velocity'].std()
    sigma3 = 3*sigma
    penguin['outlier'] = (penguin['velocity'] >= mean + sigma3) | (penguin['velocity'] <= mean - sigma3) #element-wise | and &
    return penguin

#%%

files_list = glob.glob(_DATA_FOLDER+'*.csv')

# statistical analysis
files_df = compose_statscsv(files_list)
save_barplot(files_df)
write_txt_statistics(files_df)


file = 'viaje2_newpeng03.csv'
file = 'viaje2_newpeng03_nido75.csv'

# from multiprocessing import Pool

# if __name__ == '__main__':
#     pool = Pool(processes=6) # max RAM used = 9Gbs, 62,7/9 = 6.9067
#     pool.map(function, args)

# Parse data
penguin = load_data(file)
penguin = parse_dates(penguin)
penguin = calcule_velocity (penguin)

# Penguin data
trip_number = extract_trip_number(file)
peng_number = extract_peng_number(file)

# Add penguin data to dataframe
penguin['trip'] = trip_number
penguin['peng_number'] = peng_number

# STEP 1
save_boxplot(peng_number, trip_number, penguin, string = 'step1_nofiltered')
penguin.to_csv(_NEWDATA_FOLDER + f"penguin{peng_number:02}_trip{trip_number}_step{1}.csv")

# STEP 2
# Filter velocity=<0
penguin = penguin.loc[penguin.velocity > 0,:].reset_index()
# Filter velocity>20
penguin = penguin.loc[penguin.velocity < 20,:].reset_index()
# Outlier detection and removal
save_boxplot(peng_number, trip_number, penguin, string = 'step2_filtered')
penguin = detect_velocity_outliers(penguin)
penguin.to_csv(_NEWDATA_FOLDER + f"penguin{peng_number:02}_trip{trip_number}_step{2}.csv")

# STEP 3
penguin_out = penguin.loc[penguin.outlier !=True,:]
save_boxplot(peng_number, trip_number, penguin_out, string = 'step3_withoutoutliers')
penguin_out.to_csv(_NEWDATA_FOLDER + f"penguin{peng_number:02}_trip{trip_number}_step{3}.csv")









#%% Track

lons = penguin ['lon']
lats = penguin ['lat']

track = sgeom.LineString(zip(lons, lats))

#%% Plot
lonW = min(lons) #-62.9
lonE = max(lons) #-60
latS = min(lats) #-63
latN = max(lats) #-60

fig = plt.figure()
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
plt.savefig(_RESULTS_FOLDER +'figures/test.png')
