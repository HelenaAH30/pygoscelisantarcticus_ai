#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:20:29 2020

@author: Helena
"""
#%% Libraries
import os
import math
import geopy.distance as gp
import numpy as np
import pandas as pd

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


def calcule_velocity (penguin):
    # Calcule of time delta between points
    penguin['delta_time'] = penguin.datetime.diff()
    # Calcule spatial difference between points
    penguin = _replace_lat_outofrange(penguin)
    penguin.dropna(axis=0, how='any', inplace=True)
    penguin[['lon_shift', 'lat_shift']] = penguin[['lon', 'lat']].shift(periods=1)
    penguin['delta_space'] = penguin.apply(lambda row: _distance_btwn_lonlatpoints(row.lon, row.lat, row.lon_shift, row.lat_shift), axis=1)

    ''' Deprecated
    # Matrix of spatial diferences
    matrix_cdist = cdist(penguin[['lat','lon']].to_numpy(),penguin[['lat','lon']].to_numpy())
    # Select diagonal of spatial diferences matrix
    diagonal_plus1 = np.diagonal(matrix_cdist, offset = 1) # array[i,i+1]
    penguin['delta_space'] = diagonal_plus1
    '''
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
    filename = _RESULTS_FOLDER +'figures/' + penguin_number + '.png'
    # save figure
    fig.savefig(filename)

def _distance_btwn_lonlatpoints(lon_1, lat_1, lon_2, lat_2):
    coords_1 = (lon_1, lat_1)
    coords_2 = (lon_2, lat_2)
    dist = gp.distance(coords_1, coords_2).km
    #dist = gp.vincenty(coords_1, coords_2).km

    return dist


def _replace_lat_outofrange(penguin):
    """
    There was values == -244.03267
    """
    penguin.lat[(penguin.lat<-90) | (penguin.lat>90)] = np.nan

    return penguin


def distance2apoint(lon, lat, lonp, latp):
    dist = np.zeros(lon.shape)
    for k in np.arange(lon.shape[0]):
        coords_1 = (lon[k], lat[k])
        coords_2 = (lonp, latp)
        #dist[k] = geopy.distance.vincenty(coords_1, coords_2).km
        dist[k] = gp.distance(coords_1, coords_2).km

    return dist

def _length_lon_lat_degs(lat_mean_deg, latitude=False, longitude=False):

    '''
    Function to infer the length of a degree of longitude
    and the length of a degree of latitude
    at a specific latitude position.
    Assuming that the Earth is an ellipsoid.

    input: latitude in DEGREES!!!
    output: length_deg_lon, length_deg_lat in meters

    from:
    https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
    https://en.wikipedia.org/wiki/Latitude#Length_of_a_degree_of_latitude
    '''

    ''' Earth parameters '''
    lat_mean_rad = lat_mean_deg*(np.pi/180) #in radians

    a = 6378137.0 # m (equatorial radius)
    b = 6356752.3142 # m (polar radius)

    ecc_2 = (a**2 - b**2) / a**2 # eccentricity squared
    
    if longitude:
        ''' The length of a degree of longitude is... '''

        divident_lon = (a*np.pi/180) * np.cos(lat_mean_rad)
        divisor_lon = np.sqrt(1 - (ecc_2*np.sin(lat_mean_rad)*np.sin(lat_mean_rad)))

        length_deg_lon = divident_lon / divisor_lon
    else:
        length_deg_lon = None

    if latitude:
        ''' The length of a degree of latitude is... '''

        divident_lat = (a*np.pi/180) * (1 - ecc_2)
        divisor_lat = (1 - (ecc_2 * np.sin(lat_mean_rad) * np.sin(lat_mean_rad)))**(3/2)

        length_deg_lat = divident_lat / divisor_lat
    else:
        length_deg_lat = None

    return length_deg_lon, length_deg_lat

def _convert_delta_latlon_to_kms(dlon_deg, dlat_deg, lat_mean_deg, latitude=False, longitude=False):
    length_deg_lonp, length_deg_latp = _length_lon_lat_degs(lat_mean_deg, latitude, longitude)
    if longitude:
        dx_km = dlon_deg * (length_deg_lonp/1000)
    else:
        dx_km = None
    if latitude:
        dy_km = dlat_deg * (length_deg_latp/1000)
    else:
        dy_km = None

    return dx_km, dy_km

def _distance_km(dx_km,dy_km):
    delta_km = math.sqrt(dx_km**2 + dy_km**2)
    return delta_km

def compute_delta_kms(delta_lon, delta_lat):
    lat_mean_deg = delta_lat/2
    dx_km, dy_km = _convert_delta_latlon_to_kms(delta_lon, delta_lat, lat_mean_deg, latitude, longitude)
    delta_km = _distance_km(dx_km,dy_km)
    return delta_km
    
# def delete_velocity_outliers(penguin):
#     mean = penguin['velocity'].mean()
#     sigma = penguin['velocity'].std()
#     sigma3 = 3*sigma
#     penguin.loc[(penguin['velocity'] <= mean + sigma3) | (penguin['velocity'] >= mean - sigma3),:]

#%%
file = 'viaje2_newpeng03.csv'
file = 'viaje2_newpeng03_nido75.csv'
# file = 'viaje3_newpeng23_nido91.csv'
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

# Outlier detection and removal
save_boxplot(peng_number, penguin)
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