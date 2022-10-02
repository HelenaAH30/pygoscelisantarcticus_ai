#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:20:29 2020

@author: Helena Antich Homar
"""
#%% Libraries
# Disable warnings
from operator import index
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Computation libraries
import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
import geopy.distance as gp
from multiprocessing import Process
# Plotting libraries
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
#import shapely.geometry as sgeom
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
# AI libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#%%Configuration
os.chdir('/home/helena/Documents')
_DATA_FOLDER = './nombres_unificados/'
_RESULTS_FOLDER = './results_peng/'
_NEWDATA_FOLDER = './results_peng/new_data/'
_NEWMODELS_FOLDER = './results_peng/models/'
_LOGS_FOLDER = './logs/'


#%% Functions
def _load_rawdata(filename):
    """ Load data from file
    :param filename: (str) file name
    :return: pandas dataframe
    """
    # Loading data
    penguin = pd.read_csv(filename, delim_whitespace=True, lineterminator='\n', header=None)
    # Rename columns
    penguin = penguin.rename(columns= {0:"name", 1:"date",2:"time", 3:"undef1", 4:"undef2",
                             5:"undef3", 6:"active_dry", 7:"depth", 8:"temp",
                             9:"lat", 10:"lon", 11:"undef4",  12:"undef5",
                             13:"undef6", 14:"undef7", 15:"undef8", 16: "volt"})
    # Select useful columns
    #try
    penguin = penguin [["name", "date", "time", "depth", "temp", "lon", "lat"]]
    return penguin


def _parse_rawdates(penguin):
    """ Parse dates to datetime format
    :param penguin: pandas dataframe
    :return: pandas dataframe parsed
    """
    # Parse dates
    penguin ['datetime'] = penguin['date'] + ' ' + penguin['time']
    penguin ['datetime'] = pd.to_datetime(penguin['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
    return penguin


def _distance_btwn_lonlatpoints(lon_1, lat_1, lon_2, lat_2):
    """ Calculate distance between two points
    :param lon_1: (float) longitude of point 1
    :param lat_1: (float) latitude of point 1
    :param lon_2: (float) longitude of point 2
    :param lat_2: (float) latitude of point 2
    :return: (float) distance between points (km)
    """
    coords_1 = (lat_1,lon_1)
    coords_2 = (lat_2,lon_2)
    try:
        dist = gp.distance(coords_1, coords_2).km
        #dist = gp.vincenty(coords_1, coords_2).km
        return dist
    except: 
        return np.nan
    

def _replace_lat_outofrange(penguin):
    """ Replace latitudes out of range
    There was values == -244.03267
    :param penguin: pandas dataframe
    :return: pandas dataframe with latitudes replaced
    """
    # penguin.lat[(penguin.lat<-90) | (penguin.lat>90)] = np.nan
    penguin[(penguin.lat<-90) | (penguin.lat>90)]['lat'] = np.nan
    return penguin


def _calcule_speed (penguin):
    """ Calculate speed
    :param penguin: pandas dataframe with lat, lon, depth, temp, datetime
    :return: pandas dataframe with speed (km/h)
    """
    # Calcule of time delta between points
    penguin['delta_time'] = penguin.datetime.diff()
    penguin['delta_time'] = penguin.apply(lambda row: row.delta_time.total_seconds(), axis=1)
    # Calcule spatial difference between points
    penguin = _replace_lat_outofrange(penguin)
    penguin.dropna(axis=0, how='any', inplace=True)
    penguin[['lon_shift', 'lat_shift']] = penguin[['lon', 'lat']].shift(periods=1)
    penguin['delta_space'] = penguin.apply(lambda row: _distance_btwn_lonlatpoints(row.lon, row.lat, row.lon_shift, row.lat_shift), axis=1)
    #Calcule speed column
    penguin['speed'] = penguin['delta_space']/penguin['delta_time'] # km/s
    # Convert speed to km/h
    penguin['speed'] = penguin['speed'] * 3600
    return penguin


def _calcule_time_travel(penguin):
    """ Calculate time travel
    :param penguin: pandas dataframe with delta_time
    :return: pandas dataframe with time travel"""
    penguin['time_travel'] = penguin['delta_time'].cumsum()
    return penguin


def _calcule_temperature_gradient(penguin):
    """ Calculate temperature gradient
    :param penguin: pandas dataframe with temp and delta_space
    :return: pandas dataframe with temperature gradient (ºC/km)
    """
    # Calcule temperature difference between points
    penguin['temp_delta'] = penguin.temp.diff()
    # Calcule temperature gradient column
    penguin['temp_gradient'] = penguin.apply(lambda row: row.temp_delta/row.delta_space, axis=1)
    # Convert temperature gradient to ºC/m
    penguin['temp_gradient'] = penguin['temp_gradient'] * 1000
    return penguin


def _filter_column_outliers(penguin, column):
    """ Detect temperature gradient outliers
    :param penguin: pandas dataframe with temp_gradient
    :param column: (str) column name
    :return: pandas dataframe with outliers column"""
    m = penguin[column].mean()
    s = penguin[column].std()
    sigma3 = 3*s
    # Detect outliers
    penguin['outlier_temp'] = (penguin[column] >= m + sigma3) | (penguin[column] <= m - sigma3) #element-wise | and &
    penguin = penguin.loc[not(penguin['outlier_temp'])]
    penguin.drop('outlier_temp', axis=1, inplace=True)
    penguin.reset_index(drop=True, inplace=True)
    return penguin


def _calcule_compass_direction(point_i, point_f):
    """ Calculate compass direction
    :param point_i: (tuple) longitude, latitude of point 1
    :param point_f: (tuple) longitude, latitude of point 2
    :return: (float) compass direction
    """
    lat1 = math.radians(point_i[0])
    lat2 = math.radians(point_f[0])

    delta_lon = math.radians(point_f[1] - point_i[1])

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(delta_lon))

    direction = math.atan2(x, y) #-180 to 180 (radians)
    direction = math.degrees(direction) #-180 to 180 (degrees)
    return direction


def _calcule_direction(penguin):
    """ Calculate direction
    :param penguin: pandas dataframe with lon, lat, depth, temp, datetime
    :return: pandas dataframe with direction
    """
    # Calcule direction column
    penguin['direction'] = penguin.apply(lambda row: _calcule_compass_direction((row.lon, row.lat), (row.lon_shift, row.lat_shift)), axis=1)
    return penguin


def _normalize_direction(penguin):
    """ Normalize direction
    :param penguin: pandas dataframe with direction
    :return: pandas dataframe with normalized direction
    """
    # Normalize direction column
    penguin['direction'] = penguin['direction'] * (1/180) # -1 to 1
    return penguin


def _extract_trip_number(filename):
    """ Extract trip number from filename
    :param filename: (str) file name
    :return: (int) trip number
    """
    #trip_number = int(filename.split('/')[-1].split('_')[2].split('.')[0])
    #return trip_number
    trip_number = int(filename.split('/')[-1].split('_')[0].split('viaje')[1])
    return trip_number


def _extract_peng_number(filename):
    """ Extract penguin number from filename
    :param filename: (str) file name
    :return: (int) penguin number
    """
    #peng_number = int(filename.split('/')[-1].split('_')[0])
    #return peng_number
    peng_number = int(filename.split('/')[-1].split('_')[1].split('.')[0].split('newpeng')[1])
    return peng_number


def _save_boxplot_pengspeed(penguin_number, trip_number, penguin_data, string = None):
    """ Save boxplot of penguin data
    :param penguin_number: (int) penguin number
    :param trip_number: (int) trip number
    :param penguin_data: (pandas dataframe) penguin data
    :param string: (str) string to add to filename
    :return: (str) path to saved file
    """
    # plot
    boxplot = sns.boxplot(x=penguin_data['speed'])
    # boxplot.set_xlabel('Speed (km/h)')
    boxplot.set_xlabel('Velocidad (km/h)')
    # boxplot.set(title = f'Penguin {penguin_number} - Trip {trip_number}')
    boxplot.set(title = f'Pingüino {penguin_number} - Viaje {trip_number}')
    # create figure
    fig = boxplot.get_figure()
    if string != None:
       filename = _RESULTS_FOLDER +'figures/penguin' + str(penguin_number) + '_boxplot_'+ string + '.png' 
    else:
        filename = _RESULTS_FOLDER +'figures/penguin' + str(penguin_number) + '_boxplot.png'
    # save figure
    fig.savefig(filename)
    plt.close(fig) # close the figure


def _compose_statscsv(files_list):
    """ Compose stats csv file
    :param files_list: (list) list of files
    :return: (str) path to saved file
    """
    files_df = pd.DataFrame()
    files_df['files'] = files_list
    files_df['peng_number'] = files_df.apply(lambda row: _extract_peng_number(row.files), axis=1)
    files_df['trip'] = files_df.apply(lambda row: _extract_trip_number(row.files), axis=1)
    files_df = pd.DataFrame(files_df.groupby(['peng_number']).count())
    files_df['trip'].to_csv(_RESULTS_FOLDER + "files_statisticaldata.csv")
    return files_df


def _save_barplot_penguin(df, string = None):
    """
    Save barplot of files per penguin
    :param df: (pandas dataframe) files dataframe
    :param string: (str) string to add to filename
    """
    date = pd.to_datetime('today').strftime('%Y%m%d')
    # plot
    #sns.set_palette(sns.color_palette("viridis", as_cmap=True))
    barplot = sns.barplot(data = df, y = 'trip', x = df.index, palette="viridis")
    # barplot.set_title('Trajectories per penguin',fontsize=12)
    # barplot.set_xlabel('Penguin number')
    # barplot.set_ylabel('Number of trips')
    barplot.set_title('Trajectorias por pingüino',fontsize=12)
    barplot.set_xlabel('Identificador de pingüino')
    barplot.set_ylabel('Número de viajes')
    # create figure
    fig = barplot.get_figure()
    if string != None:
       filename = _RESULTS_FOLDER +'figures/satistics_barplot_' + date +'_'+ string + '.png' 
    else:
        filename = _RESULTS_FOLDER +'figures/satistics_barplot_' + date + '.png'
    # save figure
    fig.savefig(filename)
    plt.close(fig) # close the figure


def _write_txt_statistics(files_df, files_list):
    """ Write txt file with statistics
    :param files_df: (pandas dataframe) files dataframe"""
    file_stats_route = _RESULTS_FOLDER + "files_statisticaldata.txt"
    file_stats = open(file_stats_route, "w")
    file_stats.write("**********************" + os.linesep)
    file_stats.write(f"Fecha del análisis: {str(pd.Timestamp.today())}" + os.linesep)
    file_stats.write(f"Número de archivos analizados: {len(files_df)}" + os.linesep)
    file_stats.write(f"Media de viajes por pinguino: {files_df.trip.mean()}" + os.linesep)
    file_stats.write("**********************")
    file_stats.close()


def _write_log(file, step, error, file_log, today):
    """ Write log file"""
    file_log.write("**********************" + os.linesep)
    file_log.write(f"Fecha del análisis: {today}" + os.linesep)
    file_log.write(f"Fichero: {file}" + os.linesep)
    file_log.write(f"Paso: {step}" + os.linesep)
    file_log.write(f"Error: {error}" + os.linesep)
    file_log.write("**********************")
    file_log.close()
   

def _detect_speed_outliers(penguin):
    """ Detect outliers in speed
    :param penguin: (pandas dataframe) penguin data with speed column
    :return: (pandas dataframe) penguin data with outliers removed
    """
    mean = penguin['speed'].mean()
    sigma = penguin['speed'].std()
    sigma3 = 3*sigma
    penguin['outlier'] = (penguin['speed'] >= mean + sigma3) | (penguin['speed'] <= mean - sigma3) #element-wise | and &
    return penguin


# Track
def _plot_track(penguin, dataset ='test'):
    """ Plot track
    :param penguin: (pandas dataframe) penguin data
    :param dataset: (str) dataset name
    """
    # Penguin params
    lons = penguin ['lon']
    lats = penguin ['lat']
    num = penguin.loc[1,'peng_number']
    trip = penguin.loc[1,'trip']
    depth = penguin ['depth']
    # Plot configuration
    lonW = -61.4 #min(lons) 
    lonE = -60.70 #max(lons)
    latS = -63.2 #min(lats)
    latN = -62.9 #max(lats)
    ## Axes
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree()) #ccrs.SouthPolarStereo()
    ax.set_extent([lonW, lonE, latS, latN])
    ## Track plot
    points = plt.scatter(x=lons, y=lats, c= depth, cmap='viridis', s=0.1, marker='o',zorder = 2)
    lines = plt.plot(lons, lats, color='gray', linewidth=0.1, zorder=1)
    ## Coastlines
    ax.add_feature(cfeature.GSHHSFeature(levels = [1,2,3,4],scale='full',facecolor='silver'), zorder=100)
    ## Layout
    plt.colorbar(label="Depth (m)", orientation="vertical")
    ax.set_xticks(np.round(np.linspace(lonW,lonE,5),2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.round(np.linspace(latS,latN,10),2), crs=ccrs.PlateCarree())
    # ax.set_title(f'Penguin number: {num} - trip: {trip}',fontsize=10)
    # ax.set_ylabel('Latitude',fontsize=8)
    # ax.set_xlabel('Longitude',fontsize=8)
    ax.set_title(f'Identificador de pingüino: {num} - viaje: {trip}',fontsize=10)
    ax.set_ylabel('Latitud',fontsize=10)
    ax.set_xlabel('Longitud',fontsize=10)
    # Correct bbox
    box = ax.get_position()
    ax.set_position([box.x0 + box.width*0.1, box.y0,
                    box.width, box.height])
    ## Save figure
    plt.savefig(_RESULTS_FOLDER +'figures/'+dataset+'.png', dpi=500) # resolution = 300 dpi
    # TODO: delete plt.savefig('test.png')
    ## Close figure
    plt.close(fig)


def _plot_multitrack(dataset, figname ='test'):
    """ Plot track
    :param dataset: (pandas dataframe) penguin data
    :param dataset: (str) dataset name
    """
    # Color palette
    viridis = plt.get_cmap('viridis')
    # Plot configuration
    ## Lat-Lon limits
    lonW = -61.4 #min(lons) 
    lonE = -60.70 #max(lons)
    latS = -63.2 #min(lats)
    latN = -62.9 #max(lats)
    ## Axes
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree()) #ccrs.SouthPolarStereo()
    ax.set_extent([lonW, lonE, latS, latN])

    # penguin-trip unique values
    dataset['pengs_trips'] = ['-'.join(i) for i in zip(('peng'+dataset["peng_number"].map(str)),('trip'+dataset["trip"].map(str)))]
    pengs_trips = dataset['pengs_trips'].unique()
    colors = viridis(np.linspace(0, 1, len(pengs_trips)))
    # plot each penguin-trip
    for i in range(len(pengs_trips)):
        peng_trip = pengs_trips[i]
        color = mcolors.rgb2hex(colors[i])
        peng_trip_df = dataset[dataset['pengs_trips'] == peng_trip]
        plt.plot(peng_trip_df['lon'], peng_trip_df['lat'], color=color, linewidth=0.5, label = peng_trip, zorder=0)
    ## Coastlines
    ax.add_feature(cfeature.GSHHSFeature(levels = [1,2,3,4],scale='full',facecolor='snow'), zorder=1)
    ## Layout
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_xticks(np.round(np.linspace(lonW,lonE,5),2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.round(np.linspace(latS,latN,5),2), crs=ccrs.PlateCarree())
    # ax.set_title('All trajectories',fontsize=9)
    # ax.set_ylabel('Latitude',fontsize=8)
    # ax.set_xlabel('Longitude',fontsize=8)
    ax.set_title('Todas las trayectorias',fontsize=7)
    ax.set_ylabel('Latitud',fontsize=6)
    ax.set_xlabel('Longitud',fontsize=6)
    # Legend
    # plt.legend(loc='upper center', bbox_to_anchor=(2,0.5),
    #       ncol=3, fancybox=True, shadow=True)
    # Shrink current axis's height by 20% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.55,
                    box.width, box.height * 0.8])
    # Put a legend below current axis
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.25),
            fancybox=True, shadow=True, ncol=5, prop={'size': 6})
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    ## Save figure
    plt.savefig(_RESULTS_FOLDER +'figures/'+figname+'.png', dpi=500, transparent=True) # resolution = 300 dpi
    ## Close figure
    plt.close(fig)


def trajectory_analysis(file):
    """ Trajectory analysis
    :param file: (str) file name
    """
    # Open log
    today = pd.Timestamp.today().strftime('%Y%m%d')
    file_log_route = _LOGS_FOLDER + f"log_{file.split('/')[-1].split('.')[0]}_{today}.txt"
    file_log = open(file_log_route, "w")
    try:
        # Parse data
        step = 'parse_data'
        penguin = _load_rawdata(file)
        penguin = _parse_rawdates(penguin)
        # Penguin data
        step = 'penguin_original_data'
        trip_number = _extract_trip_number(file)
        peng_number = _extract_peng_number(file)
        # Add penguin data to dataframe
        penguin['trip'] = trip_number
        penguin['peng_number'] = peng_number
        title = f"penguin{peng_number:02}_trip{trip_number}_{step}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv", index=False)


        """
        STEP 0: filtered depth by value
        The depth>5m is filtered.
        When the penguin reached this depth we can say that it's fishing.
        """
        step = 'step_0'
        print(step)
        penguin = penguin.loc[penguin.depth < 5,:].reset_index(drop=True)
        title = f"penguin{peng_number:02}_trip{trip_number}_step{0}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv", index = False)


        """
        STEP 1: Calcule speed and explore data
        Just to check if there are outliers in speed
        """
        penguin = _calcule_speed (penguin)
        step = 'step_1'
        print(step)
        _save_boxplot_pengspeed(peng_number, trip_number, penguin, string = 'step1_nofiltered')
        title = f"penguin{peng_number:02}_trip{trip_number}_step{1}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv", index = False)
        _plot_track(penguin, dataset = "track_"+title)


        """
        STEP 2: filtered speed by max value
        Mark speed by max value
        """
        step = 'step_2'
        print(step)
        # Filter speed = 0 (if penguin is not moving, no behaviour can be detected)
        penguin = penguin.loc[penguin.speed != 0,:].reset_index(drop=True) 
        # Filter speed > 60km/h
        penguin = penguin.loc[penguin.speed < 60,:].reset_index(drop=True)
        title = f"penguin{peng_number:02}_trip{trip_number}_step{2}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv", index = False)
        _plot_track(penguin, dataset = "track_"+title)
        _save_boxplot_pengspeed(peng_number, trip_number, penguin, string = 'step2_filtered')


        """
        STEP 3: Outlier removal by standard deviation
        Detect outliers in speed and temperature and remove them
        """
        step = 'step_3'
        print(step)
        # Outlier detection
        penguin = _detect_speed_outliers(penguin)
        penguin_out = penguin.loc[penguin.outlier !=True,:]
        _save_boxplot_pengspeed(peng_number, trip_number, penguin_out, string = 'step3_withoutoutliers')
        title = f"penguin{peng_number:02}_trip{trip_number}_step{3}"
        penguin_out.to_csv(_NEWDATA_FOLDER + title +".csv", index = False)
        # Detect and delete outliers in temperature gradient
        penguin_out = _filter_column_outliers(penguin_out, column = 'temp')
        _plot_track(penguin_out, dataset = "track_"+title)


        """
        STEP 4:
        Downgrade temporal resolution to 5min resolution
        """
        """ DEPRECATED
        step = 'step_4'
        print(step)
        # Filtro de datos minutales a menor resolución temporal: promedio temporal con la media
        # series.resample('3T').sum() -> series.resample('1T', on = 'datetime').mean()
        # 1T = 1 min, 5T = 5 min
        penguin_out = penguin_out[['name', 'datetime', 'depth', 'temp', 'lon', 'lat', 'speed', 'trip', 'peng_number']]
        penguin_out = penguin_out.resample('5T', axis=0, on='datetime').mean()
        title = f"penguin{peng_number:02}_trip{trip_number}_step{4}"
        penguin_out.to_csv(_NEWDATA_FOLDER + title + ".csv", index = False)
        _plot_track(penguin_out,dataset = "track_"+title)
        """


        """
        STEP 5:
        Calcule temperature gradient, time traveling and normalized direction
        """
        step = 'step_5'
        print(step)
        # Drop rows with NaN values
        penguin_out.dropna(subset = ['temp','delta_space','delta_time','lon','lat'], inplace = True)
        # Plot histogram of variables
        save_variable_histogram(penguin_out, 'temp', title = 'Histograma de temperaturas', filename=f"penguin{peng_number:02}_trip{trip_number}_temp_histogram")
        # Calcule temperature gradient
        step='calcule_temp_gradient'
        penguin_out = _calcule_temperature_gradient(penguin_out)
        # Calcule time traveling
        step='calcule_time_traveling'
        penguin_out = _calcule_time_travel(penguin_out)
        # Calcule direction in degrees
        step='calcule_direction'
        penguin_out = _calcule_direction(penguin_out)
        # Normalize direction to -1:1
        step='normalize_direction'
        penguin_out = _normalize_direction(penguin_out)
        # Save data
        title = f"penguin{peng_number:02}_trip{trip_number}_step{5}"
        penguin_out.to_csv(_NEWDATA_FOLDER + title + ".csv", index = False)
        

        """
        STEP 6:
        Final dataset per penguin
        """
        step = 'step_6'
        print(step)
        # Selected columns
        penguin_fin = penguin_out[['lon', 'lat', 'temp_gradient', 'time_travel', 'direction']]
        # Save final dataset
        title = f"penguin{peng_number:02}_trip{trip_number}_final"
        penguin_fin.to_csv(_NEWDATA_FOLDER + title + ".csv", index = False)
        # Cierre y borrado del archivo logs si no hay error
        file_log.close()
        os.remove(file_log_route)
    except Exception as error:
        _write_log(file, step, error, file_log, today)


def _save_dataset(dataset, title='dataset'):
    """
    Save dataset in a csv file
    :param df: dataframe
    :return: None
    """
    # Reset index
    dataset.reset_index(drop=True, inplace=True)
    # Save dataset
    dataset.to_csv(_NEWDATA_FOLDER + f"{title}.csv", index=False) 
    return dataset


def compose_dataset():
    """ Compose dataset """
    files_list = glob.glob(_NEWDATA_FOLDER+'*_final.csv')
    dataset = pd.DataFrame()
    for file in files_list:
        # Read file
        df = pd.read_csv(file)
        # Add peng_number and trip
        df['peng_number'] = int(file.split('/')[-1].split('_')[0].split('penguin')[-1])
        df['trip'] = int(file.split('/')[-1].split('_')[1].split('trip')[-1])
        # Add file to dataset
        dataset = pd.concat([dataset, df],axis=0, join='outer', ignore_index=True)
    # compose track indicator
    dataset['pengs_trips'] = ['-'.join(i) for i in zip(('peng'+dataset["peng_number"].map(str)),('trip'+dataset["trip"].map(str)))]
    dataset.dropna(axis=1, how='all', inplace=True)
    # Save dataset
    dataset = _save_dataset(dataset, title='dataset')
    return dataset  


def dataset_train_test():
    """ Split dataset into train and test """
    dataset = pd.read_csv(_NEWDATA_FOLDER + "dataset.csv")
    # Split dataset into train and test
    train, test = train_test_split(dataset, test_size=0.25)
    # Reset index
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    # Save dataset
    train.to_csv(_NEWDATA_FOLDER + "train.csv", index=False)
    test.to_csv(_NEWDATA_FOLDER + "test.csv", index=False)


def normalize_dataset(dataset_name, scaler = None):
    """ Normalize dataset """
    dataset = pd.read_csv(_NEWDATA_FOLDER+dataset_name+'.csv')
    if dataset_name == 'train':
        scaler = StandardScaler()
        scaler.fit(dataset[['lon','lat','temp_gradient','time_travel']])
    if dataset_name == 'test' and scaler==None:
        print("Error: scaler is None")
        return None
    # Normalize dataset
    dataset[['lon','lat','temp_gradient','time_travel']] = scaler.transform(dataset[['lon','lat','temp_gradient','time_travel']])
    dataset_scaled = dataset[['lon','lat','temp_gradient','time_travel','direction','pengs_trips']]
    # Save scaled dataset
    dataset_scaled.to_csv(_NEWDATA_FOLDER+dataset_name+'_norm.csv', index=False)
    return scaler


def _write_scores(grid_param, file = 'grid_params_scores.txt'):
    """
    Write scores to file
    :param grid_param: (dict) grid parameters
    :param file: (string) file to write
    :return: txt file with scores
    """
    with open(_NEWMODELS_FOLDER+file, 'w') as f:
        f.write('Best score')
        f.write(str(grid_param.best_score_))
        f.write('\n')
        f.write('Best parameters')
        f.write(str(grid_param.best_params_))
        f.write('\n')
        f.write('Best estimator')
        f.write(str(grid_param.best_estimator_))
        f.write('\n')
        f.write('Best index')
        f.write(str(grid_param.best_index_))
        f.write('\n')
        f.write('Scores')
        f.write(str(grid_param.cv_results_))
        f.close()


def _pooled_var(stds, pool =10):
    """
    Compute pooled variance
    :param stds: list of standard deviations
    :param pool: number of samples to pool
    :return: pooled variance
    """
    return np.sqrt(sum((pool-1)*(stds**2))/ len(stds)*(pool-1))


def _plot_errors(grid_params, number_of_best_combinations = 10,
                results = ['mean_test_score', 'std_test_score'], all_combinations = False):
    """
    Plot errors
    :param grid_params: grid search parameters
    :param number_of_best_combinations: number of best combinations to plot
    :param results: list with results
    :return: None
    """
    # Plot style
    sns.set_theme(style="whitegrid")
    # Create dataframe with results
    df = pd.DataFrame(grid_params.cv_results_)
    cols = df.filter(regex=("param_.*")).columns.to_list()
    max_combinations = number_of_best_combinations if not all_combinations else df["rank_test_score"].max()
    df_filtered = df[df["rank_test_score"] <= max_combinations]
    # Plot errors
    g = sns.barplot(data=df_filtered, x="rank_test_score", y=results[0], palette="viridis")
    g.set(xlabel = "", ylabel = "Rank test score")
    ## Save figure
    plt.savefig(_NEWDATA_FOLDER + f'{results[0]}.png')


def tunning_hyperparameters(param_dic, model, train_dataset, x_train_vars, y_train_var):
    """
    Tunning hyperparameters
    :param param_dic: (dict) grid search parameters
    :param model: (model) model to tune
    :param train_dataset: (dataframe) train dataset
    :param x_train_vars: (list) list of variables to use as x
    :param y_train_var: (string) variable to use as y
    :return: grid search object
    """
    grid_param = GridSearchCV(estimator=model, param_grid=param_dic) # grid search with cross validation
    # TODO: falta una semilla en algún sitio???? 
    # Tunning hyperparameters
    grid_param.fit(train_dataset[x_train_vars].values, train_dataset[y_train_var].values)
    # Save results of tunning hyperparameters
    ## Save results txt file 
    _write_scores(grid_param, file = 'grid_params_scores.txt')
    ## Save model
    joblib.dump(grid_param.best_estimator_, _NEWMODELS_FOLDER+'model.pkl')
    ## Plot results
    _plot_errors(grid_param)
    return grid_param


def write_ai_params(model):
    """
    Write AI parameters to file
    :param model: (object) AI model
    :return: None
    """
    parameters = model.get_params() 
    with open(_NEWMODELS_FOLDER+'ann_params.txt', 'w') as f:
        f.write(str(parameters))
        f.write('\n')
        f.close()


def reverse_transform_direction(direction):
    """ Reverse transform direction
    :param direction: direction in range -1:1
    :return: direction in degrees (-180:180)
    """
    direction_deg = (direction * 180)
    return direction_deg


def save_errorboxplot(error_values, error_title, error_unit = 'unit'):
    """ Save boxplot of penguin data
    :param error_values: (array) error values
    :param error_title: (string) error name
    :param error_unit: (string) error unit
    :return: (str) path to saved file
    """
    # plot
    boxplot = sns.boxplot(x=error_values)
    boxplot.set_xlabel(f'{error_title} ({error_unit})')
    boxplot.set(title = f'{error_title}')
    # create figure
    fig = boxplot.get_figure()
    filename = '_'.join(error_title.split(' '))
    fig.savefig(_RESULTS_FOLDER + 'figures/' + f'{filename}.png')
    plt.close(fig) # close the figure


def save_variable_histogram(df, variable, title = None, xmax = None, ymax = None, ymin = None, filename = None):
    """ Save histogram of variable
    :param df: (dataframe) dataframe with variable
    :param variable: (string) variable name
    :return: None
    """
    # plot
    histogram = sns.histplot(data=df, x=variable, kde=True)
    histogram.set_xlabel(f'{variable}')
    if title!=None:
        histogram.set(title = f'{title}')
    else:
        histogram.set(title = f'Histogram {variable}')
    # set limits
    if xmax != None:
        histogram.set_xlim(0, xmax)
    if (ymax != None) and (ymin != None):
        histogram.set_ylim(ymin, ymax)
    if filename == None:
        filename = '_'.join(variable.split(' '))
    # create figure
    fig = histogram.get_figure()
    path_filename = _RESULTS_FOLDER + 'figures/' + f'histogram_{filename}.png'
    fig.savefig(path_filename)
    plt.close(fig) # close the figure


#%% MAIN
if __name__ == '__main__':
    """ Main function """
    # files list
    files_list = glob.glob(_DATA_FOLDER+'viaje*.csv')
    # list to save the PID of the processes created
    procs = [] 
    # statistical analysis
    files_df = _compose_statscsv(files_list)
    _save_barplot_penguin(files_df)
    _write_txt_statistics(files_df, files_list)
    # TODO: delete: next line
    # files_list = glob.glob(_DATA_FOLDER+'viaje2_newpeng03_nido75.csv')
    # Paralelized process to analyze each file, with a penguin trip
    for file in files_list:
        """ STEPS 1 to 6 (both included) """
        p = Process(target=trajectory_analysis, args=(file,)) #n-1 processes
        procs.append(p)
        p.start()
    # Join all processes
    for p in procs:
        p.join()

    #%%
    """
    STEP 7:
    Composed dataset and train and test datasets
    """
    step = 'step_7'
    print(step)
    # Compose dataset
    dataset = compose_dataset()
    # Last inspection before training    
    ## Plot all tracks on a map
    _plot_multitrack(dataset, 'all_tracks')
    ### After visual inspection, we can see that there are some outliers
    ### We will remove them
    dataset = dataset[dataset['lon'] > -61.4]
    # Save dataset
    dataset = _save_dataset(dataset, title='dataset')
    # Split dataset into train and test
    dataset_train_test()
    """
    STEP 8:
    Train and test datasets normalization
    """
    step = 'step_8'
    print(step)
    scaler = normalize_dataset('train')
    normalize_dataset('test', scaler)
    #%%
    """
    STEP 9:   
    Tunning hyperparameters to select the best model
    """
    step = 'step_9' #It seems to be already parallelized
    print(step)
    # Set seed
    seed = 42
    np.random.seed(seed)
    # Define ANN
    ann = MLPRegressor(early_stopping=True,max_iter=1000) # max_iter=1000
    # Get train and test datasets
    train = pd.read_csv(_NEWDATA_FOLDER+'train_norm.csv')
    test = pd.read_csv(_NEWDATA_FOLDER+'test_norm.csv')
    # NAN values treatment: delete
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    # Define parameters to tunning
    param_dic = {
        "hidden_layer_sizes": [(5,),(10,),(50,)], # 1 hidden layer
        "activation": ["identity", "tanh","relu"], # activation function
        "solver": ["lbfgs", "sgd", "adam"], # solver
        "alpha": [0.00005,0.0005,0.005], # learning rate (alpha)
        "learning_rate_init":[0.01, 0.001, 0.0001] # initial learning rate
    }
    grid_param = tunning_hyperparameters(param_dic, ann, train, ['lon','lat','temp_gradient','time_travel'], 'direction')
    
    #%%
    """
    STEP 10:
    Train and test datasets with the best model
    """
    step = 'step_10'
    print(step)
    # Define ANN model with the best hyperparameters found
    ann = grid_param.best_estimator_
    write_ai_params(ann) # save params used
    ## Train ANN
    ann.fit(train[['lon','lat','temp_gradient','time_travel']].values, train['direction'].values)
    ## Test ANN 
    y_pred = ann.predict(test[['lon','lat','temp_gradient','time_travel']].values)
    ## Save results prediction normalized
    test['direction_pred'] = y_pred
    #%%
    """
    STEP 11:
    Computing specific errors made on test dataset
    """
    step = 'step_11'
    print(step)
    ## Get errors
    test['absolute_error'] = abs(test['direction'] - test['direction_pred'])
    scores = pd.DataFrame(test.groupby('pengs_trips').apply(lambda group: mean_absolute_error(group.direction.values, group.direction_pred.values)))
    scores = scores.merge(pd.DataFrame(test.groupby('pengs_trips').apply(lambda group: mean_squared_error(group.direction.values, group.direction_pred.values))))
    scores = scores.merge(pd.DataFrame(test.groupby('pengs_trips').apply(lambda group: r2_score(group.direction.values, group.direction_pred.values))))

    ### All in degrees
    test['direction_deg'] = reverse_transform_direction(test['direction'])
    test['direction_pred_deg'] = reverse_transform_direction(test['direction_pred'])
    ### Compute mse in degrees
    scores = scores.merge(pd.DataFrame(test.groupby('pengs_trips').apply(lambda group: mean_absolute_error(group.direction_deg.values,group.direction_pred_deg.values))))
    scores = scores.merge(pd.DataFrame(test.groupby('pengs_trips').apply(lambda group: mean_squared_error(group.direction_deg.values,group.direction_pred_deg.values))))
    scores = scores.merge(pd.DataFrame(test.groupby('pengs_trips').apply(lambda group: r2_score(group.direction_deg.values,group.direction_pred_deg.values))))
    scores.reset_index(inplace=True)
    # Merge scores with test dataset
    test = test.merge(scores, on='pengs_trips')
    ### Save predictions, errors and metrics
    test.to_csv(_NEWDATA_FOLDER + "test_prediction_norm.csv", index=False)
    ## Plot errors
    save_errorboxplot(test['absolute_error'].values, 'Error absoluto')
    save_errorboxplot(test['mae'].values, 'MAE')
    save_errorboxplot(test['mse'].values, 'MSE')
    save_errorboxplot(test['r2'].values, 'R2')
    save_errorboxplot(test['mae_deg'].values, 'MAE en grados', error_unit='degrees')
    save_errorboxplot(test['mse_deg'].values, 'MSE en grados', error_unit='degrees')
    save_errorboxplot(test['r2_deg'].values, 'R2 en grados', error_unit='degrees')


    
    
#%%