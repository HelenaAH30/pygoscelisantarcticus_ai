#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:20:29 2020

@author: Helena
"""
#%% Libraries
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
import cartopy.feature as cfeature
#import shapely.geometry as sgeom
# AI libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%Configuration
# os.chdir('/Volumes/MUSI-HAH/TFM/penguin_data/nombres_unificados/')
os.chdir('/home/helena/Documents')
_DATA_FOLDER = './nombres_unificados/'
_RESULTS_FOLDER = './results_peng/'
_NEWDATA_FOLDER = './results_peng/new_data/'
_NEWMODELS_FOLDER = './results_peng/models/'
_LOGS_FOLDER = './logs/'


#%% Functions

def load_data(filename):
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


def parse_dates(penguin):
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
    :return: (float) distance between points"""
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
    penguin.lat[(penguin.lat<-90) | (penguin.lat>90)] = np.nan
    return penguin


def calcule_speed (penguin):
    """ Calculate speed
    :param penguin: pandas dataframe with lat, lon, depth, temp, datetime
    :return: pandas dataframe with speed"""
    # Calcule of time delta between points
    penguin['delta_time'] = penguin.datetime.diff()
    penguin['delta_time'] = penguin.apply(lambda row: row.delta_time.total_seconds(), axis=1)
    # Calcule spatial difference between points
    penguin = _replace_lat_outofrange(penguin)
    penguin.dropna(axis=0, how='any', inplace=True)
    penguin[['lon_shift', 'lat_shift']] = penguin[['lon', 'lat']].shift(periods=1)
    penguin['delta_space'] = penguin.apply(lambda row: _distance_btwn_lonlatpoints(row.lon, row.lat, row.lon_shift, row.lat_shift), axis=1)
    #Calcule speed column
    penguin['speed'] = penguin['delta_space']/penguin['delta_time']
    return penguin


def calcule_time_travel(penguin):
    """ Calculate time travel
    :param penguin: pandas dataframe with delta_time
    :return: pandas dataframe with time travel"""
    penguin['time_travel'] = penguin['delta_time'].cumsum()
    return penguin


def calcule_temperature_gradient(penguin, units='km'):
    """ Calculate temperature gradient
    :param penguin: pandas dataframe with temp and delta_space
    :return: pandas dataframe with temperature gradient"""
    # Calcule temperature difference between points
    penguin['temp_delta'] = penguin.temp.diff()
    # Calcule temperature gradient column
    penguin['temp_gradient'] = penguin.apply(lambda row: row.temp_delta/row.delta_space, axis=1)
    if units == 'km':
        return penguin
    elif units == 'm':
        penguin['temp_gradient'] = penguin['temp_gradient']*1000
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

    direction = math.atan2(x, y) #-180 to 180
    direction = math.degrees(direction)
    direction = (direction + 360) % 360
    return direction


def calcule_direction(penguin):
    """ Calculate direction
    :param penguin: pandas dataframe with lon, lat, depth, temp, datetime
    :return: pandas dataframe with direction
    """
    # Calcule direction column
    penguin['direction'] = penguin.apply(lambda row: _calcule_compass_direction((row.lon, row.lat), (row.lon_shift, row.lat_shift)), axis=1)
    return penguin


def normalize_direction(penguin):
    """ Normalize direction
    :param penguin: pandas dataframe with direction
    :return: pandas dataframe with normalized direction
    """
    # Normalize direction column
    penguin['direction'] = penguin['direction'] * (1/360)
    return penguin


def extract_trip_number(filename):
    """ Extract trip number from filename
    :param filename: (str) file name
    :return: (int) trip number
    """
    #trip_number = int(filename.split('/')[-1].split('_')[2].split('.')[0])
    #return trip_number
    trip_number = int(filename.split('/')[-1].split('_')[0].split('viaje')[1])
    return trip_number


def extract_peng_number(filename):
    """ Extract penguin number from filename
    :param filename: (str) file name
    :return: (int) penguin number
    """
    #peng_number = int(filename.split('/')[-1].split('_')[0])
    #return peng_number
    peng_number = int(filename.split('/')[-1].split('_')[1].split('.')[0].split('newpeng')[1])
    return peng_number


def save_boxplot(penguin_number, trip_number, penguin_data, string = None):
    """ Save boxplot of penguin data
    :param penguin_number: (int) penguin number
    :param trip_number: (int) trip number
    :param penguin_data: (pandas dataframe) penguin data
    :param string: (str) string to add to filename
    :return: (str) path to saved file
    """
    # plot
    boxplot = sns.boxplot(x=penguin_data['speed'])
    boxplot.set_xlabel('Speed (m/s)')
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
    """ Compose stats csv file
    :param files_list: (list) list of files
    :return: (str) path to saved file
    """
    files_df = pd.DataFrame()
    files_df['files'] = files_list
    files_df['peng_number'] = files_df.apply(lambda row: extract_peng_number(row.files), axis=1)
    files_df['trip'] = files_df.apply(lambda row: extract_trip_number(row.files), axis=1)
    files_df = pd.DataFrame(files_df.groupby(['peng_number']).count())
    files_df['trip'].to_csv(_RESULTS_FOLDER + "files_statisticaldata.csv")
    return files_df


def save_barplot(df, string = None):
    """ Save barplot of files per penguin
    :param df: (pandas dataframe) files dataframe
    :param string: (str) string to add to filename
    """
    date = pd.to_datetime('today').strftime('%Y%m%d')
    # plot
    #sns.set_palette(sns.color_palette("viridis", as_cmap=True))
    barplot = sns.barplot(data = df, y = 'trip', x = df.index, palette="viridis")
    barplot.set_title('Trajectories per penguin',fontsize=12)
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


def write_log(file, step, error, file_log, today):
    """ Write log file"""
    file_log.write("**********************" + os.linesep)
    file_log.write(f"Fecha del análisis: {today}" + os.linesep)
    file_log.write(f"Fichero: {file}" + os.linesep)
    file_log.write(f"Paso: {step}" + os.linesep)
    file_log.write(f"Error: {error}" + os.linesep)
    file_log.write("**********************")
    file_log.close()
   

def detect_speed_outliers(penguin):
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
    #%% Plot configuration
    lonW = -61.167 #min(lons) 
    lonE = -60.50 #max(lons)
    latS = -63.3 #min(lats)
    latN = -62.75 #max(lats)
    ## Axes
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree()) #ccrs.SouthPolarStereo()
    ax.set_extent([lonW, lonE, latS, latN])
    ## Track plot
    points = plt.scatter(x=lons, y=lats, c= depth, cmap='viridis', s=0.1, marker='o')
    ## Coastlines
    ax.add_feature(cfeature.GSHHSFeature(levels = [1,2,3,4],scale='full',facecolor='silver'), zorder=100)
    ## Layout
    plt.colorbar(label="Depth (m)", orientation="vertical")
    ax.set_xticks(np.round(np.linspace(lonW,lonE,5),2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.round(np.linspace(latS,latN,10),2), crs=ccrs.PlateCarree())
    ax.set_title(f'Penguin number: {num} - trip: {trip}',fontsize=12)
    ax.set_ylabel('Latitude',fontsize=12)
    ax.set_xlabel('Longitude',fontsize=12)
    ## Save figure
    plt.savefig(_RESULTS_FOLDER +'figures/'+dataset+'.png', dpi=500) # resolution = 300 dpi
    # TODO: delete plt.savefig('test.png')
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
        penguin = load_data(file)
        penguin = parse_dates(penguin)
        # Penguin data
        step = 'penguin_data'
        trip_number = extract_trip_number(file)
        peng_number = extract_peng_number(file)
        # Add penguin data to dataframe
        penguin['trip'] = trip_number
        penguin['peng_number'] = peng_number
        title = f"penguin{peng_number:02}_trip{trip_number}_step{step}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv")


        """
        STEP 0: filtered depth by value
        The depth>5m is filtered.
        When the penguin reached this depth we can say that it's fishing.
        """
        step = 'step_0'
        print(step)
        penguin = penguin.loc[penguin.depth < 5,:].reset_index(drop=True)
        title = f"penguin{peng_number:02}_trip{trip_number}_step{0}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv")


        """
        STEP 1: Calcule speed and explore data
        Just to check if there are outliers in speed
        """
        penguin = calcule_speed (penguin)
        step = 'step_1'
        print(step)
        save_boxplot(peng_number, trip_number, penguin, string = 'step1_nofiltered')
        title = f"penguin{peng_number:02}_trip{trip_number}_step{1}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv")
        _plot_track(penguin, dataset = "track_"+title)


        """
        STEP 2: filtered speed by max value
        Mark speed by max value
        """
        step = 'step_2'
        print(step)
        # Filter speed=<0
        penguin = penguin.loc[penguin.speed > 0,:].reset_index(drop=True) # retrocesos: eliminados, pero podrían afectar, ver si hay negativos
        # Filter speed>20
        penguin = penguin.loc[penguin.speed < 20,:].reset_index(drop=True)
        title = f"penguin{peng_number:02}_trip{trip_number}_step{2}"
        penguin.to_csv(_NEWDATA_FOLDER + title + ".csv")
        _plot_track(penguin, dataset = "track_"+title)
        save_boxplot(peng_number, trip_number, penguin, string = 'step2_filtered')


        """
        STEP 3: Outlier removal by standard deviation
        Detect outliers in speed and remove them
        """
        step = 'step_3'
        print(step)
        # Outlier detection
        penguin = detect_speed_outliers(penguin)
        penguin_out = penguin.loc[penguin.outlier !=True,:]
        save_boxplot(peng_number, trip_number, penguin_out, string = 'step3_withoutoutliers')
        title = f"penguin{peng_number:02}_trip{trip_number}_step{3}"
        penguin_out.to_csv(_NEWDATA_FOLDER + title +".csv")
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
        penguin_out.to_csv(_NEWDATA_FOLDER + title + ".csv")
        _plot_track(penguin_out,dataset = "track_"+title)
        """


        """
        STEP 5:
        Calcule temperature gradient, time traveling and normalized direction
        """
        step = 'step_5'
        print(step)
        # Calcule temperature gradient
        penguin_out = calcule_temperature_gradient(penguin_out)
        # Calcule time traveling
        penguin_out = calcule_time_travel(penguin_out)
        # Calcule direction in degrees
        penguin_out = calcule_direction(penguin_out)
        # Normalize direction to 0-1
        penguin_out = normalize_direction(penguin_out)
        # Save data
        title = f"penguin{peng_number:02}_trip{trip_number}_step{5}"
        penguin_out.to_csv(_NEWDATA_FOLDER + title + ".csv")


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
        penguin_fin.to_csv(_NEWDATA_FOLDER + title + ".csv")

        # sigmoid
        # sin bkp
        # con bkp

        # Cierre y borrado del archivo logs si no hay error
        file_log.close()
        os.remove(file_log_route)
    except Exception as error:
        write_log(file, step, error, file_log, today)


def compose_dataset():
    """ Compose dataset """
    files_list = glob.glob(_NEWDATA_FOLDER+'*_final.csv')
    dataset = pd.DataFrame()
    for file in files_list:
        # Concatenate data
        dataset = pd.concat([dataset, pd.read_csv(file)], ignore_index=True)
    dataset.reset_index(drop=True, inplace=True)
    # Save dataset
    dataset.to_csv(_NEWDATA_FOLDER + "dataset.csv", index=False)    


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


def normalize_dataset(dataset_name):
    """ Normalize dataset """
    dataset = pd.read_csv(_NEWDATA_FOLDER+dataset_name+'.csv')
    # Normalize dataset
    scaled_array = StandardScaler().fit_transform(dataset[['lon','lat','temp_gradient','time_travel']])
    dataset_scaled = pd.DataFrame(scaled_array, columns=['lon','lat','temp_gradient','time_travel'])
    dataset_scaled['direction'] = dataset['direction']
    # Save scaled dataset
    dataset_scaled.to_csv(_NEWDATA_FOLDER+dataset_name+'_norm.csv')
    

def reverse_transform_direction(direction):
    """ Reverse transform direction
    :param direction: direction in range 0-1
    :return: direction in degrees
    """
    direction_deg = (direction * 360)
    return direction_deg


def save_errorboxplot(error_values, error_title, error_unit = 'unit', string = None):
    """ Save boxplot of penguin data
    :param error_values: (array) error values
    :param error_title: (string) error name

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

#file = 'viaje2_newpeng03.csv'
#file = 'viaje2_newpeng03_nido75.csv'


if __name__ == '__main__':
    """ Main function """
    # files list
    files_list = glob.glob(_DATA_FOLDER+'viaje*.csv')
    # list to save the PID of the processes created
    procs = [] 
    # statistical analysis
    files_df = compose_statscsv(files_list)
    save_barplot(files_df)
    write_txt_statistics(files_df)
    # TODO: delete: 
    files_list = glob.glob(_DATA_FOLDER+'viaje2_newpeng03_nido75.csv')
    # Paralelized process to analyze each file, with a penguin trip
    for file in files_list:
        """ STEPS 1 to 6 """
        p = Process(target=trajectory_analysis, args=(file,)) #n-1 processes
        procs.append(p)
        p.start()
    # Join all processes
    for p in procs:
        p.join()
    """
    STEP 7:
    Composed dataset and train and test datasets
    """
    step = 'step_7'
    print(step)
    # Compose dataset
    compose_dataset()
    # Split dataset into train and test
    dataset_train_test()
    """
    STEP 8:
    Train and test datasets normalization
    """
    step = 'step_8'
    print(step)
    normalize_dataset('train')
    normalize_dataset('test')
    """
    STEP 9:
    Tunning hyperparameters to select the best model
    """
    step = 'step_9'
    print(step)
    param_dic = {
        "hidden_layer_sizes": [(5,),(50,),(100,)], # 1 hidden layer
        "activation": ["identity", "logistic", "relu"], # activation function
        "solver": ["lbfgs", "sgd", "adam"], # solver
        "alpha": [0.00005,0.0005,0.005], # learning rate (alpha)
        "learning_rate_init":[0.01, 0.001, 0.0001] # initial learning rate
    }
    # Define ANN
    ann = MLPRegressor(early_stopping=True,max_iter=1000) # max_iter=1000
    # Grid search
    grid_param = GridSearchCV(estimator=ann, param_grid=param_dic) # grid search with cross validation
    # Get train and test datasets
    train = pd.read_csv(_NEWDATA_FOLDER+'train_norm.csv')
    test = pd.read_csv(_NEWDATA_FOLDER+'test_norm.csv')
    # NAN values treatment: delete
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    # TODO: falta una semilla en algún sitio???? 
    # Tunning hyperparameters
    grid_param.fit(train[['lon','lat','temp_gradient','time_travel']].values, train['direction'].values)
    # # Test ANN 
    # y_pred = grid_param.predict(test[['lon','lat','temp_gradient','time_travel']].values)
    # Save results of tunning hyperparameters
    ## Save results txt file 
    with open(_NEWMODELS_FOLDER+'grid_params_scores.txt', 'w') as f:
        f.write(str(grid_param.best_score_))
        f.write('\n')
        f.write(str(grid_param.best_params_))
        f.write('\n')
        f.write(str(grid_param.best_estimator_))
        f.write('\n')
        f.write(str(grid_param.best_index_))
        f.write('\n')
        f.write(str(grid_param.scores_))
        f.write('\n')
        f.write(str(grid_param.cv_results_))
        f.close()
    ## Save model
    joblib.dump(grid_param.best_estimator_, _NEWMODELS_FOLDER+'model.pkl')
    ## Plot results
    """
    df = pd.DataFrame(gs.cv_results_)
    results = ['mean_test_score',
            'mean_train_score',
            'std_test_score', 
            'std_train_score']

    def pooled_var(stds):
        # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
        n = 5 # size of each group
        return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))

    fig, axes = plt.subplots(1, len(grid_params), 
                            figsize = (5*len(grid_params), 7),
                            sharey='row')
    axes[0].set_ylabel("Score", fontsize=25)


    for idx, (param_name, param_range) in enumerate(grid_params.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results]\
            .agg({'mean_train_score': 'mean',
                'mean_test_score': 'mean',
                'std_train_score': pooled_var,
                'std_test_score': pooled_var})

        previous_group = df.groupby(f'param_{param_name}')[results]
        axes[idx].set_xlabel(param_name, fontsize=30)
        axes[idx].set_ylim(0.0, 1.1)
        lw = 2
        axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                    color="darkorange", lw=lw)
        axes[idx].fill_between(param_range,grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                        grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                        color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                    color="navy", lw=lw)
        axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                        grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                        color="navy", lw=lw)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=40)
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)

    fig.subplots_adjust(bottom=0.25, top=0.85)  
    plt.show()
    """
    """
    STEP 10:
    Train and test datasets with the best model
    """
    step = 'step_10'
    print(step)
    # Save ANN model and test results
    ## Get params used
    parameters = ann.get_params()
    with open(_NEWMODELS_FOLDER+'ann_params.txt', 'w') as f:
        f.write(str(parameters))
        f.write('\n')
        f.close()
    ## Train ANN
    ann.fit(train[['lon','lat','temp_gradient','time_travel']].values, train['direction'].values)
    ## Test ANN 
    y_pred = ann.predict(test[['lon','lat','temp_gradient','time_travel']].values)
    ## Save results prediction normalized
    test['direction_pred'] = y_pred
    """
    STEP 11:
    Computing errors
    """
    step = 'step_11'
    print(step)
    ## Get errors
    test['mae'] = mean_absolute_error(test['direction'].values, test['direction_pred'].values)
    test['mse'] = mean_squared_error(test['direction'].values, test['direction_pred'].values)
    test['r2'] = r2_score(test['direction'].values, test['direction_pred'].values)
    ### Translate mae to degrees
    test ['mae_deg'] = reverse_transform_direction(test['mae'].values)
    ### Translate mse to degrees
    test ['mse_deg'] = mean_squared_error(
        reverse_transform_direction(test['direction'].values),
        reverse_transform_direction(test['direction_pred'].values))
    ### Compute r-squared with degrees
    test ['r2_deg'] = r2_score(
        reverse_transform_direction(test['direction'].values),
        reverse_transform_direction(test['direction_pred'].values))
    ### Save predictions, errors and metrics
    test.to_csv(_RESULTS_FOLDER + "test_prediction_norm.csv")
    ## Plot errors


    
    



