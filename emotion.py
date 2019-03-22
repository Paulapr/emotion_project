
# coding: utf-8

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import subprocess, sys
import time
from scipy.signal import butter, lfilter
from scipy.fftpack import fft, ifft
from skimage import util
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
import pickle


# # 1. Conectar muse

'''
En terminal:
    - $ muselsl list                       --> Detectar Muse
    - $ muselsl stream --name MUSE-96E2    --> Comenzar stream
    - $ muselsl view --version 2           --> Ver registro, comprobar buena señal, sin artefactos
    - $ muselsl record --duration 900      --> Comenzar a grabar. Duracion en segundos = numero pics*10 + numero pics*2
    
    - Se guardará un archivo csv con el registro: timestamps, canal1, canal2, canal3, canal4
    - Hay que abrir el registro con excel y volverlo a guardar como csv (problema con timestamps)
    - Archivo: 'registro_eeg.csv'
'''

# # 2. Visualización de las imágenes

def visualize_pics (lista):
    pic_time= {}
    for p in lista:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, p])
        timestamp1 = int(round(time.time() * 1000))
        time.sleep(10)                                                   # tiempo que está la foto
        subprocess.call([opener, 'black-screen.png'])
        time.sleep(2)                                                    # tiempo pantalla negra  
        timestamp2 = int(round(time.time() * 1000))                              
        pic_time.update({p:(timestamp1,timestamp2)})
    pic_time_df = pd.DataFrame(list(pic_time.items()), columns=['Picture', 'timestamps'])
    pic_time_df = pic_time_df.join(pd.DataFrame(pic_time_df['timestamps'].values.tolist(), columns=['start_time', 'end_time']))
    pic_time_df = pic_time_df.drop(['timestamps'], axis = 1)
    return pic_time_df       


# # 3. Asignar a cada dato EEG la imagen a la que corresponde

def eeg_pic(pic, eeg):
    eeg.drop(['Right AUX'], axis=1, inplace=True)
    eeg['image']='pic' 
    for row in pic.iterrows():
        eeg.loc[(eeg['timestamps']> row[1].start_time) & 
                (eeg['timestamps'] < row[1].end_time), 'image'] = row[1].Picture
    pic_eeg = eeg[eeg.image != 'pic']
    pic_eeg.to_csv('pic_egg.csv',index=False)
    return pic_eeg

# # 4. Procesar la señal EEG

# Filtrado

def butter_bandpass(lowcut = 1, highcut = 49, fs = 256, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data):
    b, a = butter_bandpass(lowcut = 1, highcut = 49, fs = 256, order=5)
    y = lfilter(b, a, data)
    return y

def butter_cols(df):
    df['af7_filter'] = butter_bandpass_filter(df.AF7)
    df['af8_filter'] = butter_bandpass_filter(df.AF8)
    df['tp9_filter'] = butter_bandpass_filter(df.TP9)
    df['tp10_filter'] = butter_bandpass_filter(df.TP10)
    eeg_filt = df[['image','timestamps','af7_filter','af8_filter','tp9_filter','tp10_filter']]
    return eeg_filt

# Epochs

def epochs(sample):
    epochs = util.view_as_windows(sample, window_shape=(512,), step=256)
    win = np.hanning(512 + 1)[:-1]
    epochs = epochs * win
    df_epochs = (pd.DataFrame(epochs.T))
    df_epochs.columns = ['Win_1','Win_2','Win_3','Win_4','Win_5','Win_6','Win_7','Win_8','Win_9']
    return df_epochs

def sep_pics(df):
    images = list(df.image.unique())
    cols = ['af7_filter','af8_filter','tp9_filter','tp10_filter']
    pictures = []
    
    for pic in images:
        pic = pic.replace('.bmp','')
        pictures.append(pic)
        
    dict_pictures = {}
    for pic in images:
        dict_pictures[pic] = (((df[(df['image'] == pic)])[10:2571]).reset_index()).drop(['image','index'], axis = 1)
    
    dict_pic_epoch = {}

    for pic in images:
        for col in cols:
            sample = ((dict_pictures[pic])[col]).values
            dict_pic_epoch[pic+": "+col] = epochs(sample)
    
    return dict_pic_epoch, pictures

# FFT

def fft_win (to_fft, n, fs):
    win_fft = np.abs(np.fft.fft(to_fft, n=n)[:fs // 2])
    return win_fft

def apply_fft(dict_pic_epoch):
    todas_keys = list(dict_pic_epoch.keys())  
    windows = ['Win_1','Win_2','Win_3','Win_4','Win_5','Win_6','Win_7','Win_8','Win_9']                                
    data_fft_dict = {}
    for key in todas_keys:
        for win in windows:
            to_fft = ((dict_pic_epoch[key])[win]).values
            data_fft_dict[key+": "+win] = fft_win(to_fft, 8, 256)
    data_fft = pd.DataFrame.from_dict(data_fft_dict).T
    return data_fft


# # 5. Crear el dataframe final para que entre al modelo

def data_to_model(data_fft, pictures):
    name_columns = []
    chanels = ['AF7','AF8','TP9','TP10']
    windows = ['W1','W2','W3','W4','W5','W6','W7','W8','W9']
    fft_bins = ['bin1','bin2','bin3','bin4','bin5','bin6','bin7','bin8']
    for chanel in chanels:
        for window in windows:
            for bins in fft_bins:
                name_columns.append(('{}'+'-'+'{}'+'-'+'{}').format(chanel, window, bins)) 
    df_model = pd.DataFrame (data=None, index = pictures, columns=name_columns)
    images_cols = []
    images_e = []
    for i in range(len(pictures)):
        images_cols.append(np.array(data_fft.iloc[i*36:(i+1)*36]))
    for j in range(len(images_cols)): 
        for array in images_cols[j]:
            for e in array:
                images_e.append(e)
    for i in range(len(pictures)):
        df_model.iloc[i] = images_e[i*288:(i+1)*288]
    df_model.to_csv('pics_eeg_process.csv', index= True)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df_model), columns=[df_model.columns], index = df_model.index )
    return X

# # 6. Meter al modelo para ver la prediccion

def prediction(X):
    dectree_model = pickle.load(open('dectree_model.sav', 'rb'))
    ynew = dectree_model.predict(X)
    for i in range(len(X)):
        print("Predicted by decision tree for image {} -> Valencia: {:.2f}, Arousal: {:.2f}".format(i+1, ynew[i][0],ynew[i][1]))
    forest_model = pickle.load(open('forest_model.sav', 'rb'))
    ynew = forest_model.predict(X)
    for i in range(len(X)):
        print("Predicted by random forest for image {} -> Valencia: {:.2f}, Arousal: {:.2f}".format(i+1, ynew[i][0],ynew[i][1]))

def prediction2(X):
    dectree_model = pickle.load(open('dectree_model.sav', 'rb'))
    y_dectree = dectree_model.predict(X)
    forest_model = pickle.load(open('forest_model.sav', 'rb'))
    y_forest = forest_model.predict(X)
    return y_forest , y_dectree

# # 7. Visualizar las predecciones obtenidas

def pred_df(y_forest, y_dectree, orden):
    cols = ['Picture', 'Forest valence', 'Forest Arousal','Dec.Tree valence', 'Dec.Tree Arousal']
    predictions_df = pd.DataFrame()
    predictions_df['Picture'] = orden
    predictions_df['Forest Valence'] = y_forest[:,0]
    predictions_df['Dec.Tree Valence'] = y_dectree[:,0]
    predictions_df['Forest Arousal'] = y_forest[:,1]
    predictions_df['Dec.Tree Arousal'] = y_dectree[:,1]
    imagenes = []
    for row in predictions_df['Picture']:
        row = row.replace('.jpg','')
        imagenes.append(row)
    return predictions_df

def plot1 (predictions):
    fig = plt.figure(figsize= (8,8))
    plt.scatter(predictions['Forest Valence'],predictions['Forest Arousal'], label = 'Random Forest', s=100, alpha = 0.5)
    plt.scatter(predictions['Dec.Tree Valence'],predictions['Dec.Tree Arousal'], label = 'Decision Tree', s=100, alpha = 0.5)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend(fontsize = 15)
    plt.xlabel('Valencia', fontsize = 15)
    plt.ylabel('Arousal', fontsize = 15)
    fig.savefig('forestvsdectree.png')
    return ax

def plot2 (predictions):
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(predictions['Forest Valence'],predictions['Forest Arousal'], s=100, alpha = 0.5)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('Valencia')
    plt.ylabel('Arousal')
    for i, txt in enumerate (imagenes):
        ax.annotate(txt, (predictions['Forest Valence'][i],predictions['Forest Arousal'][i]), fontsize=15)
    fig.savefig('forest_pred.png')
    return ax

# PIPELINE:

list_pictures = ([f for f in listdir('/pics_to_predict') if isfile(join('/pics_to_predict', f))])   # 1. CARGA DE LA CARPETA CON LAS IMÁGENES
random.shuffle(list_pictures)
orden = list_pictures
pic_time_df = visualize_pics(list_pictures)
registro_eeg = pd.read_csv('registro.csv')    # 2. CARGA DE ARCHIVO CSV CREADO POR EL REGISTRO MUSE                                           
pic_eeg = eeg_pic(pic_time_df, registro_eeg)
df = (butter_cols(pic_eeg))
dict_pic_epoch = sep_pics(df)[0]
pictures = sep_pics(df)[1]
data_fft = apply_fft(dict_pic_epoch)
X = data_to_model(data_fft, pictures)
prediction(X)
y_forest = prediction2(X)[0]
y_dectree = prediction2(X)[1]
