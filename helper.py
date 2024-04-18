import streamlit as st 
import base_utilites as bu

import numpy as np
import scipy.io as sio
import streamlit as st 
import base_utilites as bu
import matplotlib.pyplot as plt

from models import ATCNet_
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score , cohen_kappa_score


def set_background(png_file) : 

    bin_str = bu.get_base64(png_file)
    page_bg_img = open('Assets/Texts/background_img.txt').read() % bin_str

    st.markdown(page_bg_img , unsafe_allow_html = True)

def load_kaggle_data(
    data_file_path , 
    is_training , 
    include_all_trials = True) : 

    n_channels = 22
    n_tests = 6 * 48
    window_length = 7 * 250

    sampling_rate = 250          
    start_time_point = int(1.5 * sampling_rate)  
    end_time_point = int(6 * sampling_rate)    

    class_labels = np.zeros(n_tests)
    data_array = np.zeros((n_tests, n_channels, window_length))

    valid_trial_count = 0

    if is_training : data = sio.loadmat(data_file_path)
    else : data = sio.loadmat(data_file_path)
    
    data_content = data['data']

    for index in range(data_content.size) : 

        data_content1 = data_content[0, index]
        data_content2 = [data_content1[0, 0]]
        data_content3 = data_content2[0]

        X_data = data_content3[0]
        y_data = data_content3[2]

        trial_data = data_content3[1]
        artifacts_data = data_content3[5]


        for trial_index in range(trial_data.size) : 

            if artifacts_data[trial_index] != 0 and not include_all_trials : continue

            data_array[valid_trial_count , : , :] = np.transpose(
                X_data[
                    int(trial_data[trial_index , 0]) : (int(trial_data[trial_index, 0]) + window_length) , : 22
                ]
            )
            class_labels[valid_trial_count] = int(y_data[trial_index , 0])
            valid_trial_count += 1

    data_array = data_array[0 : valid_trial_count , : , start_time_point : end_time_point]
    
    return data_array


def standardize_data(X_train , channels) : 

    for channel in range(channels) : 

        scaler = StandardScaler()
        
        scaler.fit(X_train[: , 0 , channel , :])
        
        X_train[: , 0 , channel, :] = scaler.transform(X_train[: , 0 , channel , :])

    return X_train

def get_data(path) : 

    X_train = load_kaggle_data(path , True)

    num_train , num_channel , T = X_train.shape
    X_train = X_train.reshape(num_train , 1 , num_channel , T)

    X_train = standardize_data(X_train , num_channel)

    return X_train

def plot_pie(labels , sizes) : 

    fig1 , ax1 = plt.subplots()
    ax1.pie(
        sizes , 
        labels = labels , 
        autopct = '%1.1f%%' , 
        startangle = 90)
    
    fig1.patch.set_facecolor('none')
    fig1.patch.set_alpha(0.0)

    plt.setp(
        ax1.pie(
            sizes , 
            labels = labels , 
            autopct = '%1.1f%%' , 
            startangle = 90
        )[1] , 
        color = 'white'
    )

    ax1.axis('equal')  
    plt.title('Interactive Pie Chart')
    st.pyplot(fig1)
