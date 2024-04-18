import os

import streamlit as st 
import base_utilites as bu

from models import ATCNet_

from helper import (
    set_background , 
    get_data , 
    plot_pie)

def prediction() : 

    set_background(bu.format_path(
        '''
        Assets/
            Images/
                Background/
                    prediction.jpg
        '''
    ))

    st.write(open(bu.format_path(
        '''
        Assets/
            Texts/
                prediction.txt
        '''
    )).read())

    subject = st.text_input('Enter the Subject name')
    sample = st.text_input('Enter the sample name')

    paths = os.listdir(bu.format_path(
        '''
        Assets/
            MAT_Files
        '''))
    paths = [
        bu.format_path(
            '''
            Assets/
                MAT_Files/
            '''
        ) + path
        for path 
        in paths]

    if st.button('Predict') : 

        error = 0

        if sample == '' or subject == '' :
            
            st.error('Please enter the values properly' , icon = '🚨')
            error += 1

        if error == 0 : 
            sample = int(sample)
            subject = int(subject)

            if subject < 0 or subject > 9 : 

                st.error('Please enter the Subject Number in the given range only' , icon = '🚨')
                error += 1
            if sample < 0 or sample > 287 : 

                st.error('Please enter the Sample number in the given ranges', icon = '🚨')
                error += 1

            if error == 0 : 

                with st.spinner('Just a minute, Getting your results !!') : 

                    data_path = paths[int(subject) - 1]
                    x_train = get_data(data_path)
                    model = ATCNet_(
                        n_classes = 4 , 
                        in_chans = 22 , 
                        in_samples = 1125 , 
                        n_windows = 5 , 
                        attention = 'mha' , 
                        eegn_F1 = 16 , 
                        eegn_D = 2 , 
                        eegn_kernelSize = 64 , 
                        eegn_poolSize = 7 , 
                        eegn_dropout = 0.3 , 
                        tcn_depth = 2 , 
                        tcn_kernelSize = 4 , 
                        tcn_filters = 32 , 
                        tcn_dropout = 0.3 , 
                        tcn_activation='elu')

                    model.load_weights(bu.format_path(
                        f'''
                        run-1/
                            subject-{subject}.h5
                        '''))
                    label = [
                        'right_hand' , 
                        'left_hand' , 
                        'foot' , 
                        'tongue']

                    pred = model.predict(x_train)

                labels = ['Right hand', 'Left hand', 'Foot', 'Tongue']
                sizes = pred[sample]
                plot_pie(labels , sizes)
                labels = pred.argmax(axis = -1)
                st.write(f'The predicted value is : {label[labels.tolist()[sample]]}')

def home() : 

    set_background(bu.format_path(
        '''
        Assets/
            Images/
                Background/
                    home.jpg
        '''
    ))

    st.write(open(bu.format_path(
        '''
        Assets/
            Texts/
                home.txt
        '''
    )).read())

def about() : 

    col_1 , col_2 = st.columns(2)    

    col_1.image('Assets/Images/Logos/De_Montfort.png')
    col_2.image('Assets/Images/Logos/APU.jpg')

    style = """
    <style>
    p {
    text-align: center; /* Center align the text */
    font-size: 20px;  /* Set the font size to 24 pixels */
    }
    </style>
    """

    st.markdown(style, unsafe_allow_html=True)  # Apply the CSS style

    st.write("<p align='center'><b>Final Year Project</b></p>", unsafe_allow_html=True)

    col_1 , _ , _ , _ , _ = st.columns(5)
    col_1.image('Assets/Images/Logos/Good_Health.png')

    st.write("<p align='center'><b>Efficient Detection of Human Motion Movements through</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>EEG signals using Deep learning</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>By</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>Venkta Krishna Chaitanya Bysani</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>TP062476</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>APD3F2308CSDA</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'>A report submitted in partial fulfilment of the requiremenets for the degree of</p>", unsafe_allow_html=True)
    st.write("<p align='center'>B.Sc. (Hons) Computer Scienece Specialism in Data Analaytics</p>", unsafe_allow_html=True)
    st.write("<p align='center'>at Asia Pacific University of Technology and Innovation", unsafe_allow_html=True)
    st.write("<p align='center'></p>", unsafe_allow_html=True)
    st.write("<p align='center'></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>Supervised by Dr. Mukil Alagirisamy</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>2nd Markert : Assoc Porf. Dr. Raja Rajeswari</b></p>", unsafe_allow_html=True)
    st.write("<p align='center'><b>2024</b></p>", unsafe_allow_html=True)

options = st.sidebar.selectbox(
    'Navigator' , 
    options = [
        'Home' , 
        'Prediction' , 
        'About'
    ]
)

if options == 'Home' : home()
elif options == 'Prediction' : prediction()
elif options == 'About' : about()
