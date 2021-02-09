def all_code():
    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    from IPython.display import Image
    from IPython.display import SVG
    from IPython.display import display                               
    from ipywidgets import interactive
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import random
    import re
    from math import sqrt, exp, log
    from sklearn.linear_model import Lasso, Ridge, RidgeCV, SGDRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score
    import keras
    import keras.backend as K
    from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, TimeDistributed, RepeatVector, Flatten, Input, Dropout, LSTM, concatenate, Reshape, Conv1D, GlobalMaxPool1D
    from keras.utils import plot_model

    from lionets import LioNets
    from nbeats_keras.model import NBeatsNet
    from Interpretable_PCA import iPCA
    from altruist.altruist import Altruist
    from utilities.evaluation import Evaluation
    from utilities.load_dataset import Load_Dataset

    from lime.lime_text import LimeTextExplainer
    from lime.lime_tabular import LimeTabularExplainer
    fm, feature_names = Load_Dataset.load_data_turbofan(False)

    fm1_train = fm['FaultMode1']['df_train']
    fm1_train_target = fm1_train['RUL'].values
    fm1_test= fm['FaultMode1']['df_test']
    fm1_test_target = fm1_test['RUL'].values

    LSTM_train = fm1_train.drop(columns=['t', 'os_1', 'os_2', 'os_3', 's_01', 's_05', 's_06', 's_10', 's_16', 's_18', 's_19', 's_22', 's_23', 's_24', 's_25', 's_26'])
    LSTM_test = fm1_test.drop(columns=['t', 'os_1', 'os_2', 'os_3', 's_01', 's_05', 's_06', 's_10', 's_16', 's_18', 's_19', 's_22', 's_23', 's_24', 's_25', 's_26'])

    train_units = set(LSTM_train['u'].values)
    test_units = set(LSTM_test['u'].values)

    sensors = ['s_02', 's_03', 's_04', 's_07', 's_08', 's_09', 's_11', 's_12',
                's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    scalers = {}
    for column in sensors:
        scaler = MinMaxScaler(feature_range=(0,1))
        LSTM_train[column] = scaler.fit_transform(LSTM_train[column].values.reshape(-1,1))
        LSTM_test[column] = scaler.transform(LSTM_test[column].values.reshape(-1,1))
        scalers[column] = scaler

    unit_scalers = {}
    window = 50
    temp_LSTM_x_train = []
    LSTM_y_train = []
    for unit in train_units:
        temp_unit = LSTM_train[LSTM_train['u']==unit].drop(columns=['u','RUL']).values
        temp_unit_RUL = LSTM_train[LSTM_train['u']==unit]['RUL'].values
        
        for i in range(len(temp_unit) - window + 1):#elekse edw an len temp_unit - window > 0
            temp_instance = []
            for j in range(window):
                temp_instance.append(temp_unit[i+j])
            temp_LSTM_x_train.append(np.array(temp_instance))
            LSTM_y_train.append(temp_unit_RUL[i+window-1])
    LSTM_y_train = np.array(LSTM_y_train)
    LSTM_x_train = np.array(temp_LSTM_x_train)

    temp_LSTM_x_test = []
    LSTM_y_test = []
    for unit in test_units:
        temp_unit = LSTM_test[LSTM_test['u']==unit].drop(columns=['u','RUL']).values
        temp_unit_RUL = LSTM_test[LSTM_test['u']==unit]['RUL'].values
            
        for i in range(len(temp_unit) - window + 1):#elekse edw an len temp_unit - window > 0
            temp_instance = []
            for j in range(window):
                temp_instance.append(temp_unit[i+j])
            temp_LSTM_x_test.append(np.array(temp_instance))
            LSTM_y_test.append(temp_unit_RUL[i+window-1])
    LSTM_y_test = np.array(LSTM_y_test)
    LSTM_x_test = np.array(temp_LSTM_x_test)

    temp_LSTM_y_train = [[i] for i in LSTM_y_train]
    temp_LSTM_y_test = [[i] for i in LSTM_y_test]
    target_scaler = MinMaxScaler()
    target_scaler.fit(temp_LSTM_y_train)
    temp_LSTM_y_train = target_scaler.transform(temp_LSTM_y_train)
    temp_LSTM_y_test = target_scaler.transform(temp_LSTM_y_test)

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    feature_names = fm1_train.columns
    encoder_input = Input(shape=(LSTM_x_train[0].shape))

    encoder_x = LSTM(units=80, return_sequences=True, activation='tanh')(encoder_input)
    encoder_x = Dropout(0.7)(encoder_x)
    encoder_x = LSTM(units=40, return_sequences=False, activation='tanh')(encoder_x)

    encoder_y = Conv1D(filters=40,kernel_size=3,activation='tanh')(encoder_input)
    encoder_y = GlobalMaxPool1D()(encoder_y)

    encoded = concatenate([encoder_x,encoder_y])
    encoded = Dropout(0.7)(encoded)
    encoded = Dense(80, activation='tanh')(encoded)#Relu and selu
    encoded = Dropout(0.7)(encoded)
    encoded = Dense(40, activation='tanh')(encoded)#Relu and selu
    predictions = Dense(1, activation='sigmoid')(encoded)#Relu and selu
    predictor = Model(encoder_input,predictions)

    predictor.compile(optimizer="adam",loss=[root_mean_squared_error],metrics=['mae','mse'])

    checkpoint_name = 'TEDS_Predictor_RUL.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 2, save_best_only = True, mode ='auto')

    weights_file = 'weights/TEDS_Predictor_RUL.hdf5' # choose the best checkpoint few features
    predictor.load_weights(weights_file) # load it
    predictor.compile(optimizer="adam",loss=[root_mean_squared_error],metrics=['mae','mse'])

    temp_pred = target_scaler.inverse_transform(predictor.predict(LSTM_x_train))
    predictions = [i[0] for i in temp_pred]

    temp_pred = target_scaler.inverse_transform(predictor.predict(LSTM_x_test))
    predictions = [i[0] for i in temp_pred]

    encoder = Model(input=predictor.input, output=[predictor.layers[-2].output])
    encoder.trainable = False
    encoder.compile(optimizer="adam",loss=[root_mean_squared_error],metrics=['mae','mse'])

    encoded_LSTM_x_train = encoder.predict(LSTM_x_train)
    encoded_LSTM_x_test = encoder.predict(LSTM_x_test)

    encoded_input = Input(shape=(encoded_LSTM_x_train[0].shape))
    decoded = Dense(120, activation='tanh')(encoded_input)
    decoded = Dropout(0.5)(decoded)

    decoded_y = RepeatVector(54)(decoded)
    decoded_y = Conv1D(filters=50,kernel_size=5,activation='tanh')(decoded_y)

    decoded_x = RepeatVector(50)(decoded)
    decoded_x = LSTM(units=80, return_sequences=True, activation='tanh')(decoded_x)
    decoded_x = Dropout(0.5)(decoded_x)
    decoded_x = LSTM(units=50, return_sequences=True, activation='tanh')(decoded_x)

    decoded = concatenate([decoded_x,decoded_y])
    decoded = LSTM(50, return_sequences=True, activation='sigmoid')(decoded)
    decoded = Dropout(0.5)(decoded)
    decoded = LSTM(14, return_sequences=True, activation='sigmoid')(decoded)

    decoder = Model(encoded_input,decoded)

    decoder.compile(optimizer="adam",loss=[root_mean_squared_error],metrics=['mae','mse'])

    checkpoint_name = 'TEDS_Decoder_RUL.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 2, save_best_only = True, mode ='auto')

    weights_file = 'weights/TEDS_Decoder_RUL.hdf5' # choose the best checkpoint few features
    decoder.load_weights(weights_file) # load it
    decoder.compile(optimizer="adam",loss=[root_mean_squared_error],metrics=['mae','mse'])

    window = 50
    forecast_timesteps = 5

    temp_fc_x_train = []
    temp_fc_y_train = []
    for unit in train_units:
        temp_unit = LSTM_train[LSTM_train['u']==unit].drop(columns=['u','RUL']).values
    
        for i in range(len(temp_unit) - window - forecast_timesteps + 1):#elekse edw an len temp_unit - window > 0
            temp_instance_x = []
            temp_instance_y = []
            for j in range(window):
                temp_instance_x.append(temp_unit[i+j])
            for z in range(forecast_timesteps):
                temp_instance_y.append(temp_unit[i+j+z+1])            
            temp_fc_x_train.append(np.array(temp_instance_x))
            temp_fc_y_train.append(np.array(temp_instance_y))       
    fc_x_train = np.array(temp_fc_x_train)
    fc_y_train = np.array(temp_fc_y_train)

    temp_fc_x_test = []
    temp_fc_y_test = []
    for unit in test_units:
        temp_unit = LSTM_test[LSTM_test['u']==unit].drop(columns=['u','RUL']).values
            
        for i in range(len(temp_unit) - window - forecast_timesteps + 1):#elekse edw an len temp_unit - window > 0
            temp_instance_x = []
            temp_instance_y = []
            for j in range(window):
                temp_instance_x.append(temp_unit[i+j])
            for z in range(forecast_timesteps):
                temp_instance_y.append(temp_unit[i+j+z+1])            
            temp_fc_x_test.append(np.array(temp_instance_x))
            temp_fc_y_test.append(np.array(temp_instance_y))       
    fc_x_test = np.array(temp_fc_x_test)
    fc_y_test = np.array(temp_fc_y_test)

    forecast_input = Input(shape=(LSTM_x_train[0].shape))

    forecast_x = LSTM(units=120, return_sequences=True, activation='tanh')(forecast_input)
    forecast_x = Dropout(0.7)(forecast_x)
    forecast_x = LSTM(units=50, return_sequences=True, activation='tanh')(forecast_x)
    forecast_x = Conv1D(filters=50,kernel_size=46,activation='tanh')(forecast_x)

    forecast_y = Conv1D(filters=50,kernel_size=46,activation='tanh')(forecast_input)

    forecast = concatenate([forecast_y,forecast_x])
    forecast = Dropout(0.7)(forecast)
    forecast = LSTM(40, return_sequences=True, activation='relu')(forecast)#Relu and selu
    forecast = Dropout(0.7)(forecast)
    predictions = LSTM(14, return_sequences=True, activation='linear')(forecast)#Relu and selu
    forecaster = Model(forecast_input,predictions)
    forecaster.compile(optimizer="adam", loss=[root_mean_squared_error],metrics=['mae','mse'])

    checkpoint_name = 'weights/TEDS_Forecaster_RUL.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 2, save_best_only = True, mode ='auto')

    weights_file = 'weights/TEDS_Forecaster_RUL.hdf5' # choose the best checkpoint few features
    forecaster.load_weights(weights_file) # load it
    forecaster.compile(optimizer="adam",loss=[root_mean_squared_error],metrics=['mae','mse'])

    predictions = forecaster.predict(fc_x_train)
    predictions = forecaster.predict(fc_x_test)

    # Definition of the model.
    nbeats = NBeatsNet(input_dim = 14, backcast_length=50, forecast_length=5,
                    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), 
                    nb_blocks_per_stack=2,
                    thetas_dim=(4, 4), 
                    share_weights_in_stack=True, 
                    hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    nbeats.compile_model(loss='mae', learning_rate=1e-6)
    
    # Train the model.
    # nbeats.fit(fc_x_train, fc_y_train, verbose=1, validation_split=0.3, epochs=100, batch_size=128)

    # Save the model for later.
    # nbeats.save('weights/TEDS_NBeats_RUL.hdf5')

    # # Load the model.
    nbeats = NBeatsNet.load('weights/TEDS_NBeats_RUL.hdf5')

    window = 50
    forecast_steps = 5

    rul_train, xyz7_x_train, xyz7_y_train, rul_temp = [],[],[],[]
    for unit in train_units:
        temp_unit = LSTM_train[LSTM_train['u']==unit].drop(columns=['u','RUL']).values   
        for i in range(len(temp_unit) - window + 1): # elekse edw an len temp_unit - window > 0
            temp_instance = np.array(temp_unit[i:i+window])
            rul_temp.append(temp_instance)
            xyz7_x_train.append(temp_instance[:-forecast_steps])
            xyz7_y_train.append(temp_instance[-forecast_steps:])

    rul_train = predictor.predict(np.array(rul_temp))
    xyz7_x_train = np.array(xyz7_x_train)
    xyz7_y_train = np.array(xyz7_y_train)

    rul_test, xyz7_x_test, xyz7_y_test, rul_temp = [],[],[],[]
    for unit in test_units:
        temp_unit = LSTM_test[LSTM_test['u']==unit].drop(columns=['u','RUL']).values   
        for i in range(len(temp_unit) - window + 1): # elekse edw an len temp_unit - window > 0
            temp_instance = np.array(temp_unit[i:i+window])
            rul_temp.append(temp_instance)
            xyz7_x_test.append(temp_instance[:-forecast_steps])
            xyz7_y_test.append(temp_instance[-forecast_steps:])

    rul_test = predictor.predict(np.array(rul_temp))
    xyz7_x_test = np.array(xyz7_x_test)
    xyz7_y_test = np.array(xyz7_y_test)


    forecast_input = Input(shape=(xyz7_x_train[0].shape))
    rul_input = Input(shape = (rul_train[0].shape))
    rul = RepeatVector(5)(rul_input)

    forecast_x = LSTM(units=120, return_sequences=True, activation='tanh')(forecast_input)
    forecast_x = Dropout(0.7)(forecast_x)
    forecast_x = LSTM(units=50, return_sequences=True, activation='tanh')(forecast_x)
    forecast_x = Conv1D(filters=50,kernel_size=41,activation='tanh')(forecast_x)
    forecast_x = concatenate([forecast_x, rul])


    forecast_y = Conv1D(filters=50,kernel_size=41,activation='tanh')(forecast_input)
    forecast_y = concatenate([forecast_y, rul])


    forecast = concatenate([forecast_y, forecast_x, rul])
    forecast = Dropout(0.7)(forecast)
    forecast = LSTM(40, return_sequences=True, activation='relu')(forecast)#Relu and selu
    forecast = concatenate([forecast, rul])
    forecast = Dropout(0.7)(forecast)
    predictions = LSTM(14, return_sequences=True, activation='linear')(forecast)#Relu and selu

    xyz7_model = Model([forecast_input, rul_input],predictions)
    opt = keras.optimizers.Adam(lr=0.001)
    xyz7_model.compile(optimizer=opt, loss=[root_mean_squared_error],metrics=['mae','mse'])

    checkpoint_name = 'weights/TEDS_XYZ7_RUL.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 2, save_best_only = True, mode ='auto')

    weights_file = 'weights/TEDS_XYZ7_RUL.hdf5' # choose the best checkpoint few features
    xyz7_model.load_weights(weights_file) # load it
    xyz7_model.compile(optimizer=opt,loss=[root_mean_squared_error],metrics=['mae','mse'])


    lionet = LioNets(predictor, decoder, encoder, LSTM_x_train)

    iml_methods = {'LioNets':lionet} # {'Lime':lime}

    nn_forecasters = {'forecaster':forecaster, 'nbeats':nbeats, 'xyz7_model':xyz7_model}

    return predictor, iml_methods, nn_forecasters, LSTM_x_train, LSTM_y_train, sensors, target_scaler
