import os
import numpy as np
import tensorflow as tf

from math import sqrt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.wrappers import Bidirectional
from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.filterwarnings("ignore")
#import h5py


class LSTM:
    
  def __init__(self, Input_Shape, DaysOut=1, Cells=50, Neurons=25, Dropout=0.01, activation=None):
      
      input = layers.Input(shape=Input_Shape)
      x= layers.LSTM(Cells, return_sequences = False, stateful = False, name='LSTM1')(input)
      x = layers.Dropout(Dropout)(x)
      FC = layers.Dense(DaysOut, activation='linear', name="Output")(x)

      self.model= Model(inputs= input, outputs=FC) 
      
  def training(self,trainGen,valGen, Nepoch=15, lr=0.01):
    
    self.model.summary()
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=lr) #0.005
    callBack= tf.keras.callbacks.EarlyStopping(
                                      monitor="val_loss",
                                      min_delta=0,
                                      patience=25,
                                      verbose=0,
                                      mode="min",
                                      baseline=None,
                                      restore_best_weights=True,
                                  )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 4, factor= 0.5, min_lr= 1e-7, verbose=1,min_delta=0)
    self.model.compile(optimizer=opt, loss={"Output":loss_fn}, metrics={"Output":loss_fn})
    

    self.training_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch, callbacks=callBack)
      
          
  def save_model(self,path,modelname="Model"):
    self.model.save(os.path.join(path,modelname)+".h5")

  def load_weights(self,path,checkpoint="Model_weights"):
    self.model.load_weights(os.path.join(path,checkpoint)+".h5")

class ANN:
    
  def __init__(self, Input_Shape, DaysOut=1, Neurons=50, Dropout=0.01, activation='linear'):
        # Input layerB
      input = layers.Input(shape=Input_Shape) #, batch_size= 1
      x = layers.Dense(Neurons, activation=activation,name="Hidden1")(input)
      x = layers.Dropout(Dropout)(x)
      x = layers.Dense(Neurons, activation=activation,name="Hidden2")(x)
      x = layers.Dropout(Dropout)(x)

      FC = layers.Dense(DaysOut, activation='linear',name="Output")(x)

      self.model= Model(inputs= input, outputs=FC)   
      
  def training(self,trainGen,valGen, Nepoch=15, lr=0.01):
    
    self.model.summary()
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=lr) #0.005
    callBack= tf.keras.callbacks.EarlyStopping(
                                      monitor="val_loss",
                                      min_delta=0,
                                      patience=25,
                                      verbose=0,
                                      mode="min",
                                      baseline=None,
                                      restore_best_weights=True,
                                  )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 4, factor= 0.5, min_lr= 1e-7, verbose=1,min_delta=0)
    self.model.compile(optimizer=opt, loss={"Output":loss_fn}, metrics={"Output":loss_fn})

    self.training_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch, callbacks=[callBack, rlrop])
      
      
  def save_model(self,path,modelname="Model"):
    self.model.save(os.path.join(path,modelname)+".h5")

  def load_weights(self,path,checkpoint="Model_weights"):
    self.model.load_weights(os.path.join(path,checkpoint)+".h5")

class ARIMAX:
    def __init__(self, dataset , pdq=(2,1,1)):
        self.x_train, self.x_test, self.exog_train, self.exog_test = dataset
        self.pdq = pdq
        self.predictions = list()
        self.actual = list()
        mse = 0
        self.mse = 0
        self.rmse = 0
        for t in range(len(self.x_test)):
          self.model = ARIMA(self.x_train, order=self.pdq, exog=self.exog_train)
          self.model_fit = self.model.fit()
          step_exog = self.exog_test[t,:].reshape((1,-1))
          output = self.model_fit.forecast(1,exog=step_exog)
          yhat = output[0]
          self.predictions.append(yhat)
          obs = self.x_test[t]
          self.actual.append(obs)
          tempObs = obs.reshape(-1,1)
          tempYhat = yhat.reshape(-1,1)
          mse = mean_squared_error(tempObs,tempYhat)
          self.mse = mean_squared_error(self.actual, self.predictions)
          step_rmse = sqrt(mse)
          self.rmse = sqrt(self.mse)
          self.x_train.append(obs)
          self.exog_train = np.vstack((self.exog_train,step_exog))
          print('predicted=%.2f, expected=%.2f, avg_mse=%.2f, avg_rmse=%.2f, step_mse=%.2f, step_rmse=%.2f' % (yhat, obs, self.mse, self.rmse, mse, step_rmse))