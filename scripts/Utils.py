import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

from os import makedirs, mkdir

from math import sqrt, isnan
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import combine_pvalues
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from random import randint

from tensorflow.keras.activations import gelu
import numpy as np

class ARIMA_Plot:
    def __init__(self, path):
      self.path = path
      self.LoadData()
      self.ACF_Plot()
      self.PACF_Plot()
    
    def LoadData(self):
      data = pd.read_csv(self.path + "data\clean_data.csv")
      self.data = data["MAX_AVG_LOAD"]
    
    def ACF_Plot(self):
      plot_acf(self.data, lags=50)
    
    def PACF_Plot(self):
      plot_pacf(self.data, lags=50)

class ARIMA_Results:
    def __init__(self, path, predictions, actual, save_plot=True):
        self.path = path
        self.predictions = predictions
        self.actual = actual
        self.save_plot = save_plot
        makedirs(self.path,exist_ok=True)
        self.plot()

    def plot(self):
        self.mae = mean_absolute_error(self.actual, self.predictions)
        self.mse = mean_squared_error(self.actual, self.predictions)
        self.rmse = sqrt(self.mse)
        print(f"RMSE score is: {self.rmse}")
        plt.plot(self.predictions, label="pred" , color='red')
        plt.plot(self.actual, label="test" , color='blue')
        plt.legend()
        if self.save_plot:
          plt.savefig(self.path + f"ARIMAX_RMSE_{self.rmse}.png")
        plt.show()
        



class Model_Results:
    def __init__(self, path, model, ValGen, is_LSTM=True, save_plot=False, mae=False, rmse=False, mse=False, loss=True, save_result= True):
        self.save_plot = save_plot
        self.path = path
        self.ValGen = ValGen
        self.save_result = save_result
        self.metrics = {
          'Mean Absolute Error' : mae,
          'Root Mean Squared Error' : rmse,
          'Mean Squared Error' : mse,
          'Loss': loss
        }
        makedirs(self.path,exist_ok=True)

        if is_LSTM:
          self.mode = "LSTM"
        else:
          self.mode = "ANN"
        self.plot(model)
        if self.naan == True:
          self.mse = 1000
          self.rmse = 1000
          self.mae = 1000
        else:
          self.mse, self.rmse, self.mae = self.score(model)
        

    def plot(self, model):
        print(type(model.training_history.history))
        for key, value in self.metrics.items():
            if value:
                metric = key.lower().replace(" ", "_")
                if any(isnan(val) for val in model.training_history.history[f'{metric}']):
                  self.naan = True
                  break
                else:
                  self.naan = False
                  plt.plot(model.training_history.history[f'{metric}'])
                  plt.plot(model.training_history.history[f'val_{metric}'])
                  plt.title(f'Model {key} score')
                  plt.ylabel(key)
                  plt.xlabel('Epoch')
                  plt.legend(['train', 'val'], loc='upper right')
                  if self.save_plot:
                      value = int(model.training_history.history[f'val_{metric}'][-1])
                      if key == "Loss":
                        value = model.training_history.history[f'val_{metric}'][-1]
                      plt.savefig(self.path + f"{self.mode}_{metric}_{value}.png")
                  plt.show()

    def score(self, model):
        def MAE(pred,true):
            diff = pred - true
            diff = np.abs(diff)
            diff = np.mean(diff)
            return diff
        
        def MSE(pred, true):
            return np.mean((pred-true)**2)
            
        Y = self.ValGen.Y if self.mode=="ANN" else self.ValGen.Y[-160:]
        Y = np.expand_dims(Y, axis = 1)
        Y = self.ValGen.inverse_transform(Y)
        print(self.ValGen.Y.shape)
        print(Y.shape)
        Y_pred = model.model.predict(self.ValGen)
        Y_pred = self.ValGen.inverse_transform(Y_pred)
        print(Y_pred.shape)
        
        mse = MSE(Y,Y_pred)
        rmse = sqrt(mse)
        mae=MAE(Y,Y_pred)

        if self.save_result:
              np.save(self.path + "preds.np", Y_pred)
              np.save(self.path + "trues.np", Y)
        print(f"MSE score: {mse}\nRMSE score: {rmse}\nMAE score: {mae}")

        return mse, rmse, mae



class Data_Analysis:
    def __init__(self, path, plot=False, save_file=True, analyize="WEATHER", save_test=True, ARIMAX=True):
        self.path = path
        self.save_file = save_file
        self.analyize = analyize
        self.save_test = save_test
        
        if analyize != "WEATHER" and analyize != "SHIFTED":
            raise AttributeError(f"The analysis format \"{analyize}\" for the following data is not a valid analysis!\nPlease use either \"WEATHER\" or \"SHIFTED\".")
            return 0
        
        self.data = pd.read_csv(self.path + "data/clean_data.csv")
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]

        if analyize == "SHIFTED":
            self.data = pd.read_csv(self.path + "data/shifted_data.csv")
            self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]            
        
        self.CorrelationHypothesisTest(self.data)

        if plot:
          self.PlotAnalysis(self.data)

    def CorrelationHypothesisTest(self, data):    
        targets = {
            0:'MAX_AVG_LOAD'
        }
        print("\n\nHere are some statistics for the average load in each year:")
        duration = 365
        yearly_stats = pd.DataFrame()
        for i in range(4):
          print(f"\nYear {2017+i}:\n{data['MAX_AVG_LOAD'][duration*i:duration*(i+1)].describe()}\n")
          yearly_stats = yearly_stats.append(data['MAX_AVG_LOAD'][duration*i:duration*(i+1)].describe())
        yearly_stats.insert(0,"Year",np.arange(2017,2021))
        if self.save_file:
            yearly_stats.to_csv(self.path+"data/yearly_load_statistics.csv")
        
        features = {
              0:'winddirdegree',
              1:'avgtempC',
              2:'maxtempC',
              3:'mintempC',
              4:'totalprecipMM',
              5:'windspeedKmph',
              6:'FeelsLikeC',
              7:'HeatIndexC',
              8:'humidity',
              9:'cloudcover',
              10:'WindChillC',
              11:'week_id',
              12:'is_weekend'
          }
        
        
        if self.analyize=="SHIFTED":
            features={}
            count=0
            count_1=0
            for i,j in data.items():
              count=count+1
              if count>2:
                features[count_1]=i
                count_1=count_1+1
            print(f"SHIFTED features {features}")


        spear_pvalue = pd.DataFrame(columns=['p_value','feature','target'])
        spear_correlation = pd.DataFrame(columns=['correlation','feature','target'])

        for target in targets:
          for feature in features:
            correlation, pvalue = spearmanr(data[targets[target]],data[features[feature]])
            spear_pvalue = spear_pvalue.append({'p_value':pvalue,'feature':features[feature],'target':targets[target]}, ignore_index=True)
            spear_correlation = spear_correlation.append({'correlation':correlation,'feature':features[feature],'target':targets[target]}, ignore_index=True)


        kendall_pvalue = pd.DataFrame(columns=['p_value','feature','target'])
        kendall_correlation = pd.DataFrame(columns=['correlation','feature','target'])

        for target in targets:
          for feature in features:
            correlation, pvalue = kendalltau(data[targets[target]],data[features[feature]])
            kendall_pvalue = kendall_pvalue.append({'p_value':pvalue,'feature':features[feature],'target':targets[target]}, ignore_index=True)
            kendall_correlation = kendall_correlation.append({'correlation':correlation,'feature':features[feature],'target':targets[target]}, ignore_index=True)



        pearson_pvalue = pd.DataFrame(columns=['p_value','feature','target'])
        pearson_correlation = pd.DataFrame(columns=['correlation','feature','target'])

        for target in targets:
          for feature in features:
            correlation, pvalue = pearsonr(data[targets[target]],data[features[feature]])
            pearson_pvalue = pearson_pvalue.append({'p_value':pvalue,'feature':features[feature],'target':targets[target]}, ignore_index=True)
            pearson_correlation = pearson_correlation.append({'correlation':correlation,'feature':features[feature],'target':targets[target]}, ignore_index=True)


        final_pvalues = pd.DataFrame(columns=['pearson','spearman','kendall','combined_pvalue','feature','target'])
        combined_pvalues = pd.DataFrame(columns=['pearson','spearman','kendall'])
        combined_pvalues['pearson'] = pearson_pvalue['p_value']
        combined_pvalues['spearman'] = spear_pvalue['p_value']
        combined_pvalues['kendall'] = kendall_pvalue['p_value']

        final_correlation = pd.DataFrame(columns=['pearson','spearman','kendall','average_correlation','feature','target'])
        combined_correlation = pd.DataFrame(columns=['pearson','spearman','kendall'])
        combined_correlation['pearson'] = pearson_correlation['correlation']
        combined_correlation['spearman'] = spear_correlation['correlation']
        combined_correlation['kendall'] = kendall_correlation['correlation']

        count=0
        target = 0
        for index, row in combined_pvalues.iterrows():
          static, pvalue = combine_pvalues(row)
          final_pvalues = final_pvalues.append({'combined_pvalue':pvalue,'feature':features[count],'target':targets[target]}, ignore_index=True)
          count = count + 1

        count=0
        target = 0
        for index, row in combined_correlation.iterrows():
          correlation = sum(row)/3
          final_correlation = final_correlation.append({'average_correlation':correlation,'feature':features[count],'target':targets[target]}, ignore_index=True)
          count = count + 1

        final_pvalues['pearson'] = pearson_pvalue['p_value']
        final_pvalues['spearman'] = spear_pvalue['p_value']
        final_pvalues['kendall'] = kendall_pvalue['p_value']

        final_correlation['pearson'] = pearson_correlation['correlation']
        final_correlation['spearman'] = spear_correlation['correlation']
        final_correlation['kendall'] = kendall_correlation['correlation']

        print(f"\n\nHere are the combined pvalues for {self.analyize} hypothesis test:\n")
        print(final_pvalues)
        self.final_pvalues = final_pvalues

        print(f"\n\nHere are the combined correlations for {self.analyize} hypothesis test:\n")
        print(final_correlation)
        self.final_correlation = final_correlation

        if self.save_test:
                final_pvalues.to_csv(self.path + f"data/{self.analyize}_pvalues.csv")
                final_correlation.to_csv(self.path + f"data/{self.analyize}_correlation.csv")
                print(f"\n\nFinal hypothesis tests are saved in location:\n{self.path}data/..")

 
        
    def PlotAnalysis(self,data):
        print("\n\nWe are plotting the analysis of the data!")
        plt.style.use('ggplot')
        duration=365
        initial = 2017
        data['date'] = pd.to_datetime(data['date'])
        scale_y = 1e3
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        try:
          mkdir(self.path + "data/figs")
          mkdir(self.path + "data/figs/analysis")
          
        except:
          pass

        plot_acf(data['MAX_AVG_LOAD'], lags=50)
        plt.savefig(self.path + "data/figs/analysis/acf.png")
        plt.show()
        plot_pacf(data['MAX_AVG_LOAD'], lags=50)
        plt.savefig(self.path + "data/figs/analysis/pacf.png")
        plt.show()

        #Loading the average load into 4 sections, year_2017, year_2018, year_2019, year_2020
        for i in range(4):
          locals()["year_"+str(initial+i)] = np.array(data['MAX_AVG_LOAD'][duration*i:duration*(i+1)])
          
        
        fig, axs = plt.subplots(1, figsize=(10,2.5), dpi=100)
        axs.yaxis.set_major_formatter(ticks_y)
        #plotting the values for each section
        for i in range(4):
          axs.plot(data.date[duration*i:duration*(i+1)], locals()["year_"+str(initial+i)])
          axs.set_ylabel("GigaWatt (GW)", fontsize=8)
          
        plt.savefig(self.path + f"data/figs/analysis/Average_Max_Load.png")
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.show()
        
        #Setting up the subplots
        
        #logging the values for each section and plotting them
        for i in range(4):
          #locals()["log_"+str(initial+i)] = np.log(locals()["year_"+str(initial+i)])
          fig, axs = plt.subplots(1, figsize=(10,2.5), dpi=100)
          axs.yaxis.set_major_formatter(ticks_y)
          axs.set_title(str(initial+i), fontsize=10)
          axs.plot(data.date[duration*i:duration*(i+1)], locals()["year_"+str(initial+i)], c='red')
          axs.set_ylabel("GigaWatt (GW)", fontsize=8)
          plt.savefig(self.path + f"data/figs/analysis/Average_Max_Load_{initial+i}.png")
		
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.show()
          

        #to plot the seasonal differencing   
        fig, axs = plt.subplots(3, figsize=(10,10), dpi=100)
        fig.suptitle('Seasonal Differecning', fontsize=16)
        for i in range(3):
          axs[i].set_title(f'{initial+i+1} - {initial+i}', fontsize=10)
          axs[i].plot(locals()["year_"+str(initial+i+1)] - locals()["year_"+str(initial+i)], c = 'green')
          axs[i].set_ylabel("MegaWatt (MW)", fontsize=8)
        plt.savefig(self.path + f"data/figs/analysis/Seasonal_Differencing.png")
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.show()
        
        print(f'\n\nAugmented Dickey-Fuller test pvalue for year:\n')
        for i in range(4):
          print(f'{initial+i} = {adfuller(locals()["year_"+str(initial+i)])[1]}\n')
        
def experiment_gen(model="ANN", CON_dic = "init", iteration = 0):
      
  if CON_dic == "init":
    CON_dic ={
              "model":[],
              "Cells/Neurons":[],
              "activation":[],
              "lr":[],
              "dropout":[],
              "Days_in":[],
              "p":[],
              "d":[],
              "q":[],
              "iteration":[],
              "RMSE":[],
              "MSE":[],
              "MAE":[],
              "folder_name":[]
              }
  else: 
    assert isinstance(CON_dic, dict), "Something is Wrong"

  if model !="ARIMA":
    if iteration == 0:
      activations=['tanh', 'relu', 'sigmoid', gelu]
      rand_activation = randint(0,len(activations)-1)
      lr = randint(1, 10000) * 10**-6
      cells = randint(15,50)

      dropout = randint(0,30) * 0.01
      days_in = randint(7,30)
      activation = activations[rand_activation] if model == "ANN" else "Null"
    else:
      #print(gelu if "gelu" in CON_dic["activation"][-1] else CON_dic["activation"][-1], "gelu" in CON_dic["activation"][-1])
      activation = gelu if  CON_dic["activation"][-1] not in ['tanh', 'relu', 'sigmoid']  else CON_dic["activation"][-1]
      #activation= CON_dic["activation"][-1]
      lr = CON_dic["lr"][-1]
      cells = CON_dic["Cells/Neurons"][-1]
      dropout = CON_dic["dropout"][-1]
      days_in = CON_dic["Days_in"][-1]


    CON_dic["model"].append(model)
    CON_dic["iteration"].append(iteration)
    CON_dic["Cells/Neurons"].append(cells)
    CON_dic["Days_in"].append(days_in)
    CON_dic["activation"].append(activation)
    CON_dic["lr"].append(lr)
    CON_dic["dropout"].append(dropout)
    CON_dic["p"].append('Null')
    CON_dic["q"].append('Null')
    CON_dic["d"].append('Null')
    rand = randint(0,100000)
    
    if not isinstance(activation, str):
          activation = activation.__name__
    CON_dic["folder_name"].append(f"{model}_i_{iteration}_N_{cells}_DI_{days_in}_{str(activation)}_lr_{lr}_DO_{dropout}_{rand}")

    return CON_dic

  else:
    if iteration == 0:
      p = randint(1,14)
      q = randint(1,2)
      d = 1
    else:
      p = CON_dic["p"][-1]
      q = CON_dic["q"][-1]
      d = CON_dic["d"][-1]

    CON_dic["model"].append(model)
    CON_dic["iteration"].append(iteration)
    CON_dic["Cells/Neurons"].append('Null')
    CON_dic["activation"].append('Null')
    CON_dic["lr"].append('Null')
    CON_dic["dropout"].append('Null')
    CON_dic["Days_in"].append('Null')
    CON_dic["p"].append(p)
    CON_dic["q"].append(q)
    CON_dic["d"].append(d)

    rand = randint(0,100000)
    CON_dic["folder_name"].append(f"{model}_i_{iteration}_p_{p}_d_{d}_q_{q}_{rand}")


    return CON_dic