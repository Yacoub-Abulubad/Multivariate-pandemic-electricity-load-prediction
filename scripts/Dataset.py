from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
#import random
from math import ceil

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        std = self.std
        mean = self.mean     
        return (data - mean) / std

    def inverse_transform(self, data):
        std = self.std
        mean = self.mean
        return (data * std) + mean

class Flat_Dataset(Sequence):
      
    def __init__(self, path ,Features, batch_size = 20, is_val = False, DaysIn = 7, is_LSTM = False, Mobility = False, covid_only= False, Dont_scale=['is_weekend']):
        """[Initialize the class]

        Args:
            path ([str]): [path to the folder containtin the script]
            Features (Dataframe): [dataframe containing the combined p-vales from (Analysis.final_pvalues)]
            batch_size (int, optional): [batch size]. Defaults to 20.
            is_val (bool, optional): [True for testing]. Defaults to False.
            DaysIn (int, optional): [Numbe of lag days for input]. Defaults to 7.
            is_LSTM (bool, optional): [Set True is dataset used for LSTM model]. Defaults to False.
            Mobility (bool, optional): [To include mobility factor or not]. Defaults to False.
            covid_only (bool, optional): [To use only the first half year of 2020]. Defaults to False.
            Dont_scale (list, optional): [names of binary categorial features (needed so they dont get scaled)]. Defaults to ['is_weekend'].
        """
        self.Dont_scale = Dont_scale
        self.is_val = is_val
        self.target = "MAX_AVG_LOAD"
        self.scaler = StandardScaler()
        self.scaler_fet = StandardScaler()
        self.DaysIn = DaysIn
        self.batch_size = batch_size
        self.is_LSTM = is_LSTM
        self.covid_only = covid_only
        self.path = path
        self.KeepColumn = Features[Features["combined_pvalue"]<0.05]['feature'].tolist()
        self.KeepColumn.append(self.target)
        if Mobility:
            self.mobility_col = ['residential_percent_change_from_baseline',
                            'workplaces_percent_change_from_baseline',
                            'transit_stations_percent_change_from_baseline',
                            'parks_percent_change_from_baseline',
                            'grocery_and_pharmacy_percent_change_from_baseline',
                            'retail_and_recreation_percent_change_from_baseline']
            self.KeepColumn = self.KeepColumn + self.mobility_col
            self.Dont_scale = self.Dont_scale + self.mobility_col

        self.load_scale_data()
        self.idxList = [i for i in range(len(self.X))]
    def load_scale_data(self):
        """[load scale divide the dataset]
        """
        df_raw = pd.read_csv(self.path + "data/clean_data.csv")
        df_raw = df_raw.filter(items=self.KeepColumn)
        dataset = self.ShiftPower(df_raw, self.DaysIn)
        self.Y = dataset[self.target].to_numpy()
        self.X = dataset.drop(self.target,axis=1, inplace=False)
        DontScale = []
        for col in self.Dont_scale:
            DontScale.append(self.X.columns.get_loc(col))
        
        self.X = self.X.to_numpy()

        if self.is_val:
            self.X = self.X[-160:]
            self.Y = self.Y[-160:]
        elif self.covid_only:
            self.X = self.X[-321:-160]
            self.Y = self.Y[-321:-160]   
        else:
            self.X = self.X[:-160]
            self.Y = self.Y[:-160]   

        
        self.scaler.fit(self.Y)
        self.scaler_fet.fit(self.X)
        self.Y = self.scaler.transform(self.Y)
        X0 = self.scaler_fet.transform(self.X)
        X0[:,DontScale] = self.X[:,DontScale]
        self.X = X0

    def __getitem__(self, index):
        """[function to return a single batch ]

        Args:
            index ([type]): [index of returned batch]

        Returns:
            [type]: [the title, a one hot encoded vector representing the class of the title]
        """
        start = index*self.batch_size
        ending = index*self.batch_size + self.batch_size
        if ending >= len(self.idxList):
            ending = len(self.idxList) 
        tempList = self.idxList[start:ending]
        X_batch = self.X[tempList]
        Y_batch = self.Y[tempList]

        if self.is_LSTM:
            X_batch = np.expand_dims(X_batch, axis = -1)
        return X_batch, Y_batch


    def on_epoch_end(self):
        """[Actions performed at the end of each epoch (shuffling)]
        """
        if not self.is_val:
            np.random.seed(20)
            np.random.shuffle(self.idxList)

    def __len__(self):
        """[number of bacthes in the dataloader]

        """
        return int(ceil(len(self.idxList) / self.batch_size))
        
    def inverse_transform(self, data):
        """[Can be used to scale a numpy array back to the original scale]

        Args:
            data (np.array): [array to be de-scaled (only power)]

        Returns:
            np.array: [inverse scaled array]
        """
        return self.scaler.inverse_transform(data)


    def ShiftPower(self, df_raw, shiftBy):
        """[Create shifted (lagged) power features]

        Args:
            df_raw (pd.DataFrame): [dataframe to add the lagged features to]
            shiftBy (int): [number of lagged (shifting) days]

        Returns:
            pd.DataFrame: [dataframe with the lagged features]
        """
        data = df_raw
        
        for i in range(1,shiftBy+1):
            pos = len(data.columns)
            data.insert(pos,f'MAX_AVG_LOAD_SHIFT_{i}',data['MAX_AVG_LOAD'])
            data[f'MAX_AVG_LOAD_SHIFT_{i}'] = data[f'MAX_AVG_LOAD_SHIFT_{i}'].shift(periods = i)
        data = data.dropna()

        data = data.reset_index()
        data = data.drop('index',axis=1, inplace=False)

        return data


class Mat_Dataset(Sequence):
      
    def __init__(self, path ,Features, batch_size = 20, is_val = False, DaysIn = 7, is_LSTM = True, Mobility = False, covid_only= False, Dont_scale=['is_weekend']):
        """[Initialize the class]

        Args:
            path ([str]): [path to the folder containtin the script]
            Features (Dataframe): [dataframe containing the combined p-vales from (Analysis.final_pvalues)]
            batch_size (int, optional): [batch size]. Defaults to 20.
            is_val (bool, optional): [True for testing]. Defaults to False.
            DaysIn (int, optional): [Numbe of lag days for input]. Defaults to 7.
            is_LSTM (bool, optional): [Set True is dataset used for LSTM model if False returns a flattened matric (vector)]. Defaults to True.
            Mobility (bool, optional): [To include mobility factor or not]. Defaults to False.
            covid_only (bool, optional): [To use only the first half year of 2020]. Defaults to False.
            Dont_scale (list, optional): [names of binary categorial features (needed so they dont get scaled)]. Defaults to ['is_weekend'].
        """
        self.Dont_scale = Dont_scale
        self.is_val = is_val
        self.target = "MAX_AVG_LOAD"
        self.scaler = StandardScaler()
        self.scaler_fet = StandardScaler()
        self.DaysIn = DaysIn
        self.batch_size = batch_size
        self.is_LSTM = is_LSTM
        self.covid_only = covid_only
        self.path = path
        self.KeepColumn = Features[Features["combined_pvalue"]<0.05]['feature'].tolist()
        self.KeepColumn.append(self.target)
        if Mobility:
            self.mobility_col = ['residential_percent_change_from_baseline',
                            'workplaces_percent_change_from_baseline',
                            'transit_stations_percent_change_from_baseline',
                            'parks_percent_change_from_baseline',
                            'grocery_and_pharmacy_percent_change_from_baseline',
                            'retail_and_recreation_percent_change_from_baseline']
            self.KeepColumn = self.KeepColumn + self.mobility_col
            self.Dont_scale = self.Dont_scale + self.mobility_col

        self.load_scale_data()
        self.idxList = [i for i in range(self.DaysIn,len(self.X))] 
    def load_scale_data(self):
        """[load scale divide the dataset]
        """
        df_raw = pd.read_csv(self.path + "data/clean_data.csv")
        df_raw = df_raw.filter(items=self.KeepColumn)
        dataset = self.ShiftPower(df_raw, 1)
        self.Y = dataset[self.target].to_numpy()
        self.X = dataset.drop(self.target,axis=1, inplace=False)
        DontScale = []
        for col in self.Dont_scale:
            DontScale.append(self.X.columns.get_loc(col))
        
        self.X = self.X.to_numpy()

        if self.is_val:
            self.X = self.X[-(self.DaysIn+160):]
            self.Y = self.Y[-(self.DaysIn+160):]
        elif self.covid_only:
            self.X = self.X[-321:-160]
            self.Y = self.Y[-160]   
        else:
            self.X = self.X[:-160]
            self.Y = self.Y[:-160]   

        
        self.scaler.fit(self.Y)
        self.scaler_fet.fit(self.X)
        self.Y = self.scaler.transform(self.Y)
        X0 = self.scaler_fet.transform(self.X)
        X0[:,DontScale] = self.X[:,DontScale]
        self.X = X0

    def __getitem__(self, index):
        """[function to return a single batch ]

        Args:
            index ([type]): [index of returned batch]

        Returns:
            [type]: [the title, a one hot encoded vector representing the class of the title]
        """
        #idxList = DaysIn-100
        start = (index*self.batch_size)
        ending = (index*self.batch_size) + self.batch_size
        if ending >= len(self.idxList):
            ending = len(self.idxList) 
        tempList = self.idxList[start:ending]
        
        #tempList2 = list(map(lambda x: x - self.DaysIn, tempList))
        if self.is_LSTM:
            X_batch = np.array([self.X[id-self.DaysIn+1:id+1] for id in tempList])
        else:
            X_batch = np.array([self.X[id-self.DaysIn+1:id+1].flatten() for id in tempList])
        Y_batch = self.Y[tempList]
        return np.flip(X_batch, axis=2), Y_batch


    def on_epoch_end(self):
        """[Actions performed at the end of each epoch (shuffling)]
        """
        if not self.is_val:
            np.random.seed(20)
            np.random.shuffle(self.idxList)

    def __len__(self):
        """[number of bacthes in the dataloader]

        """
        return int(ceil(len(self.idxList) / self.batch_size))
        
    def inverse_transform(self, data):
        """[Can be used to scale a numpy array back to the original scale]

        Args:
            data (np.array): [array to be de-scaled (only power)]

        Returns:
            np.array: [inverse scaled array]
        """
        return self.scaler.inverse_transform(data)


    def ShiftPower(self, df_raw, shiftBy):
        """[Create shifted (lagged) power features]

        Args:
            df_raw (pd.DataFrame): [dataframe to add the lagged features to]
            shiftBy (int): [number of lagged (shifting) days]

        Returns:
            pd.DataFrame: [dataframe with the lagged features]
        """
        data = df_raw
        
        for i in range(1,shiftBy+1):
            pos = len(data.columns)
            data.insert(pos,f'MAX_AVG_LOAD_SHIFT_{i}',data['MAX_AVG_LOAD'])
            data[f'MAX_AVG_LOAD_SHIFT_{i}'] = data[f'MAX_AVG_LOAD_SHIFT_{i}'].shift(periods = i)
            pos+=1
        data = data.dropna()

        data = data.reset_index()
        data = data.drop('index',axis=1, inplace=False)

        return data
	
    
class ARIMA_Dataset:
    def __init__(self, path, Features, Mobility=False, is_val=False):
        self.path = path
        self.is_val = is_val
        self.KeepColumn = Features[Features["combined_pvalue"]<0.05]['feature'].tolist()
    
        if Mobility:
            self.mobility_col = ['residential_percent_change_from_baseline',
                            'workplaces_percent_change_from_baseline',
                            'transit_stations_percent_change_from_baseline',
                            'parks_percent_change_from_baseline',
                            'grocery_and_pharmacy_percent_change_from_baseline',
                            'retail_and_recreation_percent_change_from_baseline']
            self.KeepColumn = self.KeepColumn + self.mobility_col
        
        
     
    
     
    def LoadData(self):
        data = pd.read_csv(self.path + "data/clean_data.csv")
        if self.is_val:
            return data['MAX_AVG_LOAD'][-160:].values
        train = data['MAX_AVG_LOAD'][:-160].values
        x_test = data['MAX_AVG_LOAD'][-160:].values
        exog=np.array(data[self.KeepColumn].values)
        exog_test = exog[len(train):]
        exog_train = exog[:len(train)]
        x_train = [x for x in train]
    
        return x_train, x_test, exog_train, exog_test