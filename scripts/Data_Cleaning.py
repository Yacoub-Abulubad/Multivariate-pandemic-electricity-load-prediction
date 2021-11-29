import pandas as pd
import numpy as np


class Data_Cleaner:
    def __init__(self, path, save_file=True, shift_by=[0], processing="WEATHER"):
        self.path = path
        self.save_file = save_file
        self.shift_by = shift_by
        self.processing = processing
    
        if processing != "WEATHER" and processing != "SHIFTED":
            raise AttributeError(f"The process \"{processing}\" for the following data is not a valid processing!\nPlease use either \"WEATHER\" or \"SHIFTED\".")
            return 0
        
        try:
            self.data = pd.read_csv(self.path + "data/clean_data.csv")
            self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]
            print("\n\nWe have loaded the clean data!")
        except:
            self.data = self.LoadAndClean(self.path, self.save_file)

        if processing == "SHIFTED":
            if len(shift_by) != 1 or shift_by[0] != 0:
                self.ShiftRows(self.data['MAX_AVG_LOAD'], self.shift_by, self.save_file)
            else:
                raise ValueError("There must be atleast one number in \"shift_by\" list parameter.")
                return 0



    def ShiftRows(self, data, shiftBy, save_file):
        data = pd.DataFrame(data)
        pos=0
        for i in shiftBy:
          data.insert(pos+1,f'MAX_AVG_LOAD_SHIFT_{i}',data['MAX_AVG_LOAD'])
          data[f'MAX_AVG_LOAD_SHIFT_{i}'] = data[f'MAX_AVG_LOAD_SHIFT_{i}'].shift(periods = i)
          pos+=1
        data = data.dropna()
        try:
            data.drop('index', axis=1, inplace=True)
        except:
            pass
        data = data.reset_index()
        if save_file:
            data.to_csv(self.path + "data/shifted_data.csv")
            print(f"\n\nShifted data is saved in location:\n{self.path}data/shifted_data.csv")


    def week_id(self,data):
        print("\n\nWe have added a \"Week ID\" feature!")
        week=0
        data['week_id'] = np.nan
        for i in range(len(data)):
          if week == 53:
            week=1
          if i%7 == 0:
            week=week+1
          data['week_id'][i] = week

        return data


    def LoadAndClean(self,path, save_file):
        print("\n\nWe are loading and cleaning up the data!")
        load_path = path + "data/raw/peakloads 2017-2020 new.xlsx"
        mobility_path = path + "data/raw/2020_JO_Region_Mobility_Report.csv"
        mobility_df = pd.read_csv(mobility_path)
        mobility_df = mobility_df[mobility_df['place_id']=='ChIJmd5kZkdvABURmU4mUQdbKI0']
        mobility_col = ['residential_percent_change_from_baseline',
                        'workplaces_percent_change_from_baseline',
                        'transit_stations_percent_change_from_baseline',
                        'parks_percent_change_from_baseline',
                        'grocery_and_pharmacy_percent_change_from_baseline',
                        'retail_and_recreation_percent_change_from_baseline',
                        'date']
        mobility_df = mobility_df.filter(mobility_col)   
        mobility_df[mobility_col[:-1]] = 0.01 * mobility_df[mobility_col[:-1]]    
        mobility_df['date'] = pd.to_datetime(mobility_df['date'])
        raw_load_df = pd.read_excel(load_path)
        clean_load_df = raw_load_df.dropna(subset=['DAY','date','MAX_MORN_LOAD','MAX_EVN_LOAD'])
        clean_load_df.reset_index(drop=True,inplace=True)
        clean_load_df.insert(2,'MAX_AVG_LOAD', np.average(clean_load_df[['MAX_MORN_LOAD','MAX_EVN_LOAD']], axis=1))
        clean_load_df = clean_load_df.drop(['MAX_EVN_LOAD','MAX_MORN_LOAD'],axis=1)

        weather_path = path + "data/raw/daily weather  data new.xlsx"
        weather_df = pd.read_excel(weather_path)
        weather_df = weather_df[:1461]
        
        weather_df.insert(2,'MAX_AVG_LOAD',clean_load_df['MAX_AVG_LOAD'])

        clean_df = weather_df[['date','MAX_AVG_LOAD','maxtempC','mintempC','avgtempC','totalprecipMM','weatherDesc','humidity','cloudcover','HeatIndexC','WindChillC','FeelsLikeC','windspeedKmph','winddirdegree']]
        clean_df = self.week_id(clean_df)
        clean_df['Weekday'] = clean_df['date'].dt.weekday
        clean_df['Weekday'] = clean_df['date'].dt.weekday
        clean_df['is_weekend'] = np.nan
        clean_df['Weekday'] = clean_df['Weekday'].replace((0,1,2,3,4,5,6), ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))

        for i in range(len(clean_df)):
          if clean_df['Weekday'][i] == 'Friday' or clean_df['Weekday'][i] == 'Saturday':
            clean_df['is_weekend'][i] = 1
          else:
            clean_df['is_weekend'][i] = 0


        clean_df = pd.merge(clean_df, mobility_df,on='date', how='outer')
        clean_df[mobility_col[:-1]] = clean_df[mobility_col[:-1]].fillna(0)
        if save_file:
            clean_df = clean_df.loc[:, ~clean_df.columns.str.contains('^Unnamed')]
            clean_df.to_csv(path + "data/clean_data.csv")
            print(f"Cleaned data is saved in location:\n{self.path}data/clean_data.csv")
        
        return clean_df
        