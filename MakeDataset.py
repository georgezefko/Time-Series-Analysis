import pandas as pd
import numpy as np
import calendar
import seaborn as sns
import joblib
import pandas.api.types as ptypes
from meteostat import Point, Daily,Hourly
from geopy.geocoders import Nominatim
import time
from pprint import pprint
from datetime import datetime
import myutils
import os



class MakeDataset:

    """A class that handles everything related to getting
    & setting up the training & test datasets
    """


    def __init__(self, siteid,missing_thres=0.5,group='freezer',
                datapath='Data\Processed\HACCP.csv',
                temppath = 'Data\Processed\Temp.csv',
                censorspath = 'Data\Processed\S3S4.csv',
                sitepath = 'Data\Raw\Dinosol_Site_Information.csv'):
        
        super().__init__()
        self.group = group
        self.missing_thres =missing_thres
        self.siteid=siteid
        self.data= datapath
        self.temppath = temppath
        self.check = censorspath
        self.site = sitepath
    
    def make_dataset(self):
        print('Start creating the dataset')

        self.features(self.data,self.temppath,
                        self.check,self.siteid,
                        self.missing_thres, self.group)

        print('Dataset created')

    def haccp(self,datapath,siteid):
        
        data = pd.read_csv(datapath)
        data = data.loc[data['PropertyName']=='HACCP']
        data['Timetag'] = pd.to_datetime(data['Timetag'])
        data['DataValue'] = data['DataValue'].apply(lambda x: float(x.split()[0].replace(',', '.')))

        assert ptypes.is_datetime64_any_dtype(data['Timetag'])
        assert ptypes.is_numeric_dtype(data['DataValue'])

        data = data[data['SiteID']==siteid]
        print('haccp',data.shape)

        return data

    def temp(self,temppath,siteid):


        temp = pd.read_csv(temppath)

        temp = temp[temp['SiteID']== siteid].pivot_table(values='ParameterValue',index='AssetName',columns='PropertyName')

        conditions = [
                            (temp['LOWTEMP'] == -21), (temp['LOWTEMP'] ==-3 ),
                            (temp['LOWTEMP'] == 1), (temp['LOWTEMP'] == 0),
                            (temp['LOWTEMP'] == 6)
                        
                            ]

            # create a list of the values we want to assign for each condition
        values = ['freezer', 'fish_meat', 'fruits', 'milk_charcuterie','charniceria']

        # create a new column & use np.select to assign values to it using our lists as arguments
        temp['Group'] = np.select(conditions, values)
            
        temp.to_csv(f'Data\Processed\{siteid}_temp_groups.csv')
        return temp

    def censors(self,censorspath,siteid):

        sensors = pd.read_csv(censorspath)
        test_check = sensors[(sensors['PropertyName']=='GLOBAL_ID') & (sensors['SiteID']== siteid)].sort_values(ascending=True,by='Timetag')

        test_check = test_check[['Timetag','AssetName','PointName','DataValue']]
        test_check['Timetag'] = pd.to_datetime(test_check['Timetag'])

        test_check['DataValue'] = test_check['DataValue'].apply(lambda x: float(x.split()[0].replace(',', '.')))

        #test_check = test_check.groupby(['Timetag','PointName'])['DataValue'].mean().unstack('PointName')
        return test_check


    
    def rearrange(self, datapath,siteid,missing_thres):

        supermarket =self.haccp(datapath,siteid)
        supermarket.head()

        df_new = supermarket.groupby(['AssetID','AssetName','Timetag'])['DataValue'].mean()

        #Convert the pandas series back to dataframe
        df_new = pd.DataFrame(df_new).reset_index()

        #Transpose the data so we get the timetag on the rows (might have to change later on)
        df_new = df_new.pivot_table('DataValue','Timetag','AssetName').reset_index()
        df_new.head()

        #print('The shape of the dataframe is:',df_new.shape)
        print('Shape before cleaning {}'.format(df_new.shape))
        df_new = myutils.missing_values.drop_columns(df_new,missing_thres)
        print('Shape after cleaning {}'.format(df_new.shape))
        df_new = myutils.missing_values.fill_nans(df_new)


        df_clean =(df_new.set_index(["Timetag"])
         .stack()
         .reset_index(name='DataValue')
         .rename(columns={'level_2':'AssetName'}))

        df_clean.to_csv(f'Data\Processed\{siteid}_clean.csv')

        return df_clean
    
    def merge_data(self,datapath,temppath,censorspath,siteid,missing_thres,group):


        data = self.rearrange(datapath,siteid,missing_thres)
        temp =self.temp(temppath,siteid)
        test_check = self.censors(censorspath,siteid)
        data['Timetag'] = pd.to_datetime(data['Timetag'])
        data = data[['Timetag','AssetName','DataValue']]

       

        data_merge = pd.merge(data, temp, left_on = 'AssetName', right_index=True)

        assert len(data_merge)>0

        conditions = [
                        (data_merge['DataValue'] < data_merge['LOWTEMP']),
                        (data_merge['DataValue'] > data_merge['HIGHTEMP']), 
                        (data_merge['DataValue'] >= data_merge['LOWTEMP'])
                        & (data_merge['DataValue'] <= data_merge['HIGHTEMP'])
                        ]
        values = [-1, 1, 0]
        data_merge['Defrost'] = np.select(conditions, values)
        print(data_merge[data_merge['Group']=='milk/charcuterie'].head(20))

        #Choose only relevant columns (Check whether the same AssetID are across the stores)
        selected = data_merge[['AssetName','DataValue','Timetag','Defrost']].loc[data_merge['Group']== group].sort_values(ascending=True,by='Timetag')#.set_index('Timetag')

        assert len(selected)>0

        check_merge=pd.merge(test_check, temp, left_on = 'AssetName', right_index=True)
        check_freezer = check_merge[check_merge['Group']== group]
        check_freezer = check_freezer[['Timetag','AssetName','PointName','DataValue']]

        assert len(check_freezer)>0


        return selected,check_freezer,data_merge
    


    def annotation(self,datapath,temppath,censorspath,siteid,missing_thres,group):

        selected,check_freezer,data_merge = self.merge_data(datapath,temppath,censorspath,siteid,missing_thres,group)

        assets = [i for i in selected.AssetName.unique()]

        def classes(df):
            if test_check.shape[1] >= 3:
                
                if (df['Status_'+i] =='Anomaly_Above') & (df['S4S3_State_'+i] ==1) & (df['open/close'] == 0):
                    classes = 'anomaly_critical_close_above'

                elif (df['Status_'+i] =='Anomaly_Above') & (df['S4S3_State_'+i] ==1) & (df['open/close'] == 1):
                    classes ='anomaly_critical_open_above'

                elif (df['Status_'+i] =='Anomaly_Above') & (df['S4S3_State_'+i] ==0 )& (df['open/close'] == 0):
                    classes ='anomaly_close_above'

                elif (df['Status_'+i] =='Anomaly_Above') & (df['S4S3_State_'+i] ==0) & (df['open/close'] == 1):
                    classes ='anomaly_open_above'

                elif (df['Status_'+i] =='Anomaly_Below') & (df['S4S3_State_'+i] ==1) & (df['open/close'] == 0):
                    classes ='anomaly_critical_close_below'

                elif (df['Status_'+i] =='Anomaly_Below') & (df['S4S3_State_'+i] ==1) & (df['open/close'] == 1):
                    classes ='anomaly_critical_open_below'

                elif (df['Status_'+i] =='Anomaly_Below') & (df['S4S3_State_'+i] ==0) & (df['open/close'] == 0):
                    classes ='anomaly_close_below'

                elif (df['Status_'+i] =='Anomaly_Below') & (df['S4S3_State_'+i] ==0) & (df['open/close'] == 1):
                    classes = 'anomaly_open_below'
                
                else:
                    classes = df['Status_'+i]
                    
            else:

                if (df['Status_'+i] =='Anomaly_Above') & (df['open/close'] == 0):
                    classes ='anomaly_close_above'

                elif (df['Status_'+i] =='Anomaly_Above') & (df['open/close'] == 1):
                    classes ='anomaly_open_above'

                elif (df['Status_'+i] =='Anomaly_Below') & (df['open/close'] == 0):
                    classes ='anomaly_close_below'

                elif (df['Status_'+i] =='Anomaly_Below') & (df['open/close'] == 1):
                    classes ='anomaly_open_below'

                else:
                    classes = df['Status_'+i]

            return classes 

        empty = []

        for i in assets:
            
            df = data_merge[data_merge['AssetName']== i]
            defrost = df[['Timetag','Defrost']]
            point = df[['Timetag','AssetName','DataValue']]
            point = point.pivot_table('DataValue','Timetag','AssetName')
            point = myutils.missing_values.drop_columns(point,0.5)
            point = myutils.missing_values.fill_nans(point)

            test_merge = pd.merge(point,defrost, right_on='Timetag',left_index=True)#.set_index('Timetag')
            test_merge['hour'] = pd.to_datetime(test_merge['Timetag'].astype(str)).dt.hour
            test_merge['open/close'] = test_merge['Timetag'].dt.time.astype(str).apply(lambda x: 0 if x >= '22:00:00' or x <'07:00:00' else 1) #for class labels
            test_merge = test_merge.set_index('Timetag')
            test_merge['Consecutive'+ i] = test_merge['Defrost'] * (test_merge['Defrost'].groupby((test_merge['Defrost'] != test_merge['Defrost'].shift()).cumsum()).cumcount() + 1)
            test_merge['Status_'+i] = test_merge['Consecutive'+i].apply(lambda x: 'Defrost' if x>0 and x<=2 else ('Anomaly_Below' if x < 0 else ('Anomaly_Above' if x >2 else 'Normal') ))

            #get points that have identified as defrost but they should be anomalies
            defrost_begin = test_merge.loc[test_merge['Consecutive'+i]==1]
            # calculate the difference between the beginning of defrosts
            defrost_begin['diff'] =defrost_begin['hour'].diff().abs()
            # if difference is below 4 hours then it can be considered anomaly
            defrost_begin['Status_'+i].loc[defrost_begin['diff']<4]='Anomaly_Above' 
            #update the database with the new labels
            test_merge.drop('hour',axis=1,inplace=True)
            test_merge.update(defrost_begin)
            
            #add s3 & S4
            test_check = check_freezer[check_freezer['AssetName']==i].sort_values(ascending=True,by='Timetag')

            test_check= test_check[['Timetag','PointName','DataValue']]
            test_check = test_check.groupby(['Timetag','PointName'])['DataValue'].mean().unstack('PointName')
            
         
            if test_check.shape[1] >= 2: #increase number if other parameter than S3 & S4 is added
                test_check['S4S3diff_'+i] = test_check.iloc[:,-1].abs()-test_check.iloc[:,-2].abs()
                test_check['S4S3_State_'+i] = test_check['S4S3diff_'+i].apply(lambda x: 0 if x<5 else 1) #for new class labels
                #print(test_check.head())
                
            


            #full dataset per asset
            check_freezer_merge = pd.merge(test_merge,test_check,right_on='Timetag',left_index=True)
            #print('shape_before',check_freezer_merge.shape)
            check_freezer_merge['Status_'+i]=check_freezer_merge.apply(classes,axis=1) #new class implementation
            check_freezer_merge.drop('Defrost',axis=1,inplace=True)
            check_freezer_merge = myutils.missing_values.drop_columns(check_freezer_merge,0.1)
            check_freezer_merge = myutils.missing_values.fill_nans(check_freezer_merge)
            #print('shape_after',check_freezer_merge.shape)

            
            #get the plots per asset
            #myutils.plot_anomalies.anomalies_plots(check_freezer,i)

            #append dataframes to a final one
            empty.append(check_freezer_merge)


        final =pd.concat(empty,axis=1)
        final.drop(['open/close'],axis=1,inplace=True)
        #generaly this dataset should be used
        #final.to_csv('Data\Processed\Final.csv')

        #for the sake of project spliot the data to test and train here
        final.to_csv(f'Data\Processed\{siteid}_{group}_Final.csv')
        

        assets_export = pd.DataFrame(assets)
        assets_export.to_csv(f'Data\Processed\{siteid}_{group}_assets.csv')

        return empty,assets
    
    def class_assign(self,data):


        classes = {'Normal':0,
            'Defrost':1,
            'anomaly_critical_open_below' :2,  
            'anomaly_critical_close_below':3, 
            'anomaly_open_below':4,               
            'anomaly_close_below':5,    
            'anomaly_critical_open_above':6,     
            'anomaly_critical_close_above':7,                  
            'anomaly_open_above':8,               
            'anomaly_close_above':9
                 }
        for i in range(len(data)):
            data[i].drop(['open/close'],axis=1,inplace=True)
            status = [j for j in data[i].columns if j.startswith('Status')]
            data[i][status[0]] = data[i][status[0]].replace(classes)
            counts=data[i][status[0]].value_counts()
            data[i] = data[i][~data[i][status[0]].isin(counts[counts < 2].index)] #remove classes with 1 incident
        #status = [i for i in data.columns if i.startswith('Status')]

        # for i in status:
        #     data[i] = data[i].replace(classes)
        # data = myutils.missing_values.fill_nans(data)
        #print('clean data',data.head())
        print('Classes labeled')

        return data
    
    def features(self,datapath,siteid,temppath,censorspath,missing_thres,group):
        pd.options.mode.chained_assignment = None
        #check why the set_index doesnt work in the makedataset.py
        final,assets = self.annotation(datapath,siteid,temppath,censorspath,missing_thres,group)
        # assets = pd.read_csv(assetspath) # input assets
        # assets.drop('Unnamed: 0',axis=1,inplace=True) #find better way to import assets
        # assets = assets.values.tolist()
        
        data = self.class_assign(final)
        #data.set_index('Timetag',inplace=True) #this shoudl be removed once the issed with set index is solved
        test = []
        train = []
        for i, j in enumerate(assets):
            dataset = data[i].loc[:,data[i].columns[data[i].columns.str.contains(assets[i])]] # this need to generalize

            dataset = dataset.reset_index()
            dataset['Timetag'] = pd.to_datetime(dataset['Timetag'])#same with the issue


            #Applying vectorization to get numerical date features
            dataset['year'] = (dataset['Timetag']).dt.year
            dataset['weekday'] = (dataset['Timetag']).dt.weekday  # Define what the numbers of the weekday are
            dataset['month'] = (dataset['Timetag']).dt.month
            dataset['weekofyear'] =(dataset['Timetag']).dt.weekofyear
            dataset['quarter'] =(dataset['Timetag']).dt.quarter
            dataset['hour'] =(dataset['Timetag']).dt.hour
            dataset['minute'] =(dataset['Timetag']).dt.minute
            dataset = dataset.assign(Hour_of_week = lambda x: x.Timetag.map(lambda y: y.dayofweek)*24+x.hour)
            dataset['open/close'] = dataset['Timetag'].dt.time.astype(str).apply(lambda x: 0 if x >= '22:00:00' or x <'07:00:00' else 1)
            #dataset= dataset.set_index('Timetag')
            #print('dataset after',dataset.head())

            #final train
            final_train = dataset[dataset['Timetag']<='2021-07-31']
            final_train = final_train.set_index('Timetag')
            
            #final test
            final_test= dataset[dataset['Timetag']>='2021-08-01']
            final_test = final_test.set_index('Timetag')

            test.append(final_test)
            train.append(final_train)

        # final_train.to_csv(f'Data\Processed\{siteid}_Final_train.csv',index_col=0)
        # final_test.to_csv(f'Data\Processed\{siteid}_Final_test.csv',index_col=0)
        print('Saving train and test sets')
        joblib.dump(test, f'{siteid}_{group}_Test.pkl')
        joblib.dump(train, f'{siteid}_{group}_Train.pkl')

    
        print('finish processing data')







