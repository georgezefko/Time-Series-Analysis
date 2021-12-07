import logging
import pandas as pd
import myutils
import joblib
import os
#from feature_extract import TimeSeriesFeatureEngineering
#from preprocess import TimeSeriesPreprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn import svm
from sklearn.metrics import ( f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier



class test:

    def __init__(self,siteid, datapath='Data\Processed\89247_Final_test.csv',
                      assetspath='Data\Processed\89259_assets.csv',group='freezer'):

        super().__init__()
        
        
        self.datapath = datapath
        self.siteid = siteid
        self.assetspath = assetspath
        self.group = group


    def testing(self):
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logger = logging.getLogger(__name__)
        logger.info("Predictions from the classifier on test data")
        logger.info('Start testing')

        self.predictions(self.datapath,
                         self.siteid,
                         self.assetspath,
                         self.group)
        
        logger.info('Done testing')
    

    def predictions(self,datapath,siteid,assetspath,group):

        test = joblib.load(datapath)
        assets = pd.read_csv(assetspath,index_col=0)
        assets = assets.iloc[:,0].tolist()
        predictions = []

        for i,j in enumerate(assets):
            X_test = test[i].drop(test[i].loc[:,test[i].columns[test[i].columns.str.contains('Status')]],axis=1).values
            y_test = test[i].loc[:,test[i].columns[test[i].columns.str.contains('Status')]].astype(int).values
            
            #laod the models
            scaler = joblib.load(f'Models/Scaler_{siteid}_{j[:3]}.pkl')
            model = joblib.load(f'Models/model_{siteid}_{j[:3]}.pkl')

            X_test = scaler.transform(X_test)
            test_predictions = model.predict(X_test)
            logging.info(f'f1 score on test data {f1_score(y_test,test_predictions, average=None)}')

            preds = pd.DataFrame(test_predictions,columns=['predictions'])
            final_preds = preds.merge(test[i].reset_index(), right_index=True,left_index=True)
            predictions.append(final_preds)

        joblib.dump(predictions,f'Data/Processed/{siteid}_{group}_predcitions.pkl')

        #final_preds.to_csv(f'Data\Processed\{siteid}_predictions.csv')
            


        

        