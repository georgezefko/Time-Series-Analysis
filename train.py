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



class train:


    def __init__(self, datapath, assetspath,siteid):

        super().__init__()
        self.datapath = datapath
        self.assetspath = assetspath
        self.siteid= siteid

    def training(self):
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logger = logging.getLogger(__name__)
        logger.info("Training a classifier for anomaly classification")
        print('Start training')

        self.models_selection(self.datapath, self.assetspath,self.siteid)

        print('Done training')


    def training_dataset(self,dataset):
        train_test_split_ratio = 0.2
        seed = 42
        
        X = dataset.drop(dataset.loc[:,dataset.columns[dataset.columns.str.contains('Status')]],axis=1).values
        y = dataset.loc[:,dataset.columns[dataset.columns.str.contains('Status')]].astype(int).values

        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=train_test_split_ratio, random_state = seed,stratify=y)
        return X_train, X_valid,y_train, y_valid
    
    def scaling(self,dataset,asset,siteid):

        X_train, X_valid, y_train, y_valid = self.training_dataset(dataset)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        assets = asset[:3] 


        joblib.dump(scaler,f'Models/Scaler_{siteid}_{assets}.pkl')

        return X_train, X_valid, y_train, y_valid

    def models_selection(self,datapath,assetspath,siteid):

        dataset = joblib.load(datapath)
        assets = pd.read_csv(assetspath,index_col=0)
        assets = assets.iloc[:,0].tolist()
        print(assets)
        best_model = []

        for i,j in enumerate(assets):


            X_train, X_valid, y_train, y_valid = self.scaling(dataset[i],j,siteid)

            train_models =['SVM','RF','ANN','KNN']
            models_performance=[]

            results =pd.DataFrame(columns=['Algorithm','F1_mean'])

            for i in train_models:
                if i=='SVM':
                    SVM = svm.SVC(kernel='linear')
                    SVM.fit(X_train,y_train.ravel())
                    SVM_predictions = SVM.predict(X_valid)
                    models_performance.append(('SVM',SVM_predictions))

                elif i=='RF':
                    RF = RandomForestClassifier(n_estimators=120,criterion='entropy',max_depth=100) #best parameters
                    RF.fit(X_train,y_train.ravel())
                    RF_predictions = RF.predict(X_valid)
                    models_performance.append(('RF',RF_predictions))

                elif i=='ANN':
                    ANN = MLPClassifier(hidden_layer_sizes= 50, learning_rate_init= 0.01, solver ='adam')           
                    ANN.fit(X_train,y_train.ravel())
                    ANN_predictions = ANN.predict(X_valid)
                    models_performance.append(('ANN',ANN_predictions))
                else:
                    KNN = KNeighborsClassifier(algorithm ='auto', leaf_size = 20, n_neighbors= 5)
                    KNN.fit(X_train,y_train.ravel())
                    KNN_predictions = KNN.predict(X_valid)
                    models_performance.append(('KNN',KNN_predictions))
            
            i=0
            for name, preds in models_performance:

                f1_scores = f1_score(y_valid,preds, average=None).mean()
                results.loc[i]=[name,f1_scores]
                i+=1

            logging.info(results)
            model = results['Algorithm'].loc[results['F1_mean']== max(results['F1_mean'])]
            logging.info(f'The best model{model}')
            best_model.append(model.iloc[0])
        
        final_dataframe = pd.DataFrame({'Models': best_model})
        final_model= pd.DataFrame(final_dataframe.Models.value_counts()).reset_index()
        logging.info(f'The best model trained{final_model.iloc[0,0]}')

        if model.iloc[0,0]=='SVM':

            joblib.dump(SVM, f'Models/model_{siteid}_{j[:3]}.pkl') 

        elif model.iloc[0,0]== 'RF':

            joblib.dump(RF, f'Models/model_{siteid}_{j[:3]}.pkl')

        elif model.iloc[0,0]=='ANN':

            joblib.dump(ANN, f'Models/model_{siteid}_{j[:3]}.pkl')

        else:

            joblib.dump(KNN, f'Models/model_{siteid}_{j[:3]}.pkl')

        



