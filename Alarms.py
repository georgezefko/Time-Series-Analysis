import pandas as pd
import numpy as np
import seaborn as sns
import joblib


def alarms(id,group,dataset,temp_group,assets,type='--- High Temp.'):
    """[summary]

    Parameters
    ----------
    id : [type]
        [description]
    group : [type]
        [description]
    dataset : [type]
        [description]
    assets : [type]
        [description]
    type : str, optional
        [description], by default '--- High Temp.'
    """

    alarms = pd.read_csv('Data\Processed\Alarms.csv')
    global_id = pd.read_csv('Data\Processed\Globalid.csv')
    temp =pd.read_csv(temp_group)
    dataset = pd.read_csv(dataset,index_col=0)
    assets= pd.read_csv(assets,index_col=0)
    assets= assets.iloc[:,0].to_list()
    

    global_id['ParameterValue'] = global_id['ParameterValue'].apply(lambda x: float(x.split()[0].replace(',', '.')))
    global_id = global_id.groupby(['CategoryId','AssetName'])['ParameterValue'].mean()
    global_id = pd.DataFrame(global_id,columns={'ParameterValue'}).reset_index()
    global_id['CategoryId']=[i.split('.p')[0] for i in global_id['CategoryId']]

    merge = pd.merge(global_id,alarms,left_on='CategoryId',right_on='description')
    merge['reportedtime'] = pd.to_datetime(merge['reportedtime'])
    merge['receivedtime'] = pd.to_datetime(merge['receivedtime'])

    merge_temp= pd.merge(merge,temp,right_on='AssetName',left_on='AssetName')

    mergeall = pd.DataFrame(merge_temp[(merge_temp['Group']==group)&(merge_temp['SiteID']==id)].groupby(['reportedtime','ReasonText','AssetName'])['Total_Alarm_Count'].count().unstack('AssetName').fillna(0)).reset_index()

    if type =='':
        alarms= mergeall[(mergeall['reportedtime']>='2020-09-01')]
    else:
        alarms= mergeall[(mergeall['reportedtime']>='2020-09-01')&(mergeall['ReasonText']==type)]

    test_assets = set(assets).difference(set(alarms.columns[2:]))

    print('Assets before',assets)
    print("Difference of first and second String: " + str(test_assets))
    if not test_assets:
        print("First and Second list are Equal")
    # if lists are not equal    
    else:
        print("First and Second list are Not Equal")
        assets = [x for x in alarms.columns[2:] if x in assets]
    print('Assets after',assets)

    empty_alarms=[]
    for i in assets:
        
        dummie_alarms = alarms[[ 'ReasonText','reportedtime', i]]
        dummie_alarms['reportedtime'] =pd.to_datetime(dummie_alarms['reportedtime'])
        
        dummie_final = dataset[i].reset_index()
        dummie_final['Timetag'] =pd.to_datetime(dummie_final['Timetag'])
        test = pd.merge_asof(dummie_alarms,dummie_final,right_on='Timetag',left_on='reportedtime',allow_exact_matches=True,direction='nearest')
        test = test.loc[test[i+'_x']>=1]
        test_alarms = test[['Timetag',i+'_y']].set_index('Timetag')
        #test_alarms = test[['ReasonText','Timetag',i+'_y']].set_index('Timetag')

        empty_alarms.append(test_alarms)
    
    joblib.dump(empty_alarms,f'Data/Processed/{id}_{group}_empty_alarms.pkl')
    
    

    
    

