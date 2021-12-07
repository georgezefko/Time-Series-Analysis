import pandas as pd
import calendar
import seaborn as sns
from geopy.geocoders import Nominatim
import time
import math
import matplotlib.pyplot as plt



def get_location_by_address(address):
        """This function returns a location as raw from an address
            will repeat until success"""
        app = Nominatim(user_agent="Getloc")
        time.sleep(1)
        try:
            return Nominatim(user_agent="Getloc").geocode(address).raw
        except:
            return get_location_by_address(address)


class missing_values:
    
    
    def check_nans(df):
        round((df.isnull().sum()/df.count()[0])*100,2).plot(kind='barh')

    def drop_columns(df,threshold):
        limitPer = len(df)* threshold
        df = df.dropna(thresh=limitPer, axis=1)
        return df

        
    def fill_nans(df,how='mean'):
        nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df),columns=['percent'])
        idx = nans['percent']>0

        nan_columns = [i for i in nans[idx].index.values]
        for i in nan_columns:
            if how=='mean':
                df[i].fillna(df[i].mean(), inplace=True)
            else:
                df[i].fillna(df[i].mode(), inplace=True)
        return df

class location:

    def get_location_by_address(address,app):
        """This function returns a location as raw from an address
            will repeat until success"""
        time.sleep(1)
        try:
            return app.geocode(address).raw
        except:
            return get_location_by_address(address)

class time_expand:
   

    #Eplore the distribution of the records throughout time
# split the data to querters

    def time(df,time):
        """ The function takes pandas dataframe timtag column 
        and adds extra time columns (eg. hour, day , month)

        Parameters
        ----------
        df : [Pandas dataframe]
        
        Returns
        -------
        [Pandas dataframe]
        """
        
        Specification = ['Timetag','Date']
        Specification = [i for i in Specification if i == time]

        #add data and time
        df['Date'] = df[Specification[0]].dt.date
        df['Time'] = df[Specification[0]].dt.time

        #Add months in order
        df['Month'] = df[Specification[0]].dt.month
        df['Month'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
        df['Month'] = df['Month'].astype('category')
        df['Month'].cat.reorder_categories(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],inplace=True,ordered=True)

        # Add hour
        df['Hour'] = df[Specification[0]].dt.hour

        #add weekday
        df['Weekday'] = df[Specification[0]].dt.day_name()
        df['Weekday'] = df['Weekday'].astype('category')
        df['Weekday'].cat.reorder_categories(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True,ordered=True)

        #hours in a week
        df = df.assign(Hour_of_week = lambda x: x.Timetag.map(lambda y: y.dayofweek)*24+x.Hour)

        return df

class plots:

    def distribution_plots(df,time):

        points = [i for i in df.AssetName.unique()]
        ncols = 5
        nrows = int(len(points)/ncols)+1

        f,ax = plt.subplots(nrows,ncols,sharex=True)
        f.set_figheight(30)
        f.set_figwidth(40)
        

        for i , point in enumerate(points):
            axis = ax[i//ncols,i%ncols]
            if time=='Month':
                measure = df.loc[df['AssetName']==point].Month.value_counts()
            elif time=='Day':
                measure = df.loc[df['AssetName']==point].Weekday.value_counts()
            measure.sort_index(inplace=True)
            axis.title.set_text("Point: {}".format(point))
            axis.bar(measure.keys(),height=measure)
            axis.set_xticklabels(measure.keys(),rotation=20)
                    
        #axis[0,0].title.set_text
        f.tight_layout()
    
    
    def statistic_plots(df,temp,style,orientation,time):
        """
        Functions that retyrns subplots for every data point
        in three kind of plots

        Parameters
        ----------
        df : [Pandas dataframe]
            
        ncols : [integer]
            The number of columns for the subplots grid
        nrows : [integer]
            The number of rows for the subplots grid
        style : [string]
            The plot style. Can be :
            line,hist,box
        orientation : [string]
            The orientation of temperature limits. Can be :
            vertical or horizontal
        time : [String]
            Defines the corresponding time interval. Can be:
            Hour, Hour_of_week, Weekday, Month
        """
        timespan = ['Hour','Hour_of_week','Weekday','Month','Date']
        interval = [i for i in timespan if i == time]
        #points = [i for i in df1.AssetName.unique()]
        points = [i for i in df.columns[1:-6]]
        ncols=8
        nrows=math.ceil(len(points)/ncols)


        f,ax = plt.subplots(ncols,nrows,sharex=False)
        f.set_figheight(40)
        f.set_figwidth(40)

        

        
        for i , point in enumerate(points):
            axis = ax[i//nrows, i%nrows]
            if style == 'box':
                graph = sns.boxplot(x= df[interval[0]],y=df[points[i]],ax=axis)

            elif style=='hist' : 
                graph = sns.histplot(df[points[i]],kde=True,ax=axis)

            elif style=='line':
                graph = sns.lineplot(x= df[interval[0]],y=df[points[i]],ax=axis)
                pass

            for k, _ in enumerate(temp.index.values):

                if point == temp.index.values[k] and orientation == 'vertical':

                    graph.axvline(temp.LOWTEMP[k],color='Red')
                    graph.axvline(temp.HIGHTEMP[k],color='Red')

                elif point == temp.index.values[k] and orientation == 'horizontal':

                    graph.axhline(temp.LOWTEMP[k],color='Red')
                    graph.axhline(temp.HIGHTEMP[k],color='Red')

        graph.set(xlabel=None)
        
        f.tight_layout()

class plot_anomalies:

    def anomalies_plots(check_freezer_merge,i):
            #plot per asset
        COLOR_TEMPERATURE = "#69b3a2"
        COLOR_CONSECUTIVE = "#3399e6"

        fig, ax1 = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [4, 2]},figsize=(20, 5))
        ax2 = ax1[0].twinx()
        #print(len(check_freezer_merge.columns))
        if len(check_freezer_merge.columns)==8:
            ax1[0].plot(check_freezer_merge.index.values[0:10*144],check_freezer_merge.iloc[0:10*144]['Consecutive'+i], color=COLOR_TEMPERATURE, lw=3,label ='Consequtive points')
            ax2.plot(check_freezer_merge.index.values[0:10*144],
                            check_freezer_merge.iloc[0:10*144].iloc[:,0], color=COLOR_CONSECUTIVE, lw=4,label='Ther')
            # ax2.plot(check_freezer_merge.index.values[0:1*144],
            #                 check_freezer_merge.iloc[0:1*144].iloc[:,1], color='Red', lw=4,label='Cut out')
            ax2.plot(check_freezer_merge.index.values[0:10*144],
                            check_freezer_merge.iloc[0:10*144].iloc[:,4], color='Green', lw=4, label='S3')
            ax2.plot(check_freezer_merge.index.values[0:10*144],
                            check_freezer_merge.iloc[0:10*144].iloc[:,5], color='Yellow', lw=4,label='S4')
            
            #measure =check_freezer_merge.iloc[:,2].value_counts()
            measure = check_freezer_merge.groupby(check_freezer_merge.iloc[:,3])['open/close'].value_counts().unstack().fillna(0)
            #axis.title.set_text("Point: {}".format(point))
            ax1[1].bar(measure.index,measure[0],label='close')
            ax1[1].bar(measure.index,measure[1],bottom=measure[0], label ='open')
            ax1[1].legend()
            ax1[1].set_xticklabels(measure.index,rotation=20)

            #ax1[1].bar(measure.keys(),height=measure)
            #ax1[1].set_xticklabels(measure.keys(),rotation=20)
                

        else:
            ax1[0].plot(check_freezer_merge.index.values[0:10*144],check_freezer_merge.iloc[0:10*144]['Consecutive'+i], color=COLOR_TEMPERATURE, lw=3,label='Consequtive points')
            ax2.plot(check_freezer_merge.index.values[0:10*144],
                            check_freezer_merge.iloc[0:10*144].iloc[:,0], color=COLOR_CONSECUTIVE, lw=4,label='Ther')
            # ax2.plot(check_freezer_merge.index.values[0:1*144],
            #                 check_freezer_merge.iloc[0:1*144].iloc[:,3], color='Red', lw=4,label='Cut out')
            # ax2.plot(check_freezer_merge.index.values[0:1*144],
            #                 check_freezer_merge.iloc[0:1*144].iloc[:,4], color='Green', lw=4, label ='S3')
            #measure =check_freezer_merge.iloc[:,2].value_counts()

            measure = check_freezer_merge.groupby(check_freezer_merge.iloc[:,3])['open/close'].value_counts().unstack().fillna(0)
            #axis.title.set_text("Point: {}".format(point))
            ax1[1].bar(measure.index,measure[0],label='close')
            ax1[1].bar(measure.index,measure[1],bottom=measure[0], label ='open')
            ax1[1].legend()
            ax1[1].set_xticklabels(measure.index,rotation=20)
            
                
        
            #ax1[1].bar(measure.keys(),height=measure)
            #ax1[1].set_xticklabels(measure.keys(),rotation=20)
        



        ax1[0].set_xlabel("Time")
        ax1[0].set_ylabel("Defrost Consecutive points", color=COLOR_TEMPERATURE, fontsize=14)
        ax1[0].tick_params(axis="y", labelcolor=COLOR_TEMPERATURE)
        ax1[0].legend(loc='upper right')

        ax2.set_ylabel("Temperature", color=COLOR_CONSECUTIVE, fontsize=14)
        ax2.tick_params(axis="y", labelcolor=COLOR_CONSECUTIVE)
        ax2.legend(loc='upper left')

        fig.suptitle("Temperature "+ i, fontsize=20)

        
        
        fig.autofmt_xdate()
        fig.tight_layout()
        


