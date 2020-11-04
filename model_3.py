# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:54:43 2020

@author: Anaji
"""

import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import yaml

from preprocessor import PreprocessData

class FlightFarePredict:
    """
    Module for Create Model and prediction logic 
    """
    def __init__(self):
        with open('config/config.yml','r') as fl:
            self.config = yaml.load(fl, Loader=yaml.FullLoader)
        
    def loadExcel(self,filePath1, filePath2):
        """
        Loading excel file
        Input:
            filepath1, filepath1: excel file path for train and test
        Output:
            train: Dataframe
            train: Dataframe
        """
        train= pd.read_excel(filePath1)
        test = pd.read_excel(filePath1)
        return train, test
    
    def preprocess(self,traindata, testdata):
        """
        Preprocess data by PreprocessData()
        Input:
            traindata, testdata = train and test dataframe
        Output:
            preprocess_traindata, preprocess_testdata = cleaned dataframes
        """
        preprocessObj = PreprocessData()
        preprocess_traindata = preprocessObj.preprocess3(traindata)
        preprocess_testdata = preprocessObj.preprocess3(testdata)
        return preprocess_traindata, preprocess_testdata
    
    def dataSplit(self,df):
        """
        Dataframe split Independent and dependent features
        Input:
            df = dataframe
        Output:
            X = Independent feature as message
            y = Dependent feature as label
        """
        X = df.drop(["Price"], axis = 1)
        y = df.iloc[:, 1]
        return X, y
        
    def TrainTestSplit(self,X, y):
        """
        Split Dataframe into train and test
        Input:
            X, y = Independent and dependent features
        Output:
            X_train, X_test, y_train, y_test : splited train and test dataframe
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 0.2, 
                                                            random_state = 42)
        return X_train, X_test, y_train, y_test
    
    def RandomForestReg(self, X_train, X_test,
                      y_train, y_test, filename):
        """
        Create Model RandomForestRegressor 
        Input:
            X_train, X_test, y_train, y_test : train and test dataframe
            filename = filename for dump pickle
        Output:
            Model save pickle file format at cache by filename
        """
        rf_random = RandomForestRegressor(n_estimators= 700, 
                                          min_samples_split= 15, 
                                          min_samples_leaf = 1, 
                                          max_features= 'auto', 
                                          max_depth= 20)
        rf_random.fit(X_train, y_train)
        prediction = rf_random.predict(X_test)
        print('MAE:', metrics.mean_absolute_error(y_test, prediction))
        print('MSE:', metrics.mean_squared_error(y_test, prediction))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
        # save model on disk
        pickle.dump(rf_random, open(filename, 'wb'))
        
    def loadpklfile(self, filePath):
        """
        Loading pkl file
        Input:
            filePath : file path for pkl file
        Output:
            model : RandomForestReg model
        """
        model=pickle.load(open(filePath,'rb'))
        return model
        
    def predict(self, X_test, y_test):
        """
        Predict FlightFare
        Input:
            X_test, y_test : test data for prediction and r2score calculate
        Output:
            my_pred : prediction FlightFare in numerical format
        """
        rf_model = self.loadpklfile(self.config['model_pkl_3']['model_path'])
        my_pred = rf_model.predict(X_test)
        print('R2 Score',metrics.r2_score(y_test, my_pred))
    
    def predictFlightFare(self, date_dep, date_arr, Total_stop, airline, Source, Dest):
        """
        Predict FlightFare for data from html form
        Input:
            date_dep: Departure Date of Journey
            date_arr: Arrival Date
            Total_stop : number of stop
            airline : Airline name
            Source : Source name
            Dest : Destination name
        Output:
            prediction : FlightFare value in numerical format
        """
        # Date_of_Journey
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # Arrival
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # Total Stops
        Total_stops = Total_stop
        # Airline
        # AIR ASIA = 0 (not in column) 
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 
        if(airline=='Jet Airways'):
            Jet_Airways = 1
        elif (airline=='IndiGo'):
            IndiGo = 1
        elif (airline=='Air India'):
            Air_India = 1
        elif (airline=='Multiple carriers'):
            Multiple_carriers = 1
        elif (airline=='SpiceJet'):
            SpiceJet = 1
        elif (airline=='Vistara'):
            Vistara = 1
        elif (airline=='GoAir'):
            GoAir = 1
        elif (airline=='Multiple carriers Premium economy'):
            Multiple_carriers_Premium_economy = 1
        elif (airline=='Jet Airways Business'):
            Jet_Airways_Business = 1
        elif (airline=='Vistara Premium economy'):
            Vistara_Premium_economy = 1
        elif (airline=='Trujet'):
            Trujet = 1
        else:
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0
        # Source
        # Banglore = 0 (not in column)
        s_Delhi = 0
        s_Kolkata = 0
        s_Mumbai = 0
        s_Chennai = 0
        if (Source == 'Delhi'):
            s_Delhi = 1
        elif (Source == 'Kolkata'):
            s_Kolkata = 1
        elif (Source == 'Mumbai'):
            s_Mumbai = 1
        elif (Source == 'Chennai'):
            s_Chennai = 1
        else:
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0
        # Destination
        # Banglore = 0 (not in column)
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 0
        d_Kolkata = 0
        if (Dest == 'Cochin'):
            d_Cochin = 1
        elif (Dest == 'Delhi'):
            d_Delhi = 1
        elif (Dest == 'New_Delhi'):
            d_New_Delhi = 1
        elif (Dest == 'Hyderabad'):
            d_Hyderabad = 1
        elif (Dest == 'Kolkata'):
            d_Kolkata = 1
        else:
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        rf_model = self.loadpklfile(self.config['model_pkl_3']['model_path'])
        prediction=rf_model.predict([[Total_stops, Journey_day,
            Journey_month, Dep_hour, Dep_min, Arrival_hour, Arrival_min,
            dur_hour, dur_min, Air_India, GoAir, IndiGo, Jet_Airways,
            Jet_Airways_Business, Multiple_carriers, 
            Multiple_carriers_Premium_economy, SpiceJet, Trujet, Vistara, 
            Vistara_Premium_economy, s_Chennai, s_Delhi, s_Kolkata, s_Mumbai, 
            d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata, d_New_Delhi ]])
        return prediction
        
    def model(self):
        """
        Process from prepocess to model creation  
        """
        train, test = self.loadExcel(
                            self.config['model_data3_1']['train_data'], 
                            self.config['model_data3_2']['test_data'])
        train, test = self.preprocess(train, test)
        X, y = self.dataSplit(train)
        X_train, X_test, y_train, y_test = self.TrainTestSplit(X, y)
        self.RandomForestReg(X_train, X_test, y_train, y_test, 
                             self.config['model_pkl_3']['model_path'])
        self.predict(X_test, y_test)
        
# Create model by using train data and save pkl file
# =============================================================================
# FlightFarePredictObj = FlightFarePredict()
# FlightFarePredictObj.model()
# 
# =============================================================================
