# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:37:22 2020

@author: Anaji
"""

import pandas as pd 
import numpy as np

from sklearn.linear_model import LinearRegression
import pickle
import yaml

from preprocessor import PreprocessData

class PredictSalary:
    """
    Module for Create Model and prediction logic 
    """
    def __init__(self):
        with open('config/config.yml','r') as fl:
            self.config = yaml.load(fl, Loader=yaml.FullLoader)
        
    def loadCSV(self,filePath):
        """
        Loading CSV file
        Input:
            filepath
        Output:
            df = Dataframe
        """
        df= pd.read_csv(filePath)
        return df
    
    def preprocess(self,data):
        """
        Preprocess data by PreprocessData()
        Input:
            data = dataframe
        Output:
            preprocess_data = cleaned dataframe
        """
        preprocessObj = PreprocessData()
        preprocess_data = preprocessObj.preprocess1(data)
        return preprocess_data
    
    def dataSplit(self,df):
        """
        Dataframe split Independent and dependent features
        Input:
            df = dataframe
        Output:
            X = Independent feature as message
            y = Dependent feature as label
        """
        X = df.iloc[:, :3]
        y = df.iloc[:, -1]
        return X, y
        
    def linearReg(self, X, y, filename1):
        #Since we have a very small dataset, 
        #we will train our model with all availabe data.
        regressor = LinearRegression()
        #Fitting model with trainig data
        regressor.fit(X, y)
        # Saving model to disk
        pickle.dump(regressor, open(filename1, 'wb'))
        
    def model(self):
        """
        Process from prepocess to model creation  
        """
        filePath1 = self.config['model_data1']['train_data']
        data = self.loadCSV(filePath1)
        cleandata = self.preprocess(data)
        X, y = self.dataSplit(cleandata)
        filepath2 = self.config['model_pkl_1']['model_path']
        self.linearReg(X, y, filepath2)
        
    def loadpklfile(self, filePath):
        """
        Loading pkl file
        Input:
            filePath1 : filePath pkl file
        Output:
            regmodel : regressor model
        """
        # Loading regressor model
        regmodel=pickle.load(open(filePath,'rb'))
        return regmodel
    
    def predictsal(self, data):
        """
        Predict predictSalary
        Input:
            data : list of data value 
        Output:
            my_pred : prediction in numerical format
        """
        model = self.loadpklfile(self.config['model_pkl_1']['model_path'])
        my_pred = model.predict(data)
        return my_pred
        
# Create model by using train data and save pkl file
# PredictSalaryObj = PredictSalary()
# PredictSalaryObj.model()
# data = [[2, 9, 6]]
# result = PredictSalaryObj.predictsal(data)
# print(result)
