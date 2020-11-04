# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:32:42 2020

@author: Anaji
"""

import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import yaml

from preprocessor import PreprocessData

class SpamPrediction:
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
        df= pd.read_csv(filePath, encoding="latin-1")
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
        preprocess_data = preprocessObj.preprocess2(data)
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
        X = df['message']
        y = df['label']
        return X, y
        
    def CountVect(self, X, filename):
        """
        CountVectorizer for feature X 
        Input:
            X = dataframe
        Output:
            df = cleaned dataframe
        """
        cv = CountVectorizer()   # Extract Feature With CountVectorizer
        X = cv.fit_transform(X)  # Fit the Data
        pickle.dump(cv, open(filename, 'wb'))  # Save CountVectorizer model
        return X
    
    def TrainTestSplit(self,X, y):
        """
        Split Dataframe into train and test
        Input:
            X, y = Independent and dependent features
        Output:
            X_train, X_test, y_train, y_test : splited train and test dataframe
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.33,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test
    
    def MultinomialNB(self, X_train, X_test,
                      y_train, y_test, filename):
        """
        Create Model MultinomialNB 
        Input:
            X_train, X_test, y_train, y_test : train and test dataframe
            filename = filename for dump pickle
        Output:
            Model save pickle file format at cache by filename
        """
        mnb = MultinomialNB()
        mnb.fit(X_train,y_train)
        mnb.score(X_test,y_test)
        pickle.dump(mnb, open(filename, 'wb'))  # Save MultinomialNB model
    
    def model(self):
        """
        Process from prepocess to model creation  
        """
        filePath = self.config['model_data2']['train_data']
        data = self.loadCSV(filePath)
        cleandata = self.preprocess(data)
        X, y = self.dataSplit(cleandata)
        X = self.CountVect(X, self.config['model_pkl_2_1']['model_path'])
        X_train, X_test, y_train, y_test = self.TrainTestSplit(X, y)
        self.MultinomialNB(X_train, X_test, y_train, y_test, self.config['model_pkl_2_2']['model_path'])
        
    def loadpklfile(self, filePath1, filePath2):
        """
        Loading pkl file
        Input:
            filePath1 : for first pkl file
            filePath2 : for second pkl file
        Output:
            cv, mnb : both model
        """
        cv=pickle.load(open(filePath1,'rb'))
        mnb = pickle.load(open(filePath2, 'rb'))
        return cv, mnb
        
    def predictSpam(self, text):
        """
        Predict spam or ham based text
        Input:
            text : text from user 
        Output:
            my_pred : prediction in binary 0 or 1
        """
        cv, mnb = self.loadpklfile(self.config['model_pkl_2_1']['model_path'], 
                                   self.config['model_pkl_2_2']['model_path'])
        vect = cv.transform(text).toarray()
        my_pred = mnb.predict(vect)
        return my_pred
        
# =============================================================================
# SpamPredictionObj = SpamPrediction()
# SpamPredictionObj.model()
# print("="*50)
# text1 = "You won 1000"
# data1 = [text1]
# result1 = SpamPredictionObj.predictSpam(data1)
# print(result1)
# print("="*50)
# text2 = "Hello how are you"
# data2 = [text2]
# result1 = SpamPredictionObj.predictSpam(data2)
# print(result1)
# =============================================================================
