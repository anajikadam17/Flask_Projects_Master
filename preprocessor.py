# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:37:23 2020

@author: Anaji
"""

import pandas as pd

class PreprocessData:
    """
    Module for preprocessing data
    """
    def preprocess1(self,dataset):
        """
        Preprocess dataframe for model_1
        Input:
            dataset = dataframe
        Output:
            dataset = cleaned dataframe
        """
        # Experience column missing value fill by 0
        dataset['experience'].fillna(0, inplace=True)
        # test_score column missing  value fill by mean value of that column
        dataset['test_score(out of 10)'].fillna(
                                    dataset['test_score(out of 10)'].mean(),
                                    inplace=True)
        # function for Converting words to integer values
        def convert_to_int(word):
            """
            function convert word to number
            Input:
                word = word for number
            Output:
                number = number for inputed word
            """
            word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
                         'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10,
                         'eleven':11, 'twelve':12, 'zero':0, 0: 0}
            return word_dict[word]
        dataset['experience'] = dataset['experience'].apply(
                                    lambda x : convert_to_int(x))
        return dataset
    
    def preprocess2(self,df):
        """
        Preprocess dataframe for model_2
        Input:
            df = dataframe
        Output:
            df = cleaned dataframe
        """
        # Drop unusefull columns 
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],
                axis=1, inplace=True)
        # Column class map with for ham = 0 and spam = 1
        df['label'] = df['class'].map(
                                {'ham': 0, 'spam': 1})
        return df
    
    def preprocess3(self, df):
        """
        Preprocess dataframe for model_3
        Input:
            df = dataframe
        Output:
            data = cleaned dataframe
        """
        # Drop missing value record
        df.dropna(inplace = True)
        # Date_of_Journey
        df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day
        df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
        df.drop(["Date_of_Journey"], axis = 1, inplace = True)
        # Dep_Time
        df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
        df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
        df.drop(["Dep_Time"], axis = 1, inplace = True)
        # Arrival_Time
        df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
        df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute
        df.drop(["Arrival_Time"], axis = 1, inplace = True)
        # Duration
        duration = list(df["Duration"])
        for i in range(len(duration)):
            if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
                if "h" in duration[i]:
                    duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                else:
                    duration[i] = "0h " + duration[i]           # Adds 0 hour
        duration_hours = []
        duration_mins = []
        for i in range(len(duration)):
            duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
            duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
        # Adding Duration column to test set
        df["Duration_hours"] = duration_hours
        df["Duration_mins"] = duration_mins
        df.drop(["Duration"], axis = 1, inplace = True)
        Airline = pd.get_dummies(df["Airline"], drop_first= True)
        Source = pd.get_dummies(df["Source"], drop_first= True)
        Destination = pd.get_dummies(df["Destination"], drop_first = True)
        # Route and Total_Stops are related to each other
        df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
        # Replacing Total_Stops
        df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
        # Concatenate dataframe --> test_data + Airline + Source + Destination
        df.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
        data = pd.concat([df, Airline, Source, Destination], axis = 1)
        return data