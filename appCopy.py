# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:36:28 2020

@author: Anaji
"""
from flask import Flask, request, jsonify, render_template,url_for,request
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model_1 import PredictSalary
from model_2 import SpamPrediction
from model_3 import FlightFarePredict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/task1")
def task1():
    return render_template("task1.html")

@app.route('/predict1',methods=['POST'])
def predict1():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    PredictSalaryObj = PredictSalary()
    my_prediction = PredictSalaryObj.predictsal(final_features)
    output = round(my_prediction[0], 2)
    return render_template('task1.html', 
                           prediction_text='Employee Salary should be $ {}'
                           .format(output))

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/task2")
def task2():
    return render_template("task2.html")
@app.route('/predict2',methods=['POST'])
def predict2():
    """
        Predict text spam or ham
    """
    if request.method == 'POST':
        message = request.form['message']
        message = [message]
        SpamPredictionObj = SpamPrediction()
        my_prediction = SpamPredictionObj.predictSpam(message)
    return render_template('task2.html',prediction = my_prediction)

@app.route("/task3")
def task3():
    return render_template("task3.html")
@app.route('/predict3', methods = ["GET", "POST"])
@cross_origin()
def predict3():
    if request.method == "POST":
        date_dep = request.form["Dep_Time"]
        date_arr = request.form["Arrival_Time"]
        Total_stops = int(request.form["stops"])
        airline = request.form['airline']
        Source = request.form["Source"]
        Dest = request.form["Destination"]
        FlightFarePredictObj = FlightFarePredict()
        my_prediction = FlightFarePredictObj.predictFlightFare(date_dep, 
                                            date_arr, Total_stops, airline,
                                            Source, Dest)
        output=round(my_prediction[0],2)
        return render_template('task3.html', 
                           prediction_text="Your Flight price is Rs. {}"
                           .format(output))
    return render_template("task3.html")

if __name__ == "__main__":
    app.run(debug=True)