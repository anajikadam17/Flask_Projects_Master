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


if __name__ == "__main__":
    app.run(debug=True)