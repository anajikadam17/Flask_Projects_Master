# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:40:42 2020

@author: Anaji
"""
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template,url_for,request
from flask_cors import cross_origin
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# =============================================================================
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
# =============================================================================
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
clf = pickle.load(open('nlp_model.pkl', 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
flight_price_rf = pickle.load(open("flight_price_rf.pkl", "rb"))

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
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('task1.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route("/index")
def index():
  return render_template("index.html")

@app.route("/task2")
def task2():
  return render_template("task2.html")
@app.route('/predict2',methods=['POST'])
def predict2():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('task2.html',prediction = my_prediction)

@app.route("/task3")
def task3():
  return render_template("task3.html")
@app.route('/predict3', methods = ["GET", "POST"])
@cross_origin()
def predict3():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # Total Stops
        Total_stops = int(request.form["stops"])

        # Airline
        # AIR ASIA = 0 (not in column) 
        airline=request.form['airline']
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
        Source = request.form["Source"]
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
        Source = request.form["Destination"]
        d_Cochin = 0
        d_Delhi = 0
        d_New_Delhi = 0
        d_Hyderabad = 0
        d_Kolkata = 0
        if (Source == 'Cochin'):
            d_Cochin = 1
        elif (Source == 'Delhi'):
            d_Delhi = 1
        elif (Source == 'New_Delhi'):
            d_New_Delhi = 1
        elif (Source == 'Hyderabad'):
            d_Hyderabad = 1
        elif (Source == 'Kolkata'):
            d_Kolkata = 1
        else:
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        
        prediction=flight_price_rf.predict([[Total_stops, Journey_day,
            Journey_month, Dep_hour, Dep_min, Arrival_hour, Arrival_min,
            dur_hour, dur_min, Air_India, GoAir, IndiGo, Jet_Airways,
            Jet_Airways_Business, Multiple_carriers, Multiple_carriers_Premium_economy,
            SpiceJet, Trujet, Vistara, Vistara_Premium_economy, s_Chennai,
            s_Delhi, s_Kolkata, s_Mumbai, d_Cochin, d_Delhi, d_Hyderabad,
            d_Kolkata, d_New_Delhi ]])
        output=round(prediction[0],2)
        return render_template('task3.html',prediction_text="Your Flight price is Rs. {}".format(output))


    return render_template("task3.html")
if __name__ == "__main__":
    app.run(debug=True)