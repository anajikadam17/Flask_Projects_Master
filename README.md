Flask_Projects
1. Predict Salary Analysis, based on Experience, test score and interview score.
2. Spam Detector For Email Messages.
3. FLIGHT PRICE PREDICTION, 
	based on Departure date, Arrival Date, Which Airline you want to travel?, Source
	Destination, Stopage.

bin=>
app.py : flask application file.
model_1.py : create model for Predict Salary Analysis.
model_2.py : create model for Spam Detector For Email Messages.
model_3.py : create model for FLIGHT PRICE PREDICTION.

bin=>
templates=>
index.html : html page for home page
task1.hrml : html page for first project Predict Salary Analysis.
task2.hrml : html page for second project Spam Detector For Email Messages.
task3.hrml : html page for third project FLIGHT PRICE PREDICTION.

cache folder and docs folder in .gitignore
create cache in Flask_Projects folder 
cache=>
pred_sal_model.pkl : pkl file for Predict Salary Analysis.
countvect.pkl : CountVectorizer model save in pkl file for Spam Detector For Email Messages.
NB_spam_model.pkl : pkl file for Spam Detector For Email Messages.
flight_price_model.pkl : pkl file for FLIGHT PRICE PREDICTION.

config=>
config.yml : configuration file for storing path

data=>
train=>
hiring.csv : csv data file for trainning for first project for Predict Salary Analysis.
spam.csv : csv data file for trainning for second project Spam Detector For Email Messages.
Data_Train.xlsx : excel file for training for third project Flight Price Prediction.

data=>
Test=>
Test_set.xlsx : excel file for testing for third project Flight Price Prediction.

Referance to www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig
