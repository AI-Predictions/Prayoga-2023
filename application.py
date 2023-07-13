from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def models():
    return render_template("models.html")

@app.route('/predict/Car', methods=['GET', 'POST'])
def car():
    
    if request.method == "POST":
        
        clf = joblib.load("model.pkl")

        power = request.form.get("power")
        year = request.form.get("year")
        kms_driven = request.form.get("kms_driven")
        mileage = request.form.get("mileage")

        X = pd.DataFrame([[power, year, kms_driven, mileage]])
        print(int(clf.predict(X)[0]))

        prediction = int(clf.predict(X)[0])
        
    else:
        prediction = ""
        
    return render_template("CarPricePredict.html", output = prediction)

@app.route('/predict/House', methods=['GET', 'POST'])
def house():
    
    if request.method == "POST":
        
        clf = joblib.load("HouseDf.pkl")
        
        a = int(request.form.get("Avg. Area Income"))
        b = int(request.form.get("Avg. Area House Age"))
        c = int(request.form.get("Avg. Area Number of Rooms"))
        d = int(request.form.get("Avg. Area Number of Bedrooms"))
        e = int(request.form.get("Area Population"))
        
        X = np.array([[a, b, c, d, e]])

        print(int(clf.predict(X)[0]))

        prediction = int(clf.predict(X))
        
    else:
        prediction = ""
        
    return render_template("HouseRentPredict.html", output = prediction)

@app.route('/predict/Salary', methods=['GET', 'POST'])
def salary():
    
    if request.method == "POST":
        
        clf = joblib.load("Salary.pkl")
        
        a = int(request.form.get("Age"))
        b = int(request.form.get("Gender"))
        c = int(request.form.get("Education Level"))
        d = int(request.form.get("Years Of Experience"))
        
        X = np.array([[a, b, c, d]])

        print(int(clf.predict(X)[0]))

        prediction = int(clf.predict(X))
        
    else:
        prediction = ""
        
    return render_template("SalaryPrediction.html", output = prediction)

if __name__ == '__main__':
    app.run(debug = True)