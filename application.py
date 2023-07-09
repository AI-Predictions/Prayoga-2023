from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("model.pkl")
        
        # Get values through input bars
        power = request.form.get("power")
        year = request.form.get("year")
        kms_driven = request.form.get("kms_driven")
        mileage = request.form.get("mileage")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[power, year, kms_driven, mileage]])
        print(int(clf.predict(X)[0]))
        # Get prediction
        prediction = int(clf.predict(X)[0])
        
    else:
        prediction = ""
        
    return render_template("model.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)