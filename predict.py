import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('train-data.csv')
data['car_company'] = data['Name'].apply(lambda x: x.split(" ")[0])
data = data.drop('Name', axis=1)
print(data)
  
# Select relevant features and target variable
features = ['Power','Year', 'Kilometers_Driven', 'Mileage']#'Location', 'Year', 'Kilometers_Driven','Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'car_company'
target = ['Price']

print(data)

X = data[features]
y = data[target]

categorical_features = ['Power','Year', 'Kilometers_Driven', 'Mileage']
preprocessor = ColumnTransformer([('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.3, random_state = 1)#104 = 76% 0.1 = 88% 0.3=80%

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)
pickle.dump(model, open('model.pkl','wb'))

# Make predictions on the training set
train_predictions = model.predict(X)

# Evaluate the model using mean squared error (MSE) on training set
train_mse = mean_squared_error(y, train_predictions)
print("Training MSE:", train_mse)

predictions = model.predict(X)

acc = (model.score(X, y)*100)
print("Accuracy : ",round(acc),"%")