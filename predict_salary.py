import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("Salary_Data.csv")

df['Gender'] = df['Gender'].map({'Male': 1,
                                 'Female': 2})
df['Education Level'] = df['Education Level'].map({'Bachelor\'s': 1,
                                 'Master\'s': 2,
                                 'PhD': 3})

df = df.dropna()

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

x = df[['Age', 'Gender', 'Education Level', 'Years of Experience']]
y = df['Salary']

x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.35, random_state = 0)

'''
lm = LinearRegression()
lm.fit(x_train, y_train)
lm.score(x_test, y_test)

predictions = lm.predict(x_test)

print("Enter Employee Details to Predict Salary")
a = int(input("Age: "))
b = int(input("Gender: "))
c = int(input("Education Level: "))
d = int(input("Years Of Experience: "))
features = np.array([[a, b, c, d]])
print("Predicted Salary = ", int(lm.predict(features)))
'''

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [100, 200 ,300],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5,
                           scoring = "neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(x_train, y_train)
best_forest = grid_search.best_estimator_

print("Enter Employee Details to Predict Salary")
a = int(input("Age: "))
b = int(input("Gender: "))
c = int(input("Education Level: "))
d = int(input("Years Of Experience: "))
features = np.array([[a, b, c, d]])
print("Predicted Salary = ", int(best_forest.predict(features)))

pickle.dump(best_forest, open('Salary.pkl','wb'))