import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import pickle

HouseDf = pd.read_csv("USA_Housing.csv")
HouseDf.head(5)

x = HouseDf[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = HouseDf['Price']

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test = tts(x, y, test_size = 0.40, random_state = 101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)
coeff_df = pd.DataFrame(lm.coef_, x.columns, columns = ["Coefficient"])

predictions = lm.predict(x_test)
pickle.dump(lm, open('HouseDf.pkl','wb'))

print("Enter House Details to Predict Rent")
a = int(input("Avg. Area Income: "))
b = int(input("Avg. Area House Age: "))
c = int(input("Avg. Area Number of Rooms: "))
d = int(input("Area Number of Bedrooms: "))
e = int(input("Area Population: "))
features = np.array([[a, b, c, d, e]])
print("Predicted House Price = ", int(lm.predict(features)))