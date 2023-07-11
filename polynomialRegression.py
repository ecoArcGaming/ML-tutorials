import matplotlib.pyplot as plt
import pandas as pd
import sys

import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
np.set_printoptions(threshold=2000)


data = pd.read_csv("FuelConsumptionCo2.csv")


scatterPlot = plt.scatter(data.ENGINESIZE, data.CO2EMISSIONS)
X = data.ENGINESIZE
y = data.CO2EMISSIONS
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pol = PolynomialFeatures(degree=3)
X_train_transformed = pol.fit_transform([X_train])
print(X_train)
print(X_train_transformed)

y_test = pol.fit(X_train_transformed,y_train)
print("coefficients: ", pol.coef_)
print ('Intercept: ',pol.intercept_)