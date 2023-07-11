import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("FuelConsumptionCo2.csv")
print(data.describe())
"""
histo = data[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
histo.hist()
plt.figure()
plt.scatter(data['ENGINESIZE'],data['CO2EMISSIONS'])
"""

X = np.asanyarray(data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB_MPG']])
y = np.asanyarray(data[['CO2EMISSIONS']])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

plt.figure()


reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
print ("Correlation Coefficient: ", reg.coef_)
print ("Bias: %.2f" %reg.intercept_)
plt.plot(X_train, reg.coef_*X_train + reg.intercept_, '-r')
y_predict=reg.predict(X_test)

print("Variance Score: ",reg.score(X_test,y_test) )
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predict- y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predict- y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test, y_predict) )
print("predicted emission for engine size 3: ", y_predict[[3,4,20]])