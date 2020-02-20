import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
data=pd.read_csv('E:\kc_house_data.csv')
x=np.array(data['sqft_living'])
y=np.array(data['price'])
xtrain,xtest,ytrain,ytest=train_test_split(x,y)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly=PolynomialFeatures()
xpoly=poly.fit_transform(xtrain.reshape(-1,1))
modelpoly=LinearRegression()
modelpoly.fit(xpoly,ytrain)

from sklearn.metrics import mean_squared_error
testpoly=poly.fit_transform(xtest.reshape(-1,1))
polypred=modelpoly.predict(testpoly)
mse=mean_squared_error(ytest,polypred)
rmse=np.sqrt(mse)
