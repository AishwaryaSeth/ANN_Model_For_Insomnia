# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:29:40 2018

@author: admin
"""
# example of making predictions for a regression problem
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# generate regression dataset
dataframe = pandas.read_csv("E:\College_Work\Sem 5\Paper_ANN\Insomnia_Probability.csv", sep=",", header=0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]


scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(Y.reshape(158,1))
X = scalarX.transform(X)
Y = scalarY.transform(Y.reshape(158,1))
X_train, X_test, y_train, y_test = train_test_split(X,Y)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=1000, verbose=0)
# new instances where we do not know the answer
W, b = model.layers[0].get_weights()
print('\n\nINPUT LAYER\n','Weights=', W, '\n\nBiases=', b)
W, b = model.layers[1].get_weights()
print('\n\nHIDDEN LAYER\n','Weights=', W, '\n\nBiases=', b)
W, b = model.layers[2].get_weights()
print('\n\nOUTPUT LAYER\n','Weights=', W, '\n\nBiases=', b)

ynew = model.predict(X_test)
print('MEAN SQUARED ERROR: ',mean_squared_error(y_test,ynew),'\n')
print('RMSE VALUE: ', (mean_squared_error(y_test,ynew)**0.5),'\n')

# show the inputs and predicted outputs
for i in range(len(X_test)):
	print("Actual=%s, Predicted=%s" % (y_test[i], ynew[i]))



np.savetxt('Results6.csv', np.column_stack([y_test, ynew]),header="Actual,Predicted",comments='',fmt='%1.4f', delimiter=',')
