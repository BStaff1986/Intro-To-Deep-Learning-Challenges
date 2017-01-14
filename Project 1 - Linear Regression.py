import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LinearRegression

df= pd.read_csv('challenge_dataset.txt', header=None,
                names=['Brain','Body'])

# Designate x and y and reshape from (97,) to (97,1)
x = df['Brain'].reshape(len(df['Brain']),1)
y = df['Body'].reshape(len(df['Body']),1)
    
# Initialize LinearRegressio and fit data            
lr = LinearRegression()
lr.fit(x,y)

# Prepare plot and draw line of best fit
plt.scatter(x,y)
plt.plot(x, lr.predict(x))
plt.title('Linear Regression with Line of Best Fit')
plt.xlabel('X Variable') # I'm assuming the units are grams
plt.ylabel('Y Variable')
plt.xlim(4,23)
plt.show()


def mean_squared_error(x, y):
    """
    Returns the mean squared error
    """
    squared_errors = np.array([])
    for y_hat, y in zip(lr.predict(x), y):
        squared_errors = np.append(squared_errors, ((y_hat - y) ** 2))
        return squared_errors.mean()     
        
# Print output
print('Prediction for 5.5277: ' + str(round(lr.predict(5.5277)[0][0],4)))
print('Actual value for 6.1101: ' + str(y[1][0]))
print('\n')        
mse = mean_squared_error(x,y)        
print('Mean Squared Error: ' + str(round(mse, 2)))
print('Root Mean Squared Error: ' + str(round(sqrt(mse),2)))
