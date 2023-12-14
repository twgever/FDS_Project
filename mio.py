#import libraries
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn import linear_model

linearReg = linear_model.LinearRegression()
linearReg.fit(X_train, y_train)
y_pred = linearReg.predict(X_test)

print("RMSE: %.2f" % metrics.mean_squared_error(y_test, y_pred))
print("R2: %.2f" % metrics.r2_score(y_test, y_pred))