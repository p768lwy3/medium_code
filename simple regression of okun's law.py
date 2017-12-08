# Import Libraries for read and visualize the data
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# and do the Simple Regression
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Read and preprocess with the data
okun = pd.read_excel('okun.xls')
okun['%change_gnp'] = okun['gnp'].pct_change() * 100
okun['%change_un'] = okun.un - okun.un.shift(1)
okun = okun.dropna()
okun.head()

# Visualize the data
okun.plot.scatter('%change_un', '%change_gnp', c='b')
plt.title('%change_un/%change_gnp')
plt.show()

# sklearn linear regression
# build and fit the model
linreg = LinReg(fit_intercept=True, normalize=False, copy_X=True, n_jobs=-1)
linreg.fit(okun['%change_un'].values.reshape(-1, 1), okun['%change_gnp'].values.reshape(-1, 1))
# print the parameters
print('coeff: {0}; intercept: {1}'.format(str(linreg.coef_[0,0]), str(linreg.intercept_[0])))
# print the R2 score in 2 ways
r2 = linreg.score(okun['%change_un'].values.reshape(-1, 1), okun['%change_gnp'].values.reshape(-1, 1))
print('r2 by .score: ', r2)
predicted_gnp = linreg.predict(okun['%change_un'].values.reshape(-1, 1))
print("r2 score by metrics: %.6f"
      % r2_score(okun['%change_gnp'].values.reshape(-1, 1), predicted_gnp))
# print the mse
print("Mean squared error: %.6f"
      % mean_squared_error(okun['%change_gnp'].values.reshape(-1, 1), predicted_gnp))

# plot the fitted line with fitted model
plt.scatter(okun['%change_un'], okun['%change_gnp'], c='b')
plt.plot(okun['%change_un'].values.reshape(-1, 1), predicted_gnp, color='red', linewidth=3)
plt.title('linear regression of %change un with % change gnp')
plt.show()

# Error analysis
# compute errors
okun['predicted_gnp'] = 0.8502 + okun['%change_un'] * -1.8097
okun['error'] = okun['%change_gnp'] - okun['predicted_gnp']
okun.head(3)

# plot histogram of error and fitted normal curve
mu = np.mean(okun['error'])
std = np.std(okun['error'])
x = np.linspace(mu - 3*std, mu + 3*std, 100)
plt.plot(x, mlab.normpdf(x, mu, std))
plt.hist(okun['error'], bins=20, normed=1)
plt.show()

# QQ plot
norm = stats.probplot(okun['error'], dist="norm", plot=plt)
plt.show()

# simple regression with other two libraries
# scipy
slope, intercept, r_value, p_value, std_err = linregress(okun['%change_un'].values, okun['%change_gnp'].values)
print("r-squared:", r_value**2)
print("slope: ", slope, "intercept: ", intercept)

# statsmodels
X = sm.add_constant(okun['%change_un'].values)
y = okun['%change_gnp'].values
model = sm.OLS(y, X)
results = model.fit()
print('Parameters: ', results.params)
print('R2: ', results.rsquared)
print(results.summary())
# plot with statsmodels
plt.plot(okun['%change_un'].values, y, 'o', label='original data')
plt.plot(okun['%change_un'].values, results.params[0] + results.params[1] * okun['%change_un'].values, 'r', label='fitted line')
plt.legend()
plt.show()
