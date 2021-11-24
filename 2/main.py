import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score,  mean_squared_error

df = pd.read_csv('two\weight-height.csv')

df['Height'] = df['Height']*2.54
df['Weight'] = df['Weight']/2.205


female_df = df[df['Gender'] == 'Female']
male_df = df[df['Gender'] == 'Male']

pyplot.subplot(211)
pyplot.scatter(female_df['Height'], female_df['Weight'], c='pink', marker=".")
pyplot.subplot(212)
pyplot.scatter(male_df['Height'], male_df['Weight'], c='blue', marker=".")

Xf = female_df[['Height']]
yf = female_df['Weight']

Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size = 0.3)

lin_reg_f = LinearRegression()
lin_reg_f.fit(Xf_train, yf_train)
yf_pred = lin_reg_f.predict(Xf_test)

m, b = np.polyfit(Xf_test['Height'], yf_pred, 1)
pyplot.subplot(211)
pyplot.plot(Xf_test['Height'], m*Xf_test['Height'] + b, c='red')


Xm = male_df[['Height']]
ym = male_df['Weight']

Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size = 0.3)

lin_reg_m = LinearRegression()
lin_reg_m.fit(Xm_train, ym_train)
ym_pred = lin_reg_m.predict(Xm_test)

m, b = np.polyfit(Xm_test['Height'], ym_pred, 1)
pyplot.subplot(212)
pyplot.plot(Xm_test['Height'], m*Xm_test['Height'] + b, c='black')
pyplot.show()

r2f = r2_score(yf_test, yf_pred)
r2m = r2_score(ym_test, ym_pred)

msef = mean_squared_error(yf_test, yf_pred)
msem = mean_squared_error(ym_test, ym_pred)

print("R2 score for females: ", r2f)
print("R2 score for males: ", r2m)
print("Mean squared error for females: ", msef)
print("Mean square error for males: ", msem)