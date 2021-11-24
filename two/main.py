import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df = pd.read_csv('two\weight-height.csv')

df['Height'] = df['Height']*2.54
df['Weight'] = df['Weight']/2.205


female_df = df[df['Gender'] == 'Female']
male_df = df[df['Gender'] == 'Male']


# pyplot.scatter(female_df['Height'], female_df['Weight'], c='pink', marker=".")
# pyplot.show()
# pyplot.scatter(male_df['Height'], male_df['Weight'], c='blue', marker=".")
# pyplot.show()


# pyplot.scatter(female_df['Height'], female_df['Weight'], c='pink', marker=".")
# pyplot.scatter(male_df['Height'], male_df['Weight'], c='blue', marker=".")
# pyplot.show()

Xf = female_df[['Height']]
yf = female_df['Weight']

Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size = 0.3)

lin_reg_f = LinearRegression()
lin_reg_f.fit(Xf_train, yf_train)
yf_pred = lin_reg_f.predict(Xf_test)

Xm = male_df[['Height']]
ym = male_df['Weight']

Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size = 0.3)

lin_reg_m = LinearRegression()
lin_reg_m.fit(Xm_train, ym_train)
ym_pred = lin_reg_m.predict(Xm_test)

print(female_df.head())