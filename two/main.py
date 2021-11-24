from matplotlib.markers import MarkerStyle
import pandas as pd
from matplotlib import pyplot


df = pd.read_csv('two\weight-height.csv')

df['Height'] = df['Height']*2.54
df['Weight'] = df['Weight']/2.205


female_df = df[df['Gender'] == 'Female']
male_df = df[df['Gender'] == 'Male']

pyplot.scatter(female_df['Height'], female_df['Weight'], c='pink', marker=".")
pyplot.show()
pyplot.scatter(male_df['Height'], male_df['Weight'], c='blue', marker=".")
pyplot.show()


pyplot.scatter(female_df['Height'], female_df['Weight'], c='pink', marker=".")
pyplot.scatter(male_df['Height'], male_df['Weight'], c='blue', marker=".")
pyplot.show()

print(female_df.head())