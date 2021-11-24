import pandas as pd

df = pd.read_csv('two\weight-height.csv')

df['Height'] = df['Height']*2.54
df['Weight'] = df['Weight']/2.205


female_df = df[df['Gender'] == 'Female']
male_df = df[df['Gender'] == 'Male']

print(female_df.head())