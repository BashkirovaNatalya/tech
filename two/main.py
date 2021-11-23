import pandas as pd

df = pd.read_csv('two\weight-height.csv')

female_df = df[df['Gender'] == 'Female']
male_df = df[df['Gender'] == 'Male']

