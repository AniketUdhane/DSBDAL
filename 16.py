import pandas as pd
import numpy as np

df = pd.read_csv('datasets/16. covid_vaccine_statewise.csv')

df.head()

df.describe()

df.info()

df.isnull().sum()

first_dose = df.groupby('State')[['First Dose Administered']].sum()

first_dose

no_of_males = df.groupby('Male (Doses Administered)').sum()

no_of_males

