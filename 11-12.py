import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("datasets/11-12.titanic.csv")

df.head()

df.info()
df.shape

df.isnull().sum()

sns.histplot(df['Fare'],bins=10,kde=True)

sns.barplot(x='Sex', y='Age', data=df, estimator=np.std)

