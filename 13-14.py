import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/5-8-13-14.iris.csv')

df.head()

df.describe()

df.dtypes

sns.histplot(x=df['sepal.length'])

fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].set_title('sepal.length')
axes[0,0].hist(df['sepal.length'],bins=7)

axes[0,1].set_title('sepal.width')
axes[0,1].hist(df['sepal.width'],bins=7)

axes[1,0].set_title('petal.length')
axes[1,0].hist(df['petal.length'],bins=7)

axes[1,1].set_title('petal.width')
axes[1,1].hist(df['petal.width'],bins=7)

def graph(y):
    sns.boxplot(x="variety", y=y, data=df)

plt.figure(figsize=(10,10))
plt.subplot(221)

graph('sepal.length')
plt.subplot(222)

graph('sepal.width')
plt.subplot(223)

graph('petal.length')
plt.subplot(224)

graph('petal.width')
plt.show()

sns.boxplot(x='sepal.width', data=df)

Q1 = np.percentile(df['sepal.width'], 25, interpolation = 'midpoint')

Q3 = np.percentile(df['sepal.width'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
print("Old Shape: ", df.shape)
upper = np.where(df['sepal.width'] >= (Q3+1.5*IQR))
lower = np.where(df['sepal.width'] <= (Q1-1.5*IQR))

df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)

print("New Shape: ", df.shape)

sns.boxplot(x='sepal.width', data=df)

