import pandas as pd
import numpy as np

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,90,np.nan],"English_marks":[70,80,90,np.nan,60],"Science_marks":[80,90,np.nan,60,70],"History_marks":[90,np.nan,60,70,80],"Geography_marks":[np.nan,60,70,80,90]}

df1=pd.DataFrame(dict1)

df1

df1.isnull().sum()

df1.info()

bool_series = pd.notnull(df1)
bool_series

# filling missing value using fillna() 
df1.fillna(0,inplace=True)
df1

df1=pd.DataFrame(dict1)
df1

# filling a missing value with
# previous ones 
dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,60],"Science_marks":[80,91,np.nan,60,70],"History_marks":[76,np.nan,60,70,80],"Geography_marks":[np.nan,60,70,80,90]}
df1=pd.DataFrame(dict1)

df1.fillna(method ='pad')

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# filling a missing value with
# next ones
df1.fillna(method ='bfill')

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# filling a null values using fillna()
df1["Maths_marks"].fillna(45, inplace = True)
df1["English_marks"].fillna(55, inplace = True)
df1["Science_marks"].fillna(65, inplace = True)
df1["History_marks"].fillna(75, inplace = True)
df1["Geography_marks"].fillna(85, inplace = True)

df1

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# filling a null values using fillna()
df1["Maths_marks"].fillna(int(df1["Maths_marks"].mean()), inplace=True)

df1

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# will replace  Nan value in dataframe with value 85 
df1.replace(to_replace = np.nan, value = 85)

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# to interpolate the missing values
df1.interpolate(method ='linear', limit_direction ='forward')
#c=(a+b)/2

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,75],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# using dropna() function 
df1.dropna()

dict1={"Roll_No":[1,2,3,4,5],"Name":["Amol","Dipak","Shreya","Krisha","Pooja"],"Maths_marks":[60,70,80,93,np.nan],"English_marks":[70,80,40,np.nan,50],"Science_marks":[80,91,np.nan,65,70],"History_marks":[76,np.nan,68,70,80],"Geography_marks":[np.nan,62,70,80,90]}
df1=pd.DataFrame(dict1)
df1

# using dropna() function    
df1.dropna(axis = 1)

#*****************

#outliers

# Importing
import sklearn
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
 
# Load the dataset
bos_hou = load_boston()
 
# Create the dataframe
column_name = bos_hou.feature_names
df_boston = pd.DataFrame(bos_hou.data)
df_boston.columns = column_name
df_boston.head()

''' Detection '''
# IQR
Q1 = np.percentile(df_boston['DIS'], 25, interpolation = 'midpoint')

Q3 = np.percentile(df_boston['DIS'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df_boston.shape)

# Upper bound
upper = np.where(df_boston['DIS'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df_boston['DIS'] <= (Q1-1.5*IQR))

''' Removing the Outliers '''
df_boston.drop(upper[0], inplace = True)
df_boston.drop(lower[0], inplace = True)

print("New Shape: ", df_boston.shape)

# *************

# data transformations

# importing pandas as pd
import pandas as pd

# Creating the DataFrame
df = pd.DataFrame({"A":[12, 4, 5, None, 1],
				"B":[7, 2, 54, 3, None],
				"C":[20, 16, 11, 3, 8],
				"D":[14, 3, None, 2, 6]})

# Create the index
index_ = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5']

# Set the index
df.index = index_

# Print the DataFrame
print(df)

# pass a list of functions
result = df.transform(func = ['sqrt', 'exp'])
result
# Print the result
#print(result)

# SCALING

import seaborn as sns
import pandas as pd
import numpy as np

data = sns.load_dataset('iris')
print('Original Dataset')
data.head()

# Min-Max Normalization
df = data.drop('species', axis=1)
df_norm = (df-df.min())/(df.max()-df.min())
df_norm = pd.concat((df_norm, data.species), 1)

print("Scaled Dataset Using Pandas")
df_norm.head()

# find skewness in each row
df.skew(axis = 1, skipna = True)

# Distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
  
# Calculating mean and standard deviation
mean = statistics.mean(df_norm["sepal_length"])
sd = statistics.stdev(df_norm["sepal_length"])
  
plt.plot(df_norm["sepal_length"], norm.pdf(df_norm["sepal_length"], mean, sd))
plt.show()
print(mean)
print(sd)






