import pandas as pd
import numpy as np

df1=pd.read_csv("titanic.csv")

df1.info()

df1.isnull().sum()

df1.describe()

df1.head()

print("Median of age is: ", df1['Age'].mean())

# Python code to demonstrate the 
# working of median() function.
 
# importing statistics module
import statistics
 
# unsorted list of random integers
data1 = [2, -2, 3, 6, 9, 4, 5, -1]
 
 
# Printing median of the
# random data-set
print("Median of data-set is : % f "
        % (statistics.median(data1)))

df1.info()

df1.size

df1.shape

df1

# importing pandas as pd
import pandas as pd
 
# sample dataframe
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [1.1, '1.0', '1.3', 2, 5],
    'D': [True,False,False,True,True]})
print(df.info())
 
# converting all columns to string type
df = df.astype(str)
print(df.dtypes)

object_columns = df1.select_dtypes(include=['object']).columns

df1[object_columns]

# categorical variables into quantitative variables
#METHOD1

# replacing values
df1['Sex'].replace(['male', 'female'],[0, 1], inplace=True)

#METHOD2

df1=pd.read_csv("data.csv")
# get the dummies and store it in a variable
dummies = pd.get_dummies(df1.Sex)
# Concatenate the dummies to original dataframe
merged = pd.concat([df1, dummies], axis='columns')
 
# drop the values
merged.drop(['Sex', 'male'], axis='columns')

#METHOD3
df=pd.read_csv("data.csv")
# converting type of columns to 'category'
df["Embarked_cat"] = df["Embarked"].astype('category')
# Assigning numerical values and storing in another column
df["Embarked_num"] = df["Embarked_cat"].cat.codes
df

df1=pd.read_csv("data.csv")
df1["Embarked"].nunique()

#METHOD4


from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df1['Labelencoding_Embarked'] = labelencoder.fit_transform(df1["Embarked"])
df1

