import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Social_Network_Ads.csv")
df.head()

df.dtypes

(df.isnull()).sum()

df.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_test,y_pred)
print('Confusion Matrix: ')
print(cm1)   
ac1 = accuracy_score(y_test, y_pred)*100
print('Accuracy Score:')
print(ac1)

tp=cm1[0][0]
tn=cm1[1][1]
fp=cm1[1][0]
fn=cm1[0][1]
total=tp+tn+fp+fn

error_rate=(fp+fn)/(total)
print('error rate: ')
print(error_rate)

from sklearn.metrics import classification_report
print('                        classification report:')
print('')
print(classification_report(y_test,y_pred))

