import seaborn as sns
import pandas as pd
import numpy as np

data = sns.load_dataset('iris')
print('Original Dataset')
data.head()

x = data.iloc[:,:4].values
y = data['species'].values

data.info()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score,auc,f1_score,precision_score,recall_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))

df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
df

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix: ')
print(cm)   
ac1 = accuracy_score(y_test, y_pred)*100
print('Accuracy Score:')
print(ac1)

print("----------For Setosa----------")
tp=cm[0][0]
fn=(cm[0][1])+(cm[0][2])
tn=(cm[1][1])+(cm[1][2])+(cm[2][1])+(cm[2][2])
fp=(cm[1][0])+(cm[2][0])
print('true positive: ',tp)
print('false positive: ',fp)
print('true negative: ',tn)
print('false negative: ',fn)
error_rate=(fp+fn)/(tp+tn+fp+fn)
print('error rate:', error_rate )

