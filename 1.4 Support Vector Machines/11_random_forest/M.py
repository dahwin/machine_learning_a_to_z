'''dahyun+darwin = dahwin'''
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
digits = load_digits()
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
df = pd.DataFrame(digits.data)
x = df
y = digits.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
randomf= RandomForestClassifier()
randomf.fit(x_train,y_train)
randomf.score(x_test,y_test)
y_predict =randomf.predict(x_test)
cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize=(10,8))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('dahwin.png')
