from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 


iris = load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


model = SVC(kernel='linear')
model.fit(x_train,y_train)
ypred = model.predict(x_test)
print("Score : ",model.score(x_test,y_test)*100)

import sklearn.metrics as m
print ("Accuracy : ",m.accuracy_score(y_test,ypred)*100)

cm = m.confusion_matrix(y_test,ypred)
print (m.classification_report(y_test,ypred))

#import seaborn as sns
import matplotlib.pyplot as plt
#sns.heatmap(cm,annot=True)
plt.show()
