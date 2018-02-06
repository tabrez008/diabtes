
#Importing all packages
import pandas as pd
from numpy import *
#import seaborn
import sklearn
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from brew.base import Ensemble
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve , auc
import matplotlib.pyplot as plt
import random

#Reading dataset
df = pd.read_csv("pima-indians-diabetes.csv",names = [0,1,2,3,4,5,6,7,8])

#Splitting dataset into train and test
from sklearn.cross_validation import train_test_split

train, test = train_test_split(df, test_size = 0.2, random_state = 0)

train_array=train.values
test_array=test.values
features_train = train_array[:,0:8]
labels_train = train_array[:,8]
features_test = test_array[:,0:8]
labels_test =  test_array[:,8]

Accuracy=[]
Precision = []
Recall = []
f1 = []

#Logistic Regression
model1 = LogisticRegression(C=10,random_state=0)
model1.fit(features_train,labels_train)
pred1 = model1.predict(features_test)
Accuracy.append(accuracy_score(pred1,labels_test))
Precision.append(precision_score(pred1,labels_test))
Recall.append(recall_score(pred1,labels_test))
f1.append(f1_score(pred1,labels_test))
false_positive_1,true_positive_1,threshold_1 = roc_curve(labels_test,pred1)

#Random Forest
model2 = RandomForestClassifier(n_estimators=500 ,random_state = 0)
model2.fit(features_train,labels_train)
pred2 = model2.predict(features_test)
Accuracy.append(accuracy_score(pred2,labels_test))
Precision.append(precision_score(pred2,labels_test))
Recall.append(recall_score(pred2,labels_test))
f1.append(f1_score(pred2,labels_test))
false_positive_2,true_positive_2,threshold_2 = roc_curve(labels_test,pred2)

#SVM
model3 = SVC(C=0.1, kernel = 'linear',probability = True)
model3.fit(features_train,labels_train)
pred3 = model3.predict(features_test)
Accuracy.append(accuracy_score(pred3,labels_test))
Precision.append(precision_score(pred3,labels_test))
Recall.append(recall_score(pred3,labels_test))
f1.append(f1_score(pred3,labels_test))
false_positive_3,true_positive_3,threshold_3 = roc_curve(labels_test,pred3)



# Stacking model


layer_1 = Ensemble([model1,model2,model3])
layer_2 = Ensemble([sklearn.clone(model3)])


stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack)


sclf.fit(feature_train , label_train)
pred = sclf.predict(feature_test)
Accuracy.append(accuracy_score(pred,label_test))
Precision.append(precision_score(pred,label_test))
Recall.append(recall_score(pred,label_test))
f1.append(f1_score(pred,label_test))
false_positive_rate, true_positive_rate, thresholds = roc_curve(label_test, pred)




import matplotlib.pyplot as plt
X=[10,15,20,25]
my_xticks = ['Logistic Regression','RandomForest','SVM','Stacking']

#accuracy curve
print("Accuracy")
print(Accuracy)
plt.xticks(X, my_xticks)
plt.xlabel('Classifier')
plt.ylabel('Value')
plt.title('ACCURACY')
plt.plot(X, Accuracy)
plt.axis([8,27,0.75,0.9])
plt.show()

#precision curve
print("Precision")
print(Precision)
plt.xticks(X, my_xticks)
plt.xlabel('Classifier')
plt.ylabel('Value')
plt.title('PRECISION')
plt.plot(X,Precision)
plt.axis([8,27,0.55,0.77])
plt.show()

#recall_score
print('Recall')
print(Recall)
plt.xticks(X, my_xticks)
plt.xlabel('Classifier')
plt.ylabel('Value')
plt.title('RECALL')
plt.plot(X,Recall)
plt.axis([8,27,0.65,0.88])
plt.show()

#F1_score
print('F1_scores') 
print(f1)
plt.xticks(X, my_xticks)
plt.xlabel('Classifier')
plt.ylabel('Value')
plt.title('F1 SCORE')
plt.plot(X,f1)
plt.axis([8,27,0.6,0.84])
plt.show()


#plotting roc curve

roc_auc = auc(false_positive_rate, true_positive_rate)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE')
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.plot(false_positive_1, true_positive_1, 'r')
plt.plot(false_positive_2, true_positive_2, 'y')
plt.plot(false_positive_3, true_positive_3, 'g')
plt.legend(['Stacking','Logistic Regression', 'RandomForest', 'SVM'],loc='lower right')
plt.show()
