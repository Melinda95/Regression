from sklearn.datasets import  load_boston
boston=load_boston()
#print(boston.DESCR)

from sklearn.cross_validation import train_test_split
import numpy as np

X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
y_train=ss.fit_transform(y_train)
y_test=ss.transform(y_test)