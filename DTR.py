import data_boston
db=data_boston
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(db.X_train,db.y_train)
dtr_y_predict=dtr.predict(db.X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('R-squared value of DTR:',dtr.score(db.X_test,db.y_test))
print('The mean squared error of DTR:',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(dtr_y_predict)))
print('The mean absolute error of DTR:',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(dtr_y_predict)))
