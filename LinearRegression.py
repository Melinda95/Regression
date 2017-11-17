from sklearn.linear_model import LinearRegression
import data_boston
db=data_boston

lr=LinearRegression()
lr.fit(db.X_train,db.y_train)
lr_y_predict=lr.predict(db.X_test)

from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(db.X_train,db.y_train)
sgdr_y_predict=sgdr.predict(db.X_test)

print('The value of default measurement of LinearRegression is',lr.score(db.X_test,db.y_test))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

print('The value of R-squared of LinearRegression is',r2_score(db.y_test,lr_y_predict))
print('The mean squared error of LinearRegression is',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(lr_y_predict)))
print('The mean absolute error of LinearRegression is',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(lr_y_predict)))

print('The value of default measurement of SGDRegression is',sgdr.score(db.X_test,db.y_test))
print('The value of R-squared of SGDRegression is',r2_score(db.y_test,sgdr_y_predict))
print('The mean squared error of SGDRegression is',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(sgdr_y_predict)))
print('The mean absolute error of SGDRegression is',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(sgdr_y_predict)))
