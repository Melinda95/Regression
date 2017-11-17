import data_boston
db=data_boston
from sklearn.svm import SVR

#线性核
linear_svr=SVR(kernel='linear')
linear_svr.fit(db.X_train,db.y_train)
linear_svr_y_predict=linear_svr.predict(db.X_test)

#多项式核
poly_svr=SVR(kernel='poly')
poly_svr.fit(db.X_train,db.y_train)
poly_svr_y_predict=poly_svr.predict(db.X_test)

#径向基核
rbf_svr=SVR(kernel='rbf')
rbf_svr.fit(db.X_train,db.y_train)
rbf_svr_y_predict=rbf_svr.predict(db.X_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('The value of R-squared of linear SVR',r2_score(db.y_test,linear_svr_y_predict))
print('The mean squared error of linear SVR is',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(linear_svr_y_predict)))
print('The mean absolute error of linear SVR is',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(linear_svr_y_predict)))

print('The value of R-squared of poly SVR',r2_score(db.y_test,poly_svr_y_predict))
print('The mean squared error of poly SVR is',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of poly SVR is',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(poly_svr_y_predict)))

print('The value of R-squared of rbf SVR',r2_score(db.y_test,rbf_svr_y_predict))
print('The mean squared error of rbf SVR is',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(rbf_svr_y_predict)))
print('The mean absolute error of rbf SVR is',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(rbf_svr_y_predict)))
