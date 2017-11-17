import data_boston
db=data_boston
from sklearn.neighbors import KNeighborsRegressor

#平均回归
uni_knr=KNeighborsRegressor(weights='uniform')
uni_knr.fit(db.X_train,db.y_train)
uni_knr_y_predict=uni_knr.predict(db.X_test)

#距离加权回归
dis_knr=KNeighborsRegressor(weights='distance')
dis_knr.fit(db.X_train,db.y_train)
dis_knr_y_predict=dis_knr.predict(db.X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('R-squared value of uniform-weighted KNR:',uni_knr.score(db.X_test,db.y_test))
print('The mean squared error of uniform-weighted KNR:',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(uni_knr_y_predict)))
print('The mean absolute error of uniform-weighted KNR:',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(uni_knr_y_predict)))

print('R-squared value of distance-weighted KNR:',dis_knr.score(db.X_test,db.y_test))
print('The mean squared error of distance-weighted KNR:',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(dis_knr_y_predict)))
print('The mean absolute error of distance-weighted KNR:',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(dis_knr_y_predict)))
