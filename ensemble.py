import data_boston
db=data_boston
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

rfr=RandomForestRegressor()
rfr.fit(db.X_train,db.y_train)
rfr_y_predict=rfr.predict(db.X_test)

etr=ExtraTreesRegressor()
etr.fit(db.X_train,db.y_train)
etr_y_predict=etr.predict(db.X_test)

gbr=GradientBoostingRegressor()
gbr.fit(db.X_train,db.y_train)
gbr_y_predict=gbr.predict(db.X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('R-squared value of RFR:',rfr.score(db.X_test,db.y_test))
print('The mean squared error of RFR:',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(rfr_y_predict)))
print('The mean absolute error of RFR:',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(rfr_y_predict)))

print('R-squared value of ETR:',etr.score(db.X_test,db.y_test))
print('The mean squared error of ETR:',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(etr_y_predict)))
print('The mean absolute error of ETR:',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(etr_y_predict)))

print('R-squared value of GBR:',gbr.score(db.X_test,db.y_test))
print('The mean squared error of GBR:',mean_squared_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(gbr_y_predict)))
print('The mean absolute error of GBR:',mean_absolute_error(db.ss.inverse_transform(db.y_test),db.ss.inverse_transform(gbr_y_predict)))
