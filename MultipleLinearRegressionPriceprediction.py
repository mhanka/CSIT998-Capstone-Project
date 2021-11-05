import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Read data from CSV
PriceData = pd.read_csv('PriceNumeric.csv')
PriceData.head(4).T

PriceData.info()
PriceData.describe().transpose()

#Remove null values
PriceData.isnull().sum()
X = PriceData.drop('SoldPrice',axis =1).values
y = PriceData['SoldPrice'].values

#test train split of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#Batch scalarization
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))


regressor = LinearRegression()
regressor.fit(X_train, y_train)
#model evaluation
print(regressor.intercept_)
print(regressor.coef_)
#predicting the test set result
y_pred = regressor.predict(X_test)
#put results as a DataFrame
coeff_df = pd.DataFrame(regressor.coef_, PriceData.drop('SoldPrice',axis =1).columns, columns=['Coefficient'])
print(coeff_df)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
print(df1)


#Model evaluation using estimators
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
print('Linear Regression Model:')
print("Train Score {:.2f}".format(regressor.score(X_train,y_train)))
print("Test Score {:.2f}".format(regressor.score(X_test, y_test)))

# Visualizing the predictions
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
#Save plot as img
plt.plot(y_test,y_test,'r')
plt.savefig('scatterplot2.png')

#Visualizing normalization
fig1 = plt.figure(figsize=(10,5))
residuals = (y_test- y_pred)
sns.distplot(residuals)
plt.savefig('residual1.png')