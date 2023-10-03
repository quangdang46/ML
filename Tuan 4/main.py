import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


data=pd.read_csv('kc_house_data.csv')
# print(df.head())

X = data[['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'sqft_living15']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# linear regression
linear=LinearRegression()
linear.fit(X_train,y_train)
linear_pred=linear.predict(X_test)
print('MAE_linear: ', mean_absolute_error(y_test, linear_pred))
# KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('MAE_KNN: ', mean_absolute_error(y_test, knn_pred))


# decision tree
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
print('MAE_tree: ', mean_absolute_error(y_test, tree_pred))


plt.scatter(y_test, linear_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression')
plt.show()

plt.scatter(y_test, tree_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Decision Tree')
plt.show()

plt.scatter(y_test, knn_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('KNN')
plt.show()





