import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


pwd = os.path.dirname(os.path.realpath(__file__))
path = os.makedirs(os.path.join(pwd,'pcs'), exist_ok=True)
# 导入加州房屋数据
california = fetch_california_housing()

# 提取特征,目标变量,和特征名,设置训练集和测试集
cal = pd.DataFrame(data=california.data, columns=california.feature_names)
target = pd.DataFrame(data=california.target, columns=california.target_names)
data_set = pd.concat([cal, target], axis=1)
train_set, test_set = train_test_split(data_set, test_size=0.2, shuffle=True,random_state=1000)
train_X_set = train_set.iloc[:,0:8]
train_y_set = train_set.iloc[:,8]
test_X_set = test_set.iloc[:,0:8]
test_y_set = test_set.iloc[:,8]
# X = california.data
# y = california.target
# X_name = california.feature_names
# print(X_name)

# 将特征向量转换为多项式特征， 即生成公式推导中的矩阵形式, 这里是二次多项式
poly = PolynomialFeatures(degree = 2)  # 实例化poly
train_X_poly = poly.fit_transform(train_X_set)

# 使用多项式回归模型进行拟合，将多项式回归转换成线性回归，从矩阵的形式来看，其实都是 Ax = b
model = LinearRegression() 
model.fit(train_X_poly, train_y_set)

# 预测新的房屋价格，测试集
# X_new = X[0].reshape(1, -1) # (1, -1) 1 表示新数组有1行，-1表示列数由原数据的长度自动判断（保持数据完整）
test_X_poly = poly.transform(test_X_set)
y_pred = model.predict(test_X_poly)

# 计算模型的性能指标
# y_pred = model.predict(X_poly)
mse = mean_squared_error(test_y_set, y_pred)

print(mse)
# 绘制原始数据散点图和拟合曲线图
feature_num = 4
plt.scatter(train_X_set.iloc[:, feature_num], train_y_set, color = 'blue', label = 'Actual', alpha=0.5)
plt.scatter(test_X_set.iloc[:, feature_num], test_y_set, color = 'red', label = 'Prediction', alpha=0.3)
plt.plot(test_X_set.iloc[:, feature_num], y_pred, color = 'green', label = 'Regression')
plt.xlabel('Population')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'{pwd}/pcs/image.png')

# MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude	MedHouseVal
