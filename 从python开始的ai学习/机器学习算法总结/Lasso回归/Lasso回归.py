from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 加载数据集 & CSV文件路径
path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(path, 'boston_housing.csv')
# 从CSV文件中读取数据 
boston_df = pd.read_csv(file_path)
# 分离特征和目标变量 
X = boston_df.drop('TARGET', axis=1) 
y = boston_df['TARGET']

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Lasso回归模型
lasso = Lasso(alpha=1.0)

# 训练模型
lasso.fit(X_train, y_train)

# 预测结果
y_pred = lasso.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差: ", mse)

# 绘制特征系数变化图
plt.plot(range(X.shape[1]), lasso.coef_)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Lasso Regression Coefficients')
plt.show()