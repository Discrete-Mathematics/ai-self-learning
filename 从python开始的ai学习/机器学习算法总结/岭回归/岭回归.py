from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 加载数据集 & CSV文件路径
path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(path, 'boston_housing.csv')
# 从CSV文件中读取数据 
boston_df = pd.read_csv(file_path)
# 分离特征和目标变量 
X = boston_df.drop('TARGET', axis=1).values 
y = boston_df['TARGET']

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 归一化特征矩阵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 岭回归模型训练和预测
ridge = Ridge(alpha=1.0) # 正则化参数 alpha 默认为 1.0
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("均方误差: ", mse)

# 绘制预测值与真实值的散点图
plt.scatter(y_test, y_pred)
plt.xlabel('actual')
plt.ylabel('predict')
plt.title('ridge')
plt.show()