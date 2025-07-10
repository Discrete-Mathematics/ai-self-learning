import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# 加载加州房屋数据集
california = fetch_california_housing()

# 数据准备, sklearn中常用bunch对象组织数据集。
# Bunch 对象具有字典的行为，同时还可以通过属性访问方式获取数据。
# Bunch 通常包含以下内容：
#
# data：特征矩阵，通常是 NumPy 数组。
# target：目标变量，通常是 NumPy 数组。
# feature_names：特征的名称。
# target_names：目标变量的名称（分类任务常见）。
# DESCR：数据集的描述信息（字符串形式）。
X = california.data
y = california.target

# 将数据集划分为训练集和测试集
X_train, X_teat, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# 创建线性回归模型实例
model = LinearRegression()

# 模型训练
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_teat)

# 计算均方误差
mse = np.mean((y_pred - y_test) ** 2)
print("均方误差：", mse)

# 结果可视化
sns.set()
plt.scatter(y_test, y_pred, s = 2 * (y_pred - y_test) ** 2, c = (y_pred - y_test) ** 2, cmap = "viridis")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2)
plt.xlabel('actual_price')
plt.ylabel('predict_price')
plt.title('linear')
plt.colorbar()
plt.show()



