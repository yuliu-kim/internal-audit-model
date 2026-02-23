import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 导入Excel文件
df = pd.read_excel('try.xlsx')  # 替换为你的Excel文件路径

# 选择特征列和目标列
X = df[['first', 'topten', 'seperation']]
Y = df['soe2']

# 划分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, Y_train)

# 预测测试集结果
Y_pred = rf.predict(X_test)

# 计算均方误差
mse = mean_squared_error(Y_test, Y_pred)
print(f'均方误差: {mse}')

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x=rf.feature_importances_, y=X.columns)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# 绘制实际值与预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.title('Actual vs Predicted')
plt.xlabel('Actual soe2')
plt.ylabel('Predicted soe2')
plt.show()

# 如果需要，可以绘制残差图来检查模型预测的准确性
residuals = Y_test - Y_pred
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.show()