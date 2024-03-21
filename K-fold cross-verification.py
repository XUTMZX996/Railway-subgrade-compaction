import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pyswarm import pso
from joblib import Parallel, delayed
import joblib


from ensemble_model import WeightedEnsembleModel
data = pd.read_csv('过渡数据3.csv')

# 处理NaN值，这里使用均值填充，你可以根据实际情况选择其他填充策略
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 提取输入和输出特征
X = data[['zhengpin', 'zhengpin2', 'cishu', 'cishu2', 'cev', 'v', 'v2']]
y = data['cev2']

# 归一化处理
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# 保存归一化器
joblib.dump(scaler, 'scaler.joblib')

# 定义神经网络的输入和输出维度
input_dim = X_normalized.shape[1]
output_dim = 1

# 定义PSO的目标函数
def objective_function(params, X_train, y_train, X_test, y_test):
    # 将PSO的参数映射到神经网络超参数
    hidden_layer_sizes = (int(params[0]), int(params[1]))  # 隐藏神经元数量
    learning_rate_init = 10 ** params[2]  # 学习率
    max_iter = int(params[3])  # 迭代次数
    batch_size = int(params[4])  # 批量大小

    # 初始化神经网络模型
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=42
    )

    # 训练神经网络模型
    model.fit(X_train, y_train)

    # 预测并评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 返回均方误差作为PSO的优化目标
    return mse

# 定义PSO的搜索范围
lb = [5, 5, -5, 100, 1]  # 下限
ub = [100, 100, 0, 1000, 100]  # 上限

# 初始化k折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每个子模型的MSE值
mse_values = []

# 并行执行PSO优化
# 在 run_pso_optimization 函数中，添加保存子模型的MSE值的代码
def run_pso_optimization(train_index, test_index, model_index):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=20, args=(X_train, y_train, X_test, y_test))

    # 使用最优参数初始化神经网络模型
    optimal_model = MLPRegressor(
        hidden_layer_sizes=(int(best_params[0]), int(best_params[1])),
        learning_rate_init=10 ** best_params[2],
        max_iter=int(best_params[3]),
        batch_size=int(best_params[4]),
        random_state=42
    )

    # 归一化处理后的数据进行训练
    optimal_model.fit(X_train, y_train)

    # 保存子模型
    model_filename = f'model_{model_index}.joblib'
    joblib.dump(optimal_model, model_filename)  # 请根据实际情况修改文件名

    # 预测验证集
    y_pred = optimal_model.predict(X_test)

    # 计算子模型在验证集上的均方误差
    mse_val = mean_squared_error(y_test, y_pred)
    mse_values.append(mse_val)

    return optimal_model, mse_val

# 进行k折交叉验证并行执行PSO优化
models_and_mse = Parallel(n_jobs=-1)(
    delayed(run_pso_optimization)(train_index, test_index, i) for i, (train_index, test_index) in enumerate(kf.split(X_normalized)))

# 提取子模型和对应的MSE值
trained_models, mse_values = zip(*models_and_mse)

# 计算每个子模型的权重（基于MSE值，MSE越小，权重越大）
weights = np.array([1 / mse for mse in mse_values])
weights /= np.sum(weights)

# 创建WeightedEnsembleModel的实例
final_model = WeightedEnsembleModel(models=trained_models, weights=weights)

# 使用joblib保存最终模型
joblib.dump(final_model, '最终模型.joblib')

# 输出每个子模型的MSE值
for i, mse_val in enumerate(mse_values):
    print(f"子模型 {i+1} 的MSE值: {mse_val}")

# 输出最终模型的平均MSE值
average_mse = np.average(mse_values, weights=weights)
print(f"最终模型的平均MSE值: {average_mse}")

















