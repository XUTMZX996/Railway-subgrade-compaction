import time
time_start = time.time()
import itertools
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.api import qqplot
import math
import os
import numpy as np
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow import optimizers
from tensorflow.keras import Sequential, callbacks
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf


class CgcpLSTM:
    def __init__(self, name):
        """
        构造函数，初始化模型
        :param data_list: 真实数据列表
        """
        # 神经网络名称
        self.name = name
        # 训练集占总样本的比例
        self.train_all_ratio = 0.875
        # 连续样本点数
        self.continuous_sample_point_num = 20
        # 定义归一化：归一化到(0，1)之间
        self.sc = MinMaxScaler(feature_range=(0, 1))
        # 每次喂入神经网络的样本数
        self.batch_size = 64
        # 数据集的迭代次数
        self.epochs = 100
        # 每多少次训练集迭代，验证一次测试集
        self.validation_freq = 1
        # 配置模型
        self.model = Sequential([
            # LSTM层（记忆体个数，是否返回输出（True：每个时间步输出ht，False：仅最后时间步输出ht））
            # 配置具有80个记忆体的LSTM层，每个时间步输出ht
            LSTM(80, return_sequences=True),
            Dropout(0.2),
            # 配置具有100个记忆体的LSTM层，仅在最后一步返回ht
            LSTM(100),
            Dropout(0.2),
            Dense(1)
        ])
        # 配置训练方法
        # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
        self.model.compile(
            optimizer=optimizers.Adam(0.001),
            loss='mean_squared_error',  # 损失函数用均方误差
        )
        # 配置断点续训文件
        self.checkpoint_save_path = os.path.abspath(os.path.dirname(__file__)) + "\\checkpoint\\" + self.name + "_LSTM_stock.weights.h5"

        if os.path.exists(self.checkpoint_save_path + '.index'):
            print('-' * 20 + "加载模型" + "-" * 20)
            self.model.load_weights(self.checkpoint_save_path)

        # 断点续训，存储最佳模型
        self.cp_callback = callbacks.ModelCheckpoint(filepath=self.checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     # monitor='val_accuracy',
                                                     monitor='val_loss',
                                                     )

    def make_set(self, data_list):
        """
        使用历史数据制作训练集和测试集
        :param data_list: 历史数据列表
        :return: train_set, test_set 归一化处理后的训练集合测试集
        """
        # 将历史数据装换为ndarray
        if isinstance(data_list, list):
            data_array = np.array(data_list)
        elif isinstance(data_list, np.ndarray):
            data_array = data_list
        else:
            raise Exception("数据源格式错误")

        # 对一维矩阵进行升维操作
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(data_array.shape[0], 1)

        if data_array.shape[1] != 1:
            raise Exception("数据源形状有误")

        # 按照比例对数据进行分割
        index = int(data_array.shape[0] * self.train_all_ratio)
        train_set = data_array[:index, :]
        test_set = data_array[index:, :]

        print("train_set_shape:{}".format(train_set.shape))
        # 对训练集和测试集进行归一化处理
        train_set, test_set = self.gui_yi(train_set, test_set)

        print("训练集长度：{}".format(len(train_set)))
        print("测试集长度：{}".format(len(test_set)))
        return train_set, test_set

    def gui_yi(self, train_set, test_set):
        """
        对训练集合测试集进行归一化处理
        :param test_set: 未进行归一化的训练集数据
        :param train_set: 未进行归一化处理的测试集数据
        :return: train_set, test_set 归一化处理后的训练集合测试集
        """
        # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
        train_set_scaled = self.sc.fit_transform(train_set)
        # 利用训练集的属性对测试集进行归一化
        test_set = self.sc.transform(test_set)
        return train_set_scaled, test_set

    def fan_gui_yi(self, data_set):
        """
        逆归一化
        :param data_set: 需要还原的数据
        :return:
        """
        # 对数据进行逆归一化还原
        data_set = self.sc.inverse_transform(data_set)
        return data_set

    def train(self, x_train, y_train, x_test, y_test):
        """
        训练模型
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        # 训练模型
        history = self.model.fit(x_train, y_train,
                                 # 每次喂入神经网络的样本数
                                 batch_size=self.batch_size,
                                 # 数据集的迭代次数
                                 epochs=self.epochs,
                                 validation_data=(x_test, y_test),
                                 # 每多少次训练集迭代，验证一次测试集
                                 validation_freq=self.validation_freq,
                                 callbacks=[self.cp_callback])
        # 输出模型各层的参数状况
        self.model.summary()
        # 参数提取
        self.save_args_to_file()

        # 获取模型当前loss值
        loss = history.history['loss']
        print("loss:{}".format(loss))
        try:
            val_loss = history.history['val_loss']
            print("val_loss:{}".format(val_loss))
        except:
            pass

    def save_args_to_file(self):
        """
        参数提取，将参数保存至文件
        :return:
        """
        # 指定参数存取目录
        file_path = os.path.abspath(
            os.path.dirname(__file__)) + "\\weights\\"
        # 目录不存在则创建
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # 打开文本文件
        file = open(file_path + self.name + "_weights.txt", 'w')
        # 将参数写入文件
        for v in self.model.trainable_variables:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')
        file.close()

    def test(self, x_test, test_set):
        """
        预测测试
        :param x_test:
        :param test_set: 测试集
        :return:
        """

        # 测试集输入模型进行预测
        predicted_stock_price = self.model.predict(x_test)
        # 对预测数据还原---从（0，1）反归一化到原始范围
        predicted_stock_price = self.fan_gui_yi(predicted_stock_price)

        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_stock_price = self.fan_gui_yi(test_set[self.continuous_sample_point_num:])

        # ##########evaluate##############
        # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
        mse = mean_squared_error(predicted_stock_price, real_stock_price)
        # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
        rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
        # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
        mae = mean_absolute_error(predicted_stock_price, real_stock_price)
        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)

    def make_x_y_train_and_test(self, data_list):
        """
        制作x_train（训练集输入特征）, y_train（训练集标签）, x_test（测试集输入特征）, y_test（测试集标签）
        :param data_list:
        :return:
        """
        # 获取归一化后的训练集合测试集
        train_set, test_set = self.make_set(data_list=data_list)
        # 初始化x_train（训练集输入特征）, y_train（训练集标签）, x_test（测试集输入特征）, y_test（测试集标签）
        x_train, y_train, x_test, y_test = [], [], [], []

        # 利用for循环，遍历整个训练集，提取训练集中连续样本为训练集输入特征和标签
        for i in range(self.continuous_sample_point_num, len(train_set)):
            x_train.append(train_set[i - self.continuous_sample_point_num:i, 0])
            y_train.append(train_set[i, 0])
        # 对训练集进行打乱
        np.random.seed(7)
        np.random.shuffle(x_train)
        np.random.seed(7)
        np.random.shuffle(y_train)
        tf.random.set_seed(7)
        # 将训练集由list格式变为array格式
        x_train, y_train = np.array(x_train), np.array(y_train)

        # 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
        x_train = self.change_data_to_rnn_input(x_train)
        # 测试集
        # 利用for循环，遍历整个测试集，提取训练集中连续样本为训练集输入特征和标签
        for i in range(self.continuous_sample_point_num, len(test_set)):
            x_test.append(test_set[i - self.continuous_sample_point_num:i, 0])
            y_test.append(test_set[i, 0])
        # 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
        x_test, y_test = np.array(x_test), np.array(y_test)
        print("x_test_shape：{}".format(x_test.shape))
        x_test = self.change_data_to_rnn_input(x_test)
        return train_set, test_set, x_train, y_train, x_test, y_test

    def change_data_to_rnn_input(self, data_array):
        """
        将数据转变为RNN输入要求的维度
        :param data_array:
        :return:
        """
        # 对输入类型进行转换
        if isinstance(data_array, list):
            data_array = np.array(data_array)
        elif isinstance(data_array, np.ndarray):
            pass
        else:
            raise Exception("数据格式错误")
        rnn_input = np.reshape(data_array, (data_array.shape[0], self.continuous_sample_point_num, 1))
        return rnn_input

    def predict(self, history_data):
        """
        使用模型进行预测
        :param history_data: 历史数据list
        :return:预测值
        """
        # 将列表或数组转换为数组并提取最后一组数据
        if isinstance(history_data, list):
            history_data_array = history_data[self.continuous_sample_point_num * -1:]
            history_data_array = np.array(history_data_array)
        elif isinstance(history_data, np.ndarray):
            history_data_array = history_data[self.continuous_sample_point_num * -1:]
        else:
            raise Exception("数据格式错误")

        # 对一维数据进行升维处理
        if len(history_data_array.shape) == 1:
            history_data_array = history_data_array.reshape(1, self.continuous_sample_point_num)

        # 对数据形状进行效验
        if history_data_array.shape[1] != self.continuous_sample_point_num:
            raise Exception("数据形状有误")

        # 对数据进行归一化处理
        history_data_array = history_data_array.T
        history_data_array = self.sc.transform(history_data_array)
        history_data_array = history_data_array.T

        # 转换为RNN需要的数据形状
        history_data_array = self.change_data_to_rnn_input(history_data_array)

        # 测试集输入模型进行预测
        predicted_stock_price = self.model.predict(history_data_array)
        # 对预测数据还原---从（0，1）反归一化到原始范围
        predicted_stock_price = self.fan_gui_yi(predicted_stock_price)
        # 预测值
        value = predicted_stock_price[-1][-1]
        print("预测值：{}".format(value))
        return value


if __name__ == '__main__':
    # 调用GPU加速
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 读入数据
    data_raw = pd.read_csv('sp500改.csv', usecols=[0, 1], squeeze=True)
    features = ['haoshi']
    data_raw = data_raw[features]

    # 显示原数据
    plt.figure(figsize=(10, 3))
    plt.title('数据haoshi')
    plt.xlabel('time')
    plt.ylabel('haoshi')
    plt.plot(data_raw, 'blue', label='haoshi')
    plt.legend()
    plt.show()
    # ACF 和 PACF 图
    plot_acf(data_raw, lags=20)
    plt.title('ACF')
    plt.show()
    #
    plot_pacf(data_raw, lags=20)
    plt.title('PACF')
    plt.show()
    # 填充缺失值
    # data_raw.fillna(method='ffill', inplace=True)  # 使用前向填充
    # data_raw.dropna(inplace=True)  # 删除缺失值
    #
    # # 根据ACF和PACF图来确定p、d、q的范围
    # acf_values, pacf_values = acf(data_raw), pacf(data_raw)
    # p_max = np.where(acf_values < 0.05)[0][0]
    # q_max = np.where(pacf_values < 0.05)[0][0]
    #
    # # 打印确定的范围
    # print("p的范围：0 -", p_max)
    # print("q的范围：0 -", q_max)
    #
    # # 注意：d的范围需要根据具体情况来确定，这里默认取0
    # d_min = 0
    # d_max = 2
    # print("d的范围：", d_min, "-", d_max)

    p_min = 0
    p_max = 5
    d_min = 0
    d_max = 2
    q_min = 0
    q_max = 5

    results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
    ts_train = data_raw.iloc[:int(len(data_raw) * 0.8)]
    for p, d, q in itertools.product(range(p_min, p_max + 1),
                                     range(d_min, d_max + 1),
                                     range(q_min, q_max + 1)):
        if p == 0 and d == 0 and q == 0:
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue

        try:
            model = sm.tsa.ARIMA(ts_train, order=(p, d, q), )
            results = model.fit()
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        except:
            continue
    results_bic = results_bic[results_bic.columns].astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.heatmap(results_bic, mask=results_bic.isnull(), ax=ax, annot=True, fmt='.2f', )
    ax.set_title('BIC')
    plt.show()
    train_results = sm.tsa.arma_order_select_ic(data_raw, ic=['aic', 'bic'], trend='c', max_ar=p_max, max_ma=q_max)
    print('AIC', train_results.aic_min_order)
    print('BIC', train_results.bic_min_order)
    # 从BIC结果中获取最佳的p、d、q值
    optimal_pdq = results_bic.stack().idxmin()
    optimal_p = int(optimal_pdq[0].replace('AR', ''))
    optimal_q = int(optimal_pdq[1].replace('MA', ''))

    # 使用最佳参数拟合ARIMA模型
    model = sm.tsa.ARIMA(data_raw, order=(optimal_p, d_max, optimal_q))
    fit = model.fit()
    # 获取残差
    resid = fit.resid
    # 画qq图
    qqplot(resid, line='q', fit=True)
    plt.show()
    # 获得ARIMA的预测值
    preds = fit.predict(1, len(data_raw), typ='levels')
    preds_pd = preds.to_frame()
    preds_pd.index -= 1
    arima_result = pd.DataFrame(columns=['haoshi'])
    arima_result['haoshi'] = data_raw
    arima_result['predicted'] = preds_pd
    arima_result['residuals'] = arima_result['haoshi'] - arima_result['predicted']
    new_data = arima_result
    lstm_data = new_data['residuals'][:].values.astype(float)
    data_list = lstm_data.tolist()
    # 初始化模型
    model = CgcpLSTM(name="时间预测")
    # 获取训练和测试的相关参数
    train_set, test_set, x_train, y_train, x_test, y_test = model.make_x_y_train_and_test(data_list=data_list)
    # 训练模型
    model.train(x_train, y_train, x_test, y_test)
    # 对模型进行测试
    model.test(x_test, test_set)

    # 利用模型进行预测
    history = [x for x in range(len(data_raw))]  # 使用整个数据集的长度
    # 获得ARIMA的预测值
    arima_preds = fit.predict(1, len(data_raw), typ='levels')
    arima_preds_pd = arima_preds.to_frame()
    arima_preds_pd.index -= 1

    # 获取LSTM的预测值
    lstm_preds = model.predict(history)

    # 最终预测值为ARIMA预测值与LSTM预测值的和
    final_preds = arima_preds_pd.values.flatten() + lstm_preds

    # 输出最终预测值
    print("Final Predictions:", final_preds)

    # 画出整个数据集的真实值和最终预测值的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(data_raw.index, data_raw['haoshi'], label='Actual haoshi', color='blue')
    plt.plot(data_raw.index, final_preds, label='Predicted haoshi', color='red')
    plt.title('Actual vs Predicted haoshi')
    plt.xlabel('Time')
    plt.ylabel('haoshi')
    plt.legend()
    plt.show()

    # 计算均方误差（MSE）
    mse = mean_squared_error(data_raw['haoshi'], final_preds)
    print("Mean Squared Error (MSE):", mse)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(data_raw['haoshi'], final_preds)
    print("Mean Absolute Error (MAE):", mae)






