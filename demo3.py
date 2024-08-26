import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from keras.utils import to_categorical
from keras.models import load_model


class LSTMModel:
    def __init__(self, data_path, time_steps=100, step_forward=0):
        # 初始化LSTMModel类，设置数据路径、时间步长和前向步数
        self.data_path = data_path
        self.time_steps = time_steps
        self.step_forward = step_forward
        self.model = None

    def create_dataset(self, X, y, timestamps):
        # 基于时间步长和前向步数创建输入序列和对应的目标值序列
        Xs, ys, ts = [], [], []
        for i in range(len(X) - self.time_steps - self.step_forward + 1):
            # 创建输入序列
            Xs.append(X[i:(i + self.time_steps)])
            # 记录目标值，对应于时间步后的数据
            ys.append(y[i + self.time_steps + self.step_forward - 1])
            # 记录时间戳，对应于目标值的位置
            ts.append(timestamps[i + self.time_steps + self.step_forward - 1])
        return np.array(Xs), np.array(ys), np.array(ts)

    def preprocess_data(self):
        # 预处理数据：加载数据、处理缺失值、转换数据类型和标准化
        data = pd.read_excel(self.data_path)
        data_filled = data.interpolate(method='nearest')  # 使用最近值插值法填充缺失值

        # 将特定列转换为整数类型
        data_filled['Robot_ProtectiveStop'] = data_filled['Robot_ProtectiveStop'].astype(int)
        data_filled['grip_lost'] = data_filled['grip_lost'].astype(int)

        # 将'Timestamp'列转换为日期时间格式
        data_filled['Timestamp'] = pd.to_datetime(data_filled['Timestamp'].str.strip('"'), format='ISO8601')

        # 将'Timestamp'列设置为索引
        data = data_filled.set_index('Timestamp')

        # 选择特征列，排除目标变量和其他不需要的列
        features = [col for col in data.columns if
                    col not in ['Robot_ProtectiveStop', 'grip_lost', 'cycle ', 'Timestamp']]
        X = data[features].values  # 提取特征值
        y = data['Robot_ProtectiveStop'].values  # 提取目标变量

        # 标准化特征数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 将目标变量转换为二进制分类格式
        y_binary = to_categorical(y)
        timestamps = data.index.values  # 提取时间戳

        # 创建基于时间步长和前向步数的数据集
        X_seq, y_seq, ts = self.create_dataset(X_scaled, y_binary, timestamps)

        return X_seq, y_seq, ts

    def split_data(self, X, y, timestamps, test_size=0.2):
        # 将数据集拆分为训练集和测试集，按比例分割数据
        split_index = int(len(X) * (1 - test_size))  # 计算分割索引
        X_train = X[:split_index]  # 训练集特征
        X_test = X[split_index:]  # 测试集特征
        y_train = y[:split_index]  # 训练集目标
        y_test = y[split_index:]  # 测试集目标
        ts_train = timestamps[:split_index]  # 训练集时间戳
        ts_test = timestamps[split_index:]  # 测试集时间戳
        return X_train, X_test, y_train, y_test, ts_train, ts_test


    def train_model(self, X_train, y_train):
        # 定义并训练LSTM模型
        self.model = Sequential([
            # LSTM层，50个单元，输入形状为(时间步长, 特征数)
            LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),  # Dropout层，防止过拟合
            Dense(2, activation='softmax')  # 全连接层，输出为2个类别的概率分布
        ])

        # 编译模型，使用二元交叉熵损失函数和Adam优化器
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 训练模型，设置训练轮数、批量大小和验证集
        self.model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, verbose=2)

        # 保存训练好的模型
        self.model.save('model/lstm_model.h5')

    def evaluate_model(self, X_test, y_test, timestamps):
        # 加载并评估已保存的模型
        self.model = load_model('model/lstm_model.h5')

        # 使用模型进行预测
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)  # 获取预测的类别

        # 计算各种评价指标
        auc = roc_auc_score(y_test[:, 1], y_pred_prob[:, 1])
        recall = recall_score(y_test.argmax(axis=1), y_pred)
        precision = precision_score(y_test.argmax(axis=1), y_pred)
        f1 = f1_score(y_test.argmax(axis=1), y_pred)
        accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)

        # 输出评价指标
        print("ROC AUC Score:", auc)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1 Score:", f1)
        print("Accuracy:", accuracy)

        # 将预测结果与时间戳合并，返回结果数据框
        results_df = pd.DataFrame({'Timestamp': timestamps, 'Predicted': y_pred})
        return results_df


# 创建LSTM模型实例
model = LSTMModel('data/dataset_02052023.xlsx')
# 预处理数据
X_seq, y_seq, timestamps = model.preprocess_data()
# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test, ts_train, ts_test = model.split_data(X_seq, y_seq, timestamps)
# 训练模型
model.train_model(X_train, y_train)
# 评估模型，并返回预测结果
results_df = model.evaluate_model(X_test, y_test, ts_test)

# 合并预测结果到原始数据
full_data = pd.read_excel('data/dataset_02052023.xlsx')
full_data['Timestamp'] = pd.to_datetime(full_data['Timestamp'].str.strip('"'), format='ISO8601')
# 本地化 results_df 的时间戳为 UTC
results_df['Timestamp'] = pd.to_datetime(results_df['Timestamp']).dt.tz_localize('UTC')
# 确认 full_data 的时间戳为 UTC
full_data['Timestamp'] = pd.to_datetime(full_data['Timestamp']).dt.tz_convert('UTC')
# 合并数据，将预测结果合并到原始数据中
merged_data = full_data.merge(results_df.set_index('Timestamp'), how='left', on='Timestamp')
# 保存结果到CSV文件
merged_data.to_csv('output/complete_with_predictions.csv')
