import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
from hmmlearn import hmm
import math


# 元过程划分方法
def meta_process_segmentation(sequence, window_size=10):
    """
    将输入的骨架图像序列进行元过程划分
    :param sequence: 输入的骨架图像序列，形状为 (N, C, T, V, M)
    :param window_size: 元过程的时间窗口大小
    :return: 划分后的元过程列表
    """
    N, C, T, V, M = sequence.size()
    segments = []
    num_segments = T // window_size

    for i in range(num_segments):
        start = i * window_size
        end = (i + 1) * window_size
        segment = sequence[:, :, start:end, :, :]
        segments.append(segment)

    return segments  # 返回分段后的元过程序列列表


# Transformer Encoder模块
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=6, num_classes=60):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Transformer层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 输出分类
        self.fc = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.view(N, T, V * C)  # 调整为 (N, T, V*C)

        # Transformer处理
        x = self.transformer_encoder(x)

        # 进行池化和分类
        x = x.mean(dim=1)  # 对时间维度求平均
        x = self.fc(x)  # 分类层

        return x


# HMM模型
class HMM_Model:
    def __init__(self, n_components=5):
        self.model = hmm.GaussianHMM(n_components=n_components)

    def fit(self, X):
        """
        训练HMM模型
        :param X: 训练数据，形状为 (N, T, D)，N为样本数，T为时间步长，D为特征维度
        """
        X = X.reshape(-1, X.shape[-1])  # 转换为 (N*T, D)
        self.model.fit(X)

    def predict(self, X):
        """
        使用训练好的HMM模型进行预测
        :param X: 测试数据
        :return: 状态预测
        """
        X = X.reshape(-1, X.shape[-1])  # 转换为 (N*T, D)
        return self.model.predict(X)


# LightGBM分类器
class LightGBM_Classifier:
    def __init__(self, params=None):
        self.model = lgb.LGBMClassifier(**(params if params else {}))

    def train(self, X_train, y_train):
        """
        训练LightGBM模型
        :param X_train: 训练数据，特征
        :param y_train: 标签
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        使用训练好的LightGBM模型进行预测
        :param X_test: 测试数据
        :return: 预测结果
        """
        return self.model.predict(X_test)


# 完整的集成模型
class FinalModel(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(FinalModel, self).__init__()

        # 图结构初始化
        if graph is None:
            raise ValueError("Graph structure must be provided.")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # 元过程划分网络（骨架序列划分）
        self.meta_process_segment = meta_process_segmentation

        # Transformer用于处理每个元过程
        self.transformer = TransformerEncoder(d_model=256, num_heads=8, num_layers=6, num_classes=num_class)

        # HMM模型用于运动共性建模
        self.hmm_model = HMM_Model(n_components=5)

        # LightGBM分类器
        self.lightgbm_classifier = LightGBM_Classifier()

    def forward(self, x):
        # 元过程划分
        segments = self.meta_process_segment(x)

        # 用Transformer处理每个元过程
        transformer_outputs = []
        for segment in segments:
            output = self.transformer(segment)
            transformer_outputs.append(output)

        # 假设transformer_outputs是 (N, num_class)
        transformer_outputs = torch.stack(transformer_outputs)

        # 将Transformer输出与HMM模型结合，进行状态预测
        hmm_outputs = []
        for i in range(transformer_outputs.size(0)):
            hmm_outputs.append(self.hmm_model.predict(transformer_outputs[i].cpu().numpy()))

        # 假设hmm_outputs是 (N, T)
        # 特征融合，最后输入LightGBM分类器进行人体行为分类
        final_features = torch.cat(hmm_outputs, dim=1)
        lgbm_result = self.lightgbm_classifier.predict(final_features)

        return lgbm_result


# 定义import_class函数，用于动态导入图结构（Graph）
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

