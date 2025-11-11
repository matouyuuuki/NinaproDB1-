import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                                     LSTM, Bidirectional, Concatenate, GlobalAveragePooling1D,
                                     Dense, Dropout, Add, GlobalAveragePooling2D, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score


def load_multi_sheet_data(file_path, window_size=200, overlap=100):
    """读取多工作表Excel数据，生成滑动窗口样本"""
    xls = pd.ExcelFile(file_path)
    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    for label, sheet_name in enumerate(xls.sheet_names):
        # 读取工作表并预处理
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = df.dropna(axis=1, how='all')  # 移除全空列
        df = df.apply(pd.to_numeric, errors='coerce')  # 转换为数值型
        data = df.values

        # 获取特征数（用于空窗口维度定义）
        n_features = data.shape[1] if data.ndim == 2 else 0

        # 按空行分割数据组（连续非空行为一组）
        groups = []
        current_group = []
        for row in data:
            if not np.all(np.isnan(row)):  # 非空行加入当前组
                current_group.append(row)
            else:
                if current_group:  # 空行时保存当前组（非空）
                    groups.append(np.array(current_group))
                    current_group = []
        if current_group:  # 处理最后一组
            groups.append(np.array(current_group))

        # 划分训练组和测试组（8:2）
        n_groups = len(groups)
        n_train = int(0.8 * n_groups)
        train_groups = groups[:n_train]
        test_groups = groups[n_train:]

        # 生成滑动窗口
        def generate_windows(groups, label):
            windows = []
            labels = []
            step = window_size - overlap
            for group in groups:
                n_samples = group.shape[0]
                # 确保窗口大小小于数据长度且数据有特征
                if n_samples >= window_size and group.shape[1] > 0:
                    for i in range(0, n_samples - window_size + 1, step):
                        window = group[i:i + window_size, :]
                        # 去除含NaN的窗口
                        if not np.any(np.isnan(window)):
                            windows.append(window)
                            labels.append(label)
            # 处理空窗口情况（确保3维：[0, window_size, n_features]）
            if len(windows) == 0:
                return np.empty((0, window_size, n_features)), np.array([])
            return np.array(windows), np.array(labels)

        # 生成训练/测试窗口
        train_win, train_lab = generate_windows(train_groups, label)
        test_win, test_lab = generate_windows(test_groups, label)

        # 只添加非空窗口（确保3维）
        if train_win.size > 0 and train_win.ndim == 3:
            train_windows.append(train_win)
            train_labels.append(train_lab)
        if test_win.size > 0 and test_win.ndim == 3:
            test_windows.append(test_win)
            test_labels.append(test_lab)

    # 合并所有窗口数据
    X_train = np.concatenate(train_windows, axis=0) if train_windows else np.array([])
    y_train = np.concatenate(train_labels, axis=0) if train_labels else np.array([])
    X_test = np.concatenate(test_windows, axis=0) if test_windows else np.array([])
    y_test = np.concatenate(test_labels, axis=0) if test_labels else np.array([])

    return X_train, y_train, X_test, y_test


def channel_attention_module(input_feature, ratio=8):
    """修复：适配2维输入的通道注意力模块"""
    # 输入特征形状：(batch_size, features)
    # 转换为 (batch_size, features, 1) 以适配全局池化
    x = Reshape((-1, 1))(input_feature)

    # 全局平均池化（对特征维度）
    avg_pool = GlobalAveragePooling1D()(x)  # 输出形状：(batch_size, 1)

    # 计算注意力权重
    channel = input_feature.shape[-1]
    fc = Dense(channel // ratio, activation='relu', use_bias=False)(avg_pool)
    fc = Dense(channel, activation='sigmoid', use_bias=False)(fc)  # 输出形状：(batch_size, channel)

    # 特征加权
    return input_feature * fc


def build_model(input_shape, num_classes):
    """构建升级后的四流特征融合模型"""
    inputs = Input(shape=input_shape)

    # -------------------------- 第一流：3层因果卷积 + 1层扩张卷积 --------------------------
    x1 = Conv1D(64, kernel_size=3, padding='causal', name='tcn1_1')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x1 = Conv1D(64, kernel_size=3, padding='causal', name='tcn1_2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x1 = Conv1D(64, kernel_size=3, padding='causal', name='tcn1_3')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # 扩张卷积（感受野扩大）
    x1 = Conv1D(64, kernel_size=3, dilation_rate=2, padding='same', name='tcn1_dilated')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    stream1 = GlobalAveragePooling1D()(x1)  # 输出2维：(batch_size, 64)

    # -------------------------- 第二流：2层1D-CNN(kernel=7) + 2层TCN(kernel=3) --------------------------
    x2 = Conv1D(64, kernel_size=7, padding='same', name='cnn2_1')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv1D(64, kernel_size=7, padding='same', name='cnn2_2')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv1D(64, kernel_size=3, padding='causal', name='tcn2_1')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv1D(64, kernel_size=3, padding='causal', name='tcn2_2')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    stream2 = GlobalAveragePooling1D()(x2)  # 输出2维：(batch_size, 64)

    # -------------------------- 第三流：双向LSTM(64) + 2层TCN(kernel=5) --------------------------
    x3 = Bidirectional(LSTM(64, return_sequences=True), name='bilstm3')(inputs)

    x3 = Conv1D(64, kernel_size=5, padding='causal', name='tcn3_1')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x3 = Conv1D(64, kernel_size=5, padding='causal', name='tcn3_2')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    stream3 = GlobalAveragePooling1D()(x3)  # 输出2维：(batch_size, 64)

    # -------------------------- 第四流：3层TCN + 残差连接 --------------------------
    # 残差投影（匹配维度）
    residual = Conv1D(64, kernel_size=1, padding='same', name='residual_proj')(inputs)

    x4 = Conv1D(64, kernel_size=3, padding='same', name='tcn4_1')(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)

    x4 = Conv1D(64, kernel_size=3, padding='same', name='tcn4_2')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)

    x4 = Conv1D(64, kernel_size=3, padding='same', name='tcn4_3')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Add()([x4, residual])  # 残差连接
    x4 = Activation('relu')(x4)
    stream4 = GlobalAveragePooling1D()(x4)  # 输出2维：(batch_size, 64)

    # -------------------------- 特征融合与分类 --------------------------
    # 拼接四流特征：(batch_size, 64*4=256)
    fused = Concatenate()([stream1, stream2, stream3, stream4])
    fused = channel_attention_module(fused)  # 通道注意力（适配2维输入）

    # 分类头
    outputs = Dense(128, activation='relu')(fused)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(num_classes, activation='softmax')(outputs)

    return Model(inputs=inputs, outputs=outputs)


def main():
    # 配置参数
    file_path = r"C:\NinaproDB2\excel\s1.xlsx"  # 输入Excel路径
    result_path = r"C:\NinaproDB2\excel\DB2s1训练结果.xlsx"  # 结果保存路径
    window_size = 200  # 窗口大小
    overlap = 100  # 重叠率
    epochs = 128
    batch_size = 32

    # 加载数据
    print("加载数据中...")
    X_train, y_train, X_test, y_test = load_multi_sheet_data(
        file_path, window_size=window_size, overlap=overlap
    )

    # 增强数据有效性检查
    if X_train.size == 0:
        print("训练数据为空，请检查Excel文件或调整窗口大小")
        return
    if X_test.size == 0:
        print("测试数据为空，请检查Excel文件或调整窗口大小")
        return
    if len(np.unique(y_train)) < 2:
        print("训练数据类别不足，无法进行分类训练")
        return

    # 标签独热编码
    num_classes = len(np.unique(y_train))
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)

    # 定义输入形状 (时间步, 特征数)
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"输入形状: {input_shape}, 类别数: {num_classes}")
    print(f"训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")

    # 构建模型
    model = build_model(input_shape, num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 训练模型
    print("开始训练...")
    history = model.fit(
        X_train, y_train_onehot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_onehot),
        verbose=1
    )

    # 评估模型
    print("评估模型...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    overall_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"总体准确率: {overall_acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 保存结果到Excel
    print("保存结果...")
    report_df = pd.DataFrame(report).transpose()
    overall_df = pd.DataFrame([{'指标': '总体准确率', '值': overall_acc}])

    with pd.ExcelWriter(result_path) as writer:
        report_df.to_excel(writer, sheet_name='分类报告')
        overall_df.to_excel(writer, sheet_name='总体指标', index=False)

    print(f"结果已保存至: {result_path}")


if __name__ == "__main__":
    main()