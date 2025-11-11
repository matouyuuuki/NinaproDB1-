import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation,
                                     LSTM, Bidirectional, Concatenate, GlobalAveragePooling1D,
                                     Dense, Dropout, Add, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
import os  # 用于路径处理


def load_multi_sheet_data(file_path, window_size=200, overlap=100):
    """读取多工作表Excel数据，生成滑动窗口样本（保持不变）"""
    xls = pd.ExcelFile(file_path)
    all_windows = []
    all_labels = []

    for label, sheet_name in enumerate(xls.sheet_names):
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = df.dropna(axis=1, how='all')
        df = df.apply(pd.to_numeric, errors='coerce')
        data = df.values

        n_features = data.shape[1] if data.ndim == 2 else 0
        groups = []
        current_group = []
        for row in data:
            if not np.all(np.isnan(row)):
                current_group.append(row)
            else:
                if current_group:
                    groups.append(np.array(current_group))
                    current_group = []
        if current_group:
            groups.append(np.array(current_group))

        def generate_windows(groups, label):
            windows = []
            labels = []
            step = window_size - overlap
            for group in groups:
                n_samples = group.shape[0]
                if n_samples >= window_size and group.shape[1] > 0:
                    for i in range(0, n_samples - window_size + 1, step):
                        window = group[i:i + window_size, :]
                        if not np.any(np.isnan(window)):
                            windows.append(window)
                            labels.append(label)
            if len(windows) == 0:
                return np.empty((0, window_size, n_features)), np.array([])
            return np.array(windows), np.array(labels)

        win, lab = generate_windows(groups, label)
        if win.size > 0 and win.ndim == 3:
            all_windows.append(win)
            all_labels.append(lab)

    X = np.concatenate(all_windows, axis=0) if all_windows else np.array([])
    y = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return X, y


def channel_attention_module(input_feature, ratio=8):
    """通道注意力模块（保持不变）"""
    x = Reshape((-1, 1))(input_feature)
    avg_pool = GlobalAveragePooling1D()(x)
    channel = input_feature.shape[-1]
    fc = Dense(channel // ratio, activation='relu', use_bias=False)(avg_pool)
    fc = Dense(channel, activation='sigmoid', use_bias=False)(fc)
    return input_feature * fc


def build_model(input_shape, num_classes):
    """模型构建（保持不变）"""
    inputs = Input(shape=input_shape)

    # 第一流
    x1 = Conv1D(64, kernel_size=3, padding='causal', name='tcn1_1')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(64, kernel_size=3, padding='causal', name='tcn1_2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(64, kernel_size=3, padding='causal', name='tcn1_3')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(64, kernel_size=3, dilation_rate=2, padding='same', name='tcn1_dilated')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    stream1 = GlobalAveragePooling1D()(x1)

    # 第二流
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
    stream2 = GlobalAveragePooling1D()(x2)

    # 第三流
    x3 = Bidirectional(LSTM(64, return_sequences=True), name='bilstm3')(inputs)
    x3 = Conv1D(64, kernel_size=5, padding='causal', name='tcn3_1')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv1D(64, kernel_size=5, padding='causal', name='tcn3_2')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    stream3 = GlobalAveragePooling1D()(x3)

    # 第四流
    residual = Conv1D(64, kernel_size=1, padding='same', name='residual_proj')(inputs)
    x4 = Conv1D(64, kernel_size=3, padding='same', name='tcn4_1')(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv1D(64, kernel_size=3, padding='same', name='tcn4_2')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv1D(64, kernel_size=3, padding='same', name='tcn4_3')(x4)
    x4 = BatchNormalization()(x4)
    x4 = Add()([x4, residual])
    x4 = Activation('relu')(x4)
    stream4 = GlobalAveragePooling1D()(x4)

    # 融合与分类
    fused = Concatenate()([stream1, stream2, stream3, stream4])
    fused = channel_attention_module(fused)
    outputs = Dense(128, activation='relu')(fused)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(num_classes, activation='softmax')(outputs)

    return Model(inputs=inputs, outputs=outputs)


def process_single_file(file_path, window_size=200, overlap=100, epochs=128, batch_size=32, n_splits=10):
    """处理单个Excel文件，返回交叉验证结果"""
    print(f"\n===== 开始处理文件: {os.path.basename(file_path)} =====")
    X, y = load_multi_sheet_data(file_path, window_size=window_size, overlap=overlap)

    # 数据有效性检查
    if X.size == 0:
        print(f"文件 {os.path.basename(file_path)} 数据为空，跳过处理")
        return None, None
    if len(np.unique(y)) < 2:
        print(f"文件 {os.path.basename(file_path)} 类别不足，跳过处理")
        return None, None

    num_classes = len(np.unique(y))
    input_shape = (X.shape[1], X.shape[2])
    print(f"输入形状: {input_shape}, 类别数: {num_classes}, 总样本数: {X.shape[0]}")

    # 十折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    all_reports = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n----- 第 {fold}/{n_splits} 折 -----")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        y_train_onehot = to_categorical(y_train, num_classes=num_classes)
        y_test_onehot = to_categorical(y_test, num_classes=num_classes)

        model = build_model(input_shape, num_classes)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            X_train, y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test_onehot),
            verbose=1
        )

        y_pred = np.argmax(model.predict(X_test), axis=1)
        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)
        fold_report = classification_report(y_test, y_pred, output_dict=True)
        all_reports.append(fold_report)

        print(f"第 {fold} 折准确率: {fold_acc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n文件 {os.path.basename(file_path)} 十折平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

    # 返回：各折准确率、各折报告、平均准确率、标准差（新增缓存所需的汇总数据）
    return fold_accuracies, all_reports, mean_acc, std_acc


def main():
    # 配置参数 - 可根据需要修改
    data_dir = r"C:\NinaproData\excel"  # 存放Excel文件的文件夹路径
    result_path = r"C:\NinaproData\excel\s11s20.xlsx"  # 最终结果保存路径
    start_idx = 11  # 开始编号（如s2）
    end_idx = 20 # 结束编号（如s10）
    window_size = 200
    overlap = 100
    epochs = 128
    batch_size = 32
    n_splits = 10  # 十折交叉验证

    # 新增：缓存所有文件的处理结果，避免重复运行
    file_results_cache = []

    # 创建Excel写入器
    with pd.ExcelWriter(result_path) as writer:
        # 处理每个文件并缓存结果
        for i in range(start_idx, end_idx + 1):
            file_name = f"s{i}.xlsx"
            file_path = os.path.join(data_dir, file_name)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"文件 {file_name} 不存在，跳过")
                continue

            # 处理单个文件（只运行一次）
            fold_accs, reports, mean_acc, std_acc = process_single_file(
                file_path,
                window_size=window_size,
                overlap=overlap,
                epochs=epochs,
                batch_size=batch_size,
                n_splits=n_splits
            )

            if fold_accs is None or reports is None:
                continue  # 跳过处理失败的文件

            # 缓存当前文件的结果（用于后续汇总）
            file_results_cache.append({
                '文件名': file_name,
                '各折准确率': fold_accs,
                '平均准确率': mean_acc,
                '准确率标准差': std_acc,
                '各折报告': reports
            })

            # 保存当前文件的各折报告
            for fold, report in enumerate(reports, 1):
                sheet_name = f"s{i}_第{fold}折"
                report_df = pd.DataFrame(report).transpose()
                report_df.to_excel(writer, sheet_name=sheet_name)

            # 保存当前文件的汇总结果
            summary_sheet = f"s{i}_汇总"
            summary_data = {
                '折数': list(range(1, n_splits + 1)),
                '准确率': fold_accs
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.loc['平均'] = ['', mean_acc]  # 直接使用缓存的结果
            summary_df.loc['标准差'] = ['', std_acc]  # 直接使用缓存的结果
            summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

        # 生成所有文件的总体汇总表（使用缓存结果，不重复运行模型）
        overall_summary = []
        for cache in file_results_cache:
            overall_summary.append({
                '文件名': cache['文件名'],
                '平均准确率': cache['平均准确率'],
                '准确率标准差': cache['准确率标准差']
            })

        # 保存总体汇总
        if overall_summary:
            overall_df = pd.DataFrame(overall_summary)
            overall_df.to_excel(writer, sheet_name='所有文件汇总', index=False)

    print(f"\n所有文件处理完成，结果已保存至: {result_path}")


if __name__ == "__main__":
    main()