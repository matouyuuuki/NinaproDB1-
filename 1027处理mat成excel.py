import os
import numpy as np
import scipy.io as sio
import pandas as pd
from openpyxl import Workbook


def find_all_occurrence_ranges(restimulus):
    """
    找到restimulus中每个有效动作（≥1）的所有连续出现范围
    返回格式：{动作编号: [(start1, end1), (start2, end2), ...]}
    """
    ranges_dict = {}
    n = len(restimulus)
    if n == 0:
        return ranges_dict

    current_action = None
    start_idx = 0

    for i in range(n):
        val = restimulus[i]
        # 有效动作（≥1的整数）
        if isinstance(val, (int, np.integer)) and val >= 1:
            if val != current_action:
                # 新动作开始，保存上一个动作的范围（如果存在）
                if current_action is not None:
                    if current_action not in ranges_dict:
                        ranges_dict[current_action] = []
                    ranges_dict[current_action].append((start_idx, i - 1))
                # 更新当前动作和起始索引
                current_action = val
                start_idx = i
        else:
            # 无效值，结束当前动作范围（如果存在）
            if current_action is not None:
                if current_action not in ranges_dict:
                    ranges_dict[current_action] = []
                ranges_dict[current_action].append((start_idx, i - 1))
                current_action = None

    # 处理最后一个动作的范围
    if current_action is not None:
        if current_action not in ranges_dict:
            ranges_dict[current_action] = []
        ranges_dict[current_action].append((start_idx, n - 1))

    return ranges_dict


def extract_emg_data(emg, ranges_dict, local_to_global):
    """
    提取指定动作范围的EMG数据（自定义列范围）
    emg: 原始EMG数据（二维数组：行×列）
    ranges_dict: 本地动作→片段范围的字典
    local_to_global: 本地动作→全局动作的映射字典
    """
    extracted = {}
    emg = np.array(emg)

    if emg.ndim < 2:
        raise ValueError("EMG数据格式错误，应为二维数组（行×列）")

    # -------------------------- 自定义列范围配置 --------------------------
    col_start = 0  # 第1列对应的索引（Python是0-based）
    col_end = 12  # 第12列对应的索引（切片左闭右开，所以取到12即包含0-11索引）
    # 如需修改列范围，直接改上面两个值即可，例如：
    # 第3-15列：col_start=2, col_end=15
    # 第2-8列：col_start=1, col_end=8
    # ---------------------------------------------------------------------
    columns = slice(col_start, col_end)

    for local_num, ranges in ranges_dict.items():
        global_num = local_to_global.get(local_num)
        if global_num is None:
            continue  # 跳过未映射的本地动作
        if global_num not in extracted:
            extracted[global_num] = []
        for (start, end) in ranges:
            # 提取指定行范围和列范围的EMG数据
            segment_data = emg[start:end + 1, columns]
            extracted[global_num].append(segment_data)

    return extracted


def save_to_excel(all_files_data, output_file):
    """
    将所有文件的EMG数据按全局动作分工作表保存到Excel
    """
    # 计算最大列数（从第一个有效片段动态获取，避免硬编码）
    max_cols = 0
    for file_data in all_files_data.values():
        for segments in file_data.values():
            if segments:
                max_cols = segments[0].shape[1]
                break
        if max_cols > 0:
            break

    # 生成列名（通道1到通道N，N为实际提取的列数）
    columns = [f'通道{j + 1}' for j in range(max_cols)] if max_cols > 0 else ['通道1']

    # 获取排序后的全局动作编号
    sorted_actions = sorted([action for file_data in all_files_data.values() for action in file_data.keys()])
    sorted_actions = list(dict.fromkeys(sorted_actions))  # 去重并保持顺序

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for action in sorted_actions:
            sheet_name = f'动作{action}'
            all_data = []

            # 遍历所有文件，收集该动作的所有片段
            for file_idx, (file_name, file_data) in enumerate(all_files_data.items(), 1):
                action_segments = file_data.get(action, [])
                if not action_segments:
                    continue

                # 添加文件标识行
                all_data.append(pd.DataFrame(
                    [[f'文件{file_idx}（{os.path.basename(file_name)}）'] + [''] * (max_cols - 1)],
                    columns=columns
                ))

                # 添加每个片段的数据
                for i, seg in enumerate(action_segments):
                    # 添加片段标识行
                    all_data.append(pd.DataFrame(
                        [[f'  片段{i + 1}'] + [''] * (max_cols - 1)],
                        columns=columns
                    ))
                    # 添加片段数据
                    seg_df = pd.DataFrame(seg, columns=columns)
                    all_data.append(seg_df)
                    # 片段之间添加空行（最后一个片段除外）
                    if i != len(action_segments) - 1:
                        all_data.append(pd.DataFrame(index=[0], columns=columns))

                # 文件之间添加空行（最后一个文件除外）
                if file_idx != len(all_files_data):
                    all_data.append(pd.DataFrame(index=[0], columns=columns))

            # 合并数据并保存
            if all_data:
                final_df = pd.concat(all_data, ignore_index=True)
            else:
                final_df = pd.DataFrame(columns=columns)
            final_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"已保存 全局动作{action} 的汇总数据（工作表：{sheet_name}）")

    print(f"\n所有数据已成功保存到：{output_file}")


def main():
    # -------------------------- 配置参数 --------------------------
    mat_folder = r"C:\NinaproDB2\s2"  # 替换为你的MAT文件文件夹路径
    output_excel = r"C:\NinaproDB2\excel\DB2s2训练结果.xlsx"   # 输出Excel文件名及路径
    # ---------------------------------------------------------------------

    # 检查MAT文件夹是否存在
    if not os.path.exists(mat_folder):
        print(f"错误：MAT文件文件夹不存在 → {mat_folder}")
        return

    all_files_data = {}  # 存储所有文件的处理结果：{文件名: {全局动作号: [片段1, 片段2, ...]}}
    global_action_offset = 0  # 全局动作编号偏移量（避免不同文件动作编号冲突）

    # 遍历文件夹中的所有MAT文件
    mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]
    if not mat_files:
        print("错误：指定文件夹中未找到MAT文件")
        return
    print(f"找到 {len(mat_files)} 个MAT文件，开始处理...")

    for mat_file in sorted(mat_files):  # 按文件名排序处理
        file_path = os.path.join(mat_folder, mat_file)
        print(f"\n处理文件：{mat_file}")

        try:
            # 读取MAT文件
            mat_data = sio.loadmat(file_path)

            # 检查必要字段是否存在
            if 'emg' not in mat_data or 'restimulus' not in mat_data:
                print(f"警告：{mat_file} 缺少'emg'或'restimulus'字段，跳过该文件")
                continue

            emg = mat_data['emg']
            restimulus = mat_data['restimulus'].flatten()  # 转换为一维数组

            # 找到所有有效动作的片段范围
            ranges_dict = find_all_occurrence_ranges(restimulus)
            if not ranges_dict:
                print(f"警告：{mat_file} 中未找到有效动作片段，跳过该文件")
                continue
            print(f"  找到 {len(ranges_dict)} 个本地动作，片段范围：{ranges_dict}")

            # 构建本地动作→全局动作的映射（避免不同文件动作编号冲突）
            local_actions = sorted(ranges_dict.keys())
            local_to_global = {local: global_action_offset + idx + 1 for idx, local in enumerate(local_actions)}
            print(f"  本地动作→全局动作映射：{local_to_global}")

            # 提取EMG数据
            file_emg_data = extract_emg_data(emg, ranges_dict, local_to_global)
            if not file_emg_data:
                print(f"警告：{mat_file} 未提取到有效EMG数据，跳过该文件")
                continue

            # 保存当前文件的处理结果
            all_files_data[file_path] = file_emg_data
            print(f"  成功提取 {len(file_emg_data)} 个全局动作的EMG数据")

            # 更新全局动作偏移量（确保下一个文件的动作编号不冲突）
            global_action_offset += len(local_actions)

        except Exception as e:
            print(f"错误：处理 {mat_file} 时发生异常 → {str(e)}，跳过该文件")
            continue

    # 保存到Excel
    if all_files_data:
        save_to_excel(all_files_data, output_excel)
    else:
        print("\n所有文件处理完成，但未提取到任何有效数据，无法生成Excel文件")


if __name__ == "__main__":
    main()