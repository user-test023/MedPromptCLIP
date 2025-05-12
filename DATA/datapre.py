import os
import json
import random
from sklearn.model_selection import train_test_split

# 设置随机种子以保证结果可重复
random.seed(42)

# 数据集路径
data_dir = "/old_home/lyt/zxj_workplaces/pro_text/DATA/Knee/data"

# 检查路径是否存在
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"路径 {data_dir} 不存在")

# 获取所有子文件夹
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
categories.sort()  # 排序以保持一致

# 初始化数据集字典
dataset = {
    "train": [],
    "val": [],
    "test": []
}

# 为每个类别处理文件
for category_idx, category in enumerate(categories):
    category_dir = os.path.join(data_dir, category)
    # 获取该类别的所有文件
    files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]
    files.sort()  # 排序以保持一致

    # 检查文件列表是否为空
    if not files:
        print(f"警告：类别 {category} 中没有找到文件")
        continue

        # 随机划分数据集（70% train, 15% val, 15% test）
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # 构建数据列表
    for file_path in train_files:
        dataset["train"].append([
            f"{category}/{file_path}",  # 文件路径（相对于类别文件夹）
            category_idx,  # 标签编号
            category  # 类别名称
        ])

    for file_path in val_files:
        dataset["val"].append([
            f"{category}/{file_path}",
            category_idx,
            category
        ])

    for file_path in test_files:
        dataset["test"].append([
            f"{category}/{file_path}",
            category_idx,
            category
        ])

# 保存为 JSON 文件
output_file = "/old_home/lyt/zxj_workplaces/pro_text/DATA/Knee/data_split.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"JSON 文件已生成：{output_file}")