import os

def replace_class(folder_path, old_num, new_num):
    # 遍历文件夹内所有txt标注文件
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, file_name)
        
        # 读取原标注内容
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 替换每行开头的类别编号
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 只修改行首的类别索引
            if parts[0] == str(old_num):
                parts[0] = str(new_num)
            new_lines.append(" ".join(parts) + "\n")
        
        # 写回修正后的内容
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    print(f"✅ {folder_path} 处理完成")

# 处理训练集
replace_class("./dataset/labels/train/football", 0, 1)
replace_class("./dataset/labels/train/obstacle", 0, 2)

# 处理验证集
replace_class("./dataset/labels/val/football", 0, 1)
replace_class("./dataset/labels/val/obstacle", 0, 2)

print("\n🎉 所有标注文件索引修正完成")
