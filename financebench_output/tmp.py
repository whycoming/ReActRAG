import glob
import os
import shutil

index_dir = r"indexes"
markdown_dir = r"financebench_output\markdown"
target_dir = r"financebench_output\eval\mini_markdown"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

for json_file in glob.glob(os.path.join(index_dir, "*.json")):
    # 获取文件名（不含路径）
    filename = os.path.basename(json_file)
    # 替换扩展名为 .md
    md_filename = os.path.splitext(filename)[0] + ".md"
    # 拼接完整的 markdown 文件路径
    md_file = os.path.join(markdown_dir, md_filename)
    # 目标文件路径
    target_file = os.path.join(target_dir, md_filename)
    
    print(f"JSON文件: {json_file}")
    print(f"对应MD文件: {md_file}")
    print(f"复制到: {target_file}")
    
    # 检查源文件是否存在再复制
    if os.path.exists(md_file):
        shutil.copy(md_file, target_file)
    else:
        print(f"警告: 源文件不存在 - {md_file}")
