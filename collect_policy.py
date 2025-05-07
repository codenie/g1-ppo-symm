import os, sys, time, pdb
import numpy as np
import shutil


def copy_and_rename_file(source_path:str, destination_dir:str, new_name:str=None):
    """
    复制Python文件并重命名为a.py
    参数:
    source_path: 源文件路径
    destination_dir: 目标目录
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            print(f"错误：源文件 {source_path} 不存在")
            return
        
        # 检查目标目录是否存在，如果不存在则创建
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print(f"已创建目标目录: {destination_dir}")
        
        # 目标文件路径
        destination_path = os.path.join(destination_dir, new_name)
        
        # 复制文件
        shutil.copy2(source_path, destination_path)
        print(f"文件已成功复制并重命名为: {destination_path}")
        
    except PermissionError:
        print("错误：权限不足，无法执行文件操作")
    except Exception as e:
        print(f"发生错误：{str(e)}")



### TODO: 待完成

copy_and_rename_file()