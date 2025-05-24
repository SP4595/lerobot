import os
import shutil
import logging

# 配置 logging（如果尚未配置）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_backup(dir: str, backup_path: str, replace: bool):
    """
    将 dir 目录备份到 backup_path 下。

    :param dir: 要备份的源目录路径
    :param backup_path: 备份目标根路径
    :param replace: 是否替换已存在的备份目录 (True: 替换；False: 抛出异常)
    """
    # 获取要备份的目录名
    dir_name = os.path.basename(os.path.normpath(dir))
    target_dir = os.path.join(backup_path, dir_name)

    # 判断目标目录是否存在
    if os.path.exists(target_dir):
        if replace:
            logging.info(f"已有备份目录 {target_dir}，正在删除并重新创建...")
            shutil.rmtree(target_dir)
        else:
            raise FileExistsError(f"备份目录 {target_dir} 已存在，且 replace=False，操作已取消。")

    # 执行复制
    logging.info(f"正在备份目录 {dir} 到 {target_dir}")
    shutil.copytree(dir, target_dir)
    logging.info("备份完成")