'''
键位：
- 1. `ESC`: 停止当前 episode，保存，后续的 episode 也不继续测了

- 2. `键盘右键`: 停止当前 episode，保存。然后等一下开始新 episode

- 3. `键盘左键`: rerecord_episode，停止当前episode并重启。因为只有这样才能保证错误数据不会被写入


### 设置 ###
robot = 'so101'
control_type = 'record'
repo_id = 'DATASET/so101_test' # 如果有 root了，这个也要看！
root = './SO101Datasets/so101_test'
single_task = "Grasp and handle a key." # 任务
tags = ["so101"] # 可以不加，如果要push到hugging face上最好加
fps = 30
warmup_time_s = 5 # 启动后等 5 秒再开始
episode_time_s = 30 # 一个 episode 最大 30 个
reset_time_s = 30 # 每次收集完数据等 30 秒钟
num_episodes = 10 # 收集多少个 episode
push_to_hub = False 
############
'''

import subprocess
import json
import sys


class DataCollector:
    """
    用于控制机器人并收集数据的封装类。
    
    该类将参数构建成标准命令行调用格式，并在系统终端中执行。
    支持 Windows、Linux 等多个平台。

    参数说明：
    -----------
    robot : str
        机器人型号名称，例如 'so101'
    control_type : str
        控制类型，例如 'record' 表示录制模式
    repo_id : str
        数据集存储路径标识符 (Hugging Face 风格）
    root : str
        数据集本地保存路径
    single_task : str
        当前任务描述字符串
    tags : list of str
        标签列表，用于上传 Hugging Face 时使用
    fps : int
        每秒帧数（采集频率）
    warmup_time_s : int
        启动后等待时间（单位：秒）
    episode_time_s : int
        单个 episode 的最大持续时间（单位：秒）
    reset_time_s : int
        每次 episode 结束后等待时间（单位：秒）
    num_episodes : int
        要采集的 episode 数量
    push_to_hub : bool
        是否将数据集推送到 Hugging Face Hub
    """

    def __init__(
        self,
        robot: str,
        control_type: str,
        repo_id: str,
        root: str,
        single_task: str,
        tags: list,
        fps: int,
        warmup_time_s: int,
        episode_time_s: int,
        reset_time_s: int,
        num_episodes: int,
        push_to_hub: bool,
    ):
        # 初始化所有参数为实例变量
        self.robot = robot
        self.control_type = control_type
        self.repo_id = repo_id
        self.root = root
        self.single_task = single_task
        self.tags = tags
        self.fps = fps
        self.warmup_time_s = warmup_time_s
        self.episode_time_s = episode_time_s
        self.reset_time_s = reset_time_s
        self.num_episodes = num_episodes
        self.push_to_hub = push_to_hub

    def build_command(self) -> list:
        """
        构建要执行的命令行参数列表
        
        返回:
        -------
        list of str
            完整的命令行参数列表（可用于 subprocess.run)
        """
        cmd = ["python", "lerobot/scripts/control_robot.py"]

        args = {
            "robot.type": self.robot,
            "control.type": self.control_type,
            "control.fps": self.fps,
            "control.single_task": self.single_task,
            "control.repo_id": self.repo_id,
            "control.tags": json.dumps(self.tags),  # 将标签转为 JSON 字符串
            "control.warmup_time_s": self.warmup_time_s,
            "control.episode_time_s": self.episode_time_s,
            "control.reset_time_s": self.reset_time_s,
            "control.num_episodes": self.num_episodes,
            "control.push_to_hub": str(self.push_to_hub).lower(),  # 转换为小写字符串
            "control.root": self.root
        }

        for key, value in args.items():
            if value is not None:
                cmd.append(f"--{key}={value}")

        return cmd

    def run(self):
        """
        执行构建好的命令行指令
        
        使用 subprocess.run 来运行命令，根据操作系统决定是否启用 shell=True
        并输出执行结果状态码
        """
        command = self.build_command()

        # Windows 上需要开启 shell=True 来正确识别引号等特殊字符
        use_shell = sys.platform == "win32"

        print("🚀 正在运行命令：")
        print(" ".join(command))

        result = subprocess.run(
            command,
            shell=use_shell,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            print("✅ 数据收集完成！")
        else:
            print("❌ 数据收集失败！返回码:", result.returncode)


# 示例用法
if __name__ == "__main__":
    collector = DataCollector(
        robot="so101",
        control_type="record",
        repo_id="SO101Datasets/so101_snack",
        root="./SO101Datasets/so101_snack",
        single_task="Place the snacks in the bowl.",
        tags=["so101"],
        fps=30,
        warmup_time_s=10,
        episode_time_s=30,
        reset_time_s=30,
        num_episodes=10,
        push_to_hub=False,
    )
    collector.run()