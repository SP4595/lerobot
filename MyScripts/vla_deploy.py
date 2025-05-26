'''
推理和数据采集其实是共用框架的！

### 设置 ###
robot = 'so101'
control_type = 'record'
repo_id = 'DATASET/so101_test' # 如果有 root了，这个也要看！
root = './SO101Datasets/so101_test'
single_task = "Grasp and handle a key." # 任务
tags = ["so101"] # 可以不加，如果要push到hugging face上最好加
fps = 30
warmup_time_s = 5 # 整个数据收集启动后等 5 秒再开始（注意，只有最开始的episode受影响！）
episode_time_s = 30 # 一个 episode 最大 30 秒钟
reset_time_s = 20 # 每次收集完数据等 30 秒钟 （注意，每个episode都会受影响！）
num_episodes = 10 # 收集多少个 episode
push_to_hub = False 

policy.path : 推理用模型位置
############
'''

import subprocess
import json
import sys
import os


class VLAInfer:
    """
    用于控制机器人测试并收集测试数据的封装类。
    
    该类将参数构建成标准命令行调用格式，并在系统终端中执行。
    支持 Windows、Linux 等多个平台。

    参数说明：
    -----------
    policy_path : 
        推理用的模型位置
    robot : str
        机器人型号名称，例如 'so101'
    control_type : str
        控制类型，例如 'record' 表示录制模式
    repo_id : str
        测试产生的数据集 标签 (必须有，但是可以留空为 NAN/NAN)
    root : str
        车市产生的数据集本地保存路径
    single_task : str
        当前任务描述字符串
    tags : list of str
        标签列表，用于上传 Hugging Face 时使用
    fps : int
        每秒帧数（采集频率）
    warmup_time_s : int
        开始收集数据后等待时间（单位：秒）
    episode_time_s : int
        单个 episode 的最大持续时间（单位：秒）
    reset_time_s : int
        每次 episode 结束后等待时间（单位：秒）
    num_episodes : int
        要采集的 episode 数量
    push_to_hub : bool
        是否将数据集推送到 Hugging Face Hub
    resume: bool
        是否在已有数据集的基础上添加数据
    """

    def __init__(
        self,
        robot: str,
        control_type: str,
        root:str,
        policy_path: str,
        single_task: str,
        tags: list,
        fps: int,
        warmup_time_s: int,
        episode_time_s: int,
        reset_time_s: int,
        num_episodes: int,
        push_to_hub: bool,
        resume : bool,
        repo_id : str = 'nan/eval_act' # 必须是 eval 开头，必须是一个 /！
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
        self.resume = resume
        self.policy_path = policy_path
        
    def __process_bool(self, boolean : bool):
        '''
        # bool 转换为小写字符串
        '''
        return str(boolean).lower()

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
            "control.push_to_hub": self.__process_bool(self.push_to_hub),
            "control.resume": self.__process_bool(self.resume),
            "control.root": self.root,
            "control.policy.path":self.policy_path
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
    import shutil
    buffer_dir = "./outputs/test_record/act-so101-v3/checkpoints/test"
    
    if os.path.exists(buffer_dir):
        shutil.rmtree(buffer_dir)
    
    collector = VLAInfer(
        robot="so101",
        control_type="record",
        root = buffer_dir,
        policy_path="./outputs/train/act-so101-v3/checkpoints/080000/pretrained_model", # 开始推理
        single_task="Pick up the snack and place it in the bowl.",
        tags=["so101"],
        fps=30,
        warmup_time_s=2,
        episode_time_s=30,
        reset_time_s=2,
        num_episodes=10,
        push_to_hub=False,
        resume = False # 是否在已有数据集的基础上添加数据
    )
    
    collector.run()