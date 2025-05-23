from lerobot.scripts.control_robot import record, _init_rerun
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.robot_devices.control_configs import RecordControlConfig

'''
键位：
- 1. `ESC`: 停止当前 episode，保存，后续的 episode 也不继续测了

- 2. `键盘右键`: 停止当前 episode，保存。然后等一下开始新 episode

- 3. `键盘左键`: rerecord_episode，停止当前episode并重启。因为只有这样才能保证错误数据不会被写入
'''

### 设置 ###
repo_id = 'DATASET/so101_test' # 如果有 root了，那么这个就没关系了
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

control_config = RecordControlConfig(
    repo_id= repo_id, # Dataset identifier, 一串 string。表示数据集在
    fps=fps,
    tags = tags,
    root = root,
    single_task = single_task,
    warmup_time_s = warmup_time_s,
    episode_time_s = episode_time_s,
    reset_time_s = reset_time_s,
    num_episodes = num_episodes,
    push_to_hub = push_to_hub
)

robot_config = So101RobotConfig()

robot : Robot = make_robot_from_config(robot_config) # 构建机器人对象（包括底层方法）

_init_rerun(control_config=control_config, session_name="lerobot_control_loop_record")

record(
    robot=robot,
    cfg = control_config
)