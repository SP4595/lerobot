from lerobot.scripts.control_robot import control_robot
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.control_configs import RecordControlConfig


### 设置 ###
repo_id = 'DATASET/so101_test'
root = './'
############

robot_config = RobotConfig()

control_config = RecordControlConfig(
    repo_id= repo_id, # Dataset identifier, 一串 string。表示数据集在
    fps=30,
    tags = 'so101',
    
)

config = ControlPipelineConfig(
    robot=robot_config,
    control=control_config
)

control_robot(
    cfg = config
)