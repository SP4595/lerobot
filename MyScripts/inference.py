from lerobot.scripts.control_robot import record, _init_rerun
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.robots.configs import So101RobotConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
import logging
import torch
import time

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


####
POLICY_PATH = "./outputs/train/act-so101-v3/checkpoints/060000/pretrained_model"
inference_time_s = 60
fps = 30
device = "cuda"
####

'''
如何使用 Lerobot 更泛化的推理？（使用 lerobot 作为 底层 SDK 的方法）
'''

# 1. 加载policy
policy = ACTPolicy.from_pretrained(POLICY_PATH)
policy.to(device)
policy.eval()

# 2. 根据config加载robot
robot_config = So101RobotConfig()
robot : Robot = make_robot_from_config(robot_config) # 构建机器人对象（包括底层方法）
robot.connect() # 连接 robot

# 3. Loop
total_iterations = inference_time_s * fps

logging.info("Inference start")

for i in range(total_iterations): # 一共这么多回合
    
    start_time = time.perf_counter() # 高精度计数
    
    # 1. 获取机器人的observation
    
    observation = robot.capture_observation()
    
    # Convert to pytorch format: channel first and float32 in [0,1]
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255 # uint8 -> float32
            observation[name] = observation[name].permute(2, 0, 1).contiguous() # 从 CV 中的 channel 在最后，转到深度学习中 channel 在最前
        observation[name] = observation[name].unsqueeze(0) # 添加 batch 维度
        observation[name] = observation[name].to(device)
        
    # policy 推理获得动作 （一般是关节角）
    # 不用 toch.no_grad 了， select action 内部已经设定了！
    action = policy.select_action(observation)
    
    # 取消 batch 维度
    action = action.squeeze(0)
    
    # Move to cpu, if not already the case
    action = action.cpu()
    
    # 机器人 step (这样机器人和 policy 就分离了！)
    robot.send_action(action) 
    
    end_time = time.perf_counter()
    dt_s = end_time - start_time
    
    busy_wait(1 / fps - dt_s) # 保证按固定帧率（例如 30 FPS）运行推理和控制机器人动作（如果快了就减慢到 30 FPS）
    
    logging.info(f"iteration {i}/{total_iterations}, inference time: {dt_s:.4f}/{1 / fps:.4f}")

# 取消连接
robot.disconnect()

logging.info("Inference end")