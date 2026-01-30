#!/usr/bin/env python3
"""
根据门的位置生成episode的核心函数
只供play_rvo_agent.py调用，不提供命令行接口
"""

import json
import gzip
import numpy as np
import copy
import math
from typing import List, Optional

# 尝试导入 magnum，用于处理 Vector3
try:
    import magnum as mn
    HAS_MAGNUM = True
except ImportError:
    HAS_MAGNUM = False


def load_json_dataset(json_path: str) -> dict:
    """加载JSON数据集"""
    if json_path.endswith('.gz'):
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_json_dataset(data: dict, output_path: str):
    """保存JSON数据集"""
    if output_path.endswith('.gz'):
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存到: {output_path}")


def get_door_middle(door_start: List[float], door_end: List[float]) -> np.ndarray:
    """计算门的中点"""
    return np.array([
        (door_start[0] + door_end[0]) / 2,
        door_start[1],  # 使用door_start的y坐标
        (door_start[2] + door_end[2]) / 2
    ])


def get_door_direction(door_start: List[float], door_end: List[float]) -> np.ndarray:
    """获取门的方向向量（归一化）"""
    direction = np.array([
        door_end[0] - door_start[0],
        0,
        door_end[2] - door_start[2]
    ])
    norm = np.linalg.norm(direction)
    if norm > 0:
        return direction / norm
    return np.array([1.0, 0.0, 0.0])


def to_list(pos):
    """将位置转换为列表格式（处理Vector3、numpy数组等）"""
    if pos is None:
        return None
    
    # 检查是否是 Vector3 类型
    if HAS_MAGNUM and isinstance(pos, mn.Vector3):
        return [float(pos.x), float(pos.y), float(pos.z)]
    
    # 检查是否有 x, y, z 属性（Vector3 的特征）
    if hasattr(pos, 'x') and hasattr(pos, 'y') and hasattr(pos, 'z'):
        return [float(pos.x), float(pos.y), float(pos.z)]
    
    # 处理 numpy 数组
    if isinstance(pos, np.ndarray):
        return pos.tolist()
    
    # 处理列表和元组
    if isinstance(pos, (list, tuple)):
        return [float(x) for x in pos]
    
    # 尝试索引访问
    try:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except (TypeError, IndexError, KeyError):
        return pos


def find_valid_position_on_circle(
    pathfinder,
    center: np.ndarray,
    radius: float,
    angle: float,
    perp_vec: np.ndarray,
    door_direction: np.ndarray,
    height: float = 0.16441,
    max_attempts: int = 50
) -> Optional[np.ndarray]:
    """
    在圆上找到一个合法的位置
    
    Args:
        pathfinder: habitat pathfinder对象
        center: 圆心位置（门中点）[x, y, z]
        radius: 圆的半径（米）
        angle: 角度（弧度），相对于垂直于门方向的向量
        perp_vec: 垂直于门方向的向量（归一化）
        door_direction: 门的方向向量（归一化）
        height: 高度
        max_attempts: 最大尝试次数
    
    Returns:
        合法的位置 [x, y, z] 或 None
    """
    for attempt in range(max_attempts):
        # 在圆上生成点
        # 使用perp_vec作为主要方向，door_direction作为次要方向
        offset_x = radius * (math.cos(angle) * perp_vec[0] + math.sin(angle) * door_direction[0])
        offset_z = radius * (math.cos(angle) * perp_vec[2] + math.sin(angle) * door_direction[2])
        
        pos = np.array([
            center[0] + offset_x,
            height,
            center[2] + offset_z
        ])
        
        # 尝试snap到最近的可导航点
        snapped_pos = pathfinder.snap_point(pos)
        
        # 检查是否可导航
        if pathfinder.is_navigable(snapped_pos):
            # 检查距离是否合理（允许一些误差）
            dist = np.linalg.norm(np.array([snapped_pos[0], snapped_pos[2]]) - np.array([center[0], center[2]]))
            if abs(dist - radius) < 0.3:  # 允许0.3米的误差
                return snapped_pos
        
        # 如果失败，稍微调整角度
        angle += (attempt + 1) * 0.1
    
    return None


def generate_episode_from_door(
    base_episode: dict,
    door_start: List[float],
    door_end: List[float],
    pathfinder,
    episode_id: int,
    distance: float = 1.0,
    height: float = 0.16441
) -> dict:
    """
    基于门的位置生成一个episode
    
    Args:
        base_episode: 基础episode模板
        door_start: 门起点 [x, y, z]
        door_end: 门终点 [x, y, z]
        pathfinder: habitat pathfinder对象
        episode_id: episode ID
        distance: 距离门中点的距离（米），默认2.7
        height: 高度，默认0.16441
    
    Returns:
        新的episode字典
    """
    # 计算门的中点
    door_middle = get_door_middle(door_start, door_end)
    door_direction = get_door_direction(door_start, door_end)
    
    # 计算垂直于门方向的向量（用于在门的两边生成位置）
    perp_vec = np.array([-door_direction[2], 0, door_direction[0]])  # 旋转90度
    perp_vec = perp_vec / np.linalg.norm(perp_vec)  # 归一化
    
    # ========== 新方法：圆形生成 ==========
    # 使用随机种子确保可重复性
    np.random.seed(episode_id * 1000)
    
    # 判断点在门的哪一边：计算点到门中点的向量与perp_vec的点积
    # 正数表示在perp_vec方向，负数表示在-perp_vec方向
    
    # 生成起始位置（在门的一边）
    max_start_attempts = 100
    robot_start = None
    human_start = None
    
    for attempt in range(max_start_attempts):
        # 随机生成角度
        angle_robot = np.random.uniform(0, 2 * math.pi)
        angle_human = np.random.uniform(0, 2 * math.pi)
        
        # 在圆上生成位置
        robot_start_candidate = find_valid_position_on_circle(
            pathfinder, door_middle, distance, angle_robot, perp_vec, door_direction, height
        )
        human_start_candidate = find_valid_position_on_circle(
            pathfinder, door_middle, distance, angle_human, perp_vec, door_direction, height
        )
        
        if robot_start_candidate is None or human_start_candidate is None:
            continue
        
        # 检查是否在门的同一侧（通过点积判断）
        # 只使用x和z坐标（忽略y高度）
        robot_vec = np.array([robot_start_candidate[0] - door_middle[0], robot_start_candidate[2] - door_middle[2]])
        human_vec = np.array([human_start_candidate[0] - door_middle[0], human_start_candidate[2] - door_middle[2]])
        perp_vec_2d = np.array([perp_vec[0], perp_vec[2]])  # 只取x和z分量
        
        robot_side = np.dot(robot_vec, perp_vec_2d)  # 正数表示在perp_vec方向
        human_side = np.dot(human_vec, perp_vec_2d)
        
        # 确保在门的两边（起始位置）：robot_side和human_side符号相反
        if robot_side * human_side < 0:  # 符号相反表示在门的两边
            # 检查距离是否相近（差别小于0.3米）
            robot_dist = np.linalg.norm(robot_vec)
            human_dist = np.linalg.norm(human_vec)
            if abs(robot_dist - human_dist) <= 0.3:
                robot_start = robot_start_candidate
                human_start = human_start_candidate
                break
    
    # 生成目标位置（在门的另一边）
    max_goal_attempts = 100
    robot_goal = None
    human_goal = None
    
    for attempt in range(max_goal_attempts):
        # 随机生成角度
        angle_robot = np.random.uniform(0, 2 * math.pi)
        angle_human = np.random.uniform(0, 2 * math.pi)
        
        # 在圆上生成位置
        robot_goal_candidate = find_valid_position_on_circle(
            pathfinder, door_middle, distance, angle_robot, perp_vec, door_direction, height
        )
        human_goal_candidate = find_valid_position_on_circle(
            pathfinder, door_middle, distance, angle_human, perp_vec, door_direction, height
        )
        
        if robot_goal_candidate is None or human_goal_candidate is None:
            continue
        
        # 检查是否在门的另一侧（与起始位置相反）
        # 只使用x和z坐标（忽略y高度）
        robot_vec = np.array([robot_goal_candidate[0] - door_middle[0], robot_goal_candidate[2] - door_middle[2]])
        human_vec = np.array([human_goal_candidate[0] - door_middle[0], human_goal_candidate[2] - door_middle[2]])
        perp_vec_2d = np.array([perp_vec[0], perp_vec[2]])  # 只取x和z分量
        
        robot_side = np.dot(robot_vec, perp_vec_2d)  # 正数表示在perp_vec方向
        human_side = np.dot(human_vec, perp_vec_2d)
        
        # 确保在门的两边（目标位置）：robot_side和human_side符号相反
        # 且与起始位置在不同侧（如果起始位置已确定）
        if robot_side * human_side < 0:  # 符号相反表示在门的两边
            # 如果起始位置已确定，确保目标位置与起始位置在不同侧
            if robot_start is not None and human_start is not None:
                robot_start_vec = np.array([robot_start[0] - door_middle[0], robot_start[2] - door_middle[2]])
                robot_start_side = np.dot(robot_start_vec, perp_vec_2d)
                # 确保机器人目标与起始在不同侧
                if robot_side * robot_start_side > 0:
                    continue  # 在同一侧，跳过
            # 检查距离是否相近（差别小于0.3米）
            robot_dist = np.linalg.norm(robot_vec)
            human_dist = np.linalg.norm(human_vec)
            if abs(robot_dist - human_dist) <= 0.3:
                robot_goal = robot_goal_candidate
                human_goal = human_goal_candidate
                break
    
    # 如果找不到合法位置，使用随机可导航点
    if robot_start is None:
        robot_start = pathfinder.get_random_navigable_point()
        print(f"  警告: 无法找到合法的机器人起始位置，使用随机点")
    if human_start is None:
        human_start = pathfinder.get_random_navigable_point()
        print(f"  警告: 无法找到合法的人的起始位置，使用随机点")
    if robot_goal is None:
        robot_goal = pathfinder.get_random_navigable_point()
        print(f"  警告: 无法找到合法的机器人目标位置，使用随机点")
    if human_goal is None:
        human_goal = pathfinder.get_random_navigable_point()
        print(f"  警告: 无法找到合法的人的目标位置，使用随机点")
    
    # ========== 旧方法：直线生成（已注释保留） ==========
    # # 生成3个不同的配置（每个episode使用不同的偏移）
    # # 在垂直于门方向的直线上，人和机器人稍微错开
    # offsets = [
    #     (-0.3, 0.3),   # episode 0: 机器人稍微左，人稍微右
    #     (-0.5, 0.5),   # episode 1: 机器人更左，人更右
    #     (-0.2, 0.2),   # episode 2: 机器人稍微左，人稍微右
    # ]
    # 
    # offset_idx = episode_id % 3
    # robot_offset, human_offset = offsets[offset_idx]
    # 
    # # 在门的一边生成起始位置（使用正距离，即perp_vec方向）
    # # 先找到基础位置，然后添加小偏移
    # base_start_pos = door_middle + distance * perp_vec
    # base_start_pos[1] = height  # 设置高度
    # base_start_pos = pathfinder.snap_point(base_start_pos)
    # 
    # # 添加沿着门方向的小偏移，使人和机器人稍微错开
    # robot_start = base_start_pos + robot_offset * door_direction
    # robot_start = pathfinder.snap_point(robot_start)
    # 
    # human_start = base_start_pos + human_offset * door_direction
    # human_start = pathfinder.snap_point(human_start)
    # 
    # # 在门的另一边生成目标位置（使用负距离，即-perp_vec方向）
    # base_goal_pos = door_middle - distance * perp_vec
    # base_goal_pos[1] = height  # 设置高度
    # base_goal_pos = pathfinder.snap_point(base_goal_pos)
    # 
    # robot_goal = base_goal_pos + robot_offset * door_direction
    # robot_goal = pathfinder.snap_point(robot_goal)
    # 
    # human_goal = base_goal_pos + human_offset * door_direction
    # human_goal = pathfinder.snap_point(human_goal)
    # 
    # # 验证位置是否可导航
    # if not pathfinder.is_navigable(robot_start):
    #     robot_start = pathfinder.get_random_navigable_point()
    #     print(f"  警告: 机器人起始位置不可导航，使用随机点")
    # if not pathfinder.is_navigable(human_start):
    #     human_start = pathfinder.get_random_navigable_point()
    #     print(f"  警告: 人的起始位置不可导航，使用随机点")
    # if not pathfinder.is_navigable(robot_goal):
    #     robot_goal = pathfinder.get_random_navigable_point()
    #     print(f"  警告: 机器人目标位置不可导航，使用随机点")
    # if not pathfinder.is_navigable(human_goal):
    #     human_goal = pathfinder.get_random_navigable_point()
    #     print(f"  警告: 人的目标位置不可导航，使用随机点")
    
    # 生成随机旋转（前两个值为0，后两个值随机）
    # 旋转格式: [x, y, z, w] (quaternion)
    # 我们只随机化z轴旋转（yaw角）
    np.random.seed(episode_id * 1000)  # 使用episode_id作为种子，确保可重复
    z_angle = np.random.uniform(0, 2 * math.pi)
    robot_rot = [0.0, 0.0, math.sin(z_angle / 2), math.cos(z_angle / 2)]
    
    z_angle_human = np.random.uniform(0, 2 * math.pi)
    human_rot = [0.0, 0.0, math.sin(z_angle_human / 2), math.cos(z_angle_human / 2)]
    
    # 创建新episode（深拷贝避免修改原始数据）
    new_episode = copy.deepcopy(base_episode)
    new_episode['episode_id'] = episode_id
    
    # 更新位置和旋转信息
    new_episode['start_position'] = to_list(robot_start)
    new_episode['start_rotation'] = robot_rot
    
    # 更新info
    new_episode['info']['human_start'] = to_list(human_start)
    new_episode['info']['human_goal'] = to_list(human_goal)
    new_episode['info']['robot_goal'] = to_list(robot_goal)
    new_episode['info']['door_start'] = door_start
    new_episode['info']['door_end'] = door_end
    new_episode['info']['human_rot'] = human_rot
    
    return new_episode
