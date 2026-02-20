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

# 用于路径存在性校验（规避生成无法寻路的 episode）
try:
    import habitat_sim
    HAS_HABITAT_SIM = True
except ImportError:
    HAS_HABITAT_SIM = False


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


def find_valid_position_along_perp(
    pathfinder,
    center: np.ndarray,
    distance: float,
    perp_sign: float,
    perp_vec: np.ndarray,
    door_direction: np.ndarray,
    door_offset: float = 0.0,
    height: float = 0.16441,
    max_attempts: int = 50
) -> Optional[np.ndarray]:
    """
    在「垂直于门」的直线上找一个合法位置：center + distance * (perp_sign * perp_vec) + door_offset * door_direction。
    用于 perpendicular 采样：人/机器人起始点连线与门垂直。
    perp_sign: +1 或 -1，表示在门的哪一侧。
    door_offset: 沿门方向的偏移（米），用于在同一垂直线上多点尝试。
    """
    for attempt in range(max_attempts):
        offset = distance * (perp_sign * perp_vec) + door_offset * door_direction
        pos = np.array([center[0] + offset[0], height, center[2] + offset[2]])
        snapped = pathfinder.snap_point(pos)
        if pathfinder.is_navigable(snapped):
            dist_actual = np.linalg.norm(np.array([snapped[0], snapped[2]]) - np.array([center[0], center[2]]))
            if abs(dist_actual - distance) < 0.3:
                return snapped
        door_offset += (attempt + 1) * 0.05 * (1 if attempt % 2 == 0 else -1)
    return None


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


def path_exists(pathfinder, start, end) -> bool:
    """
    检查两点之间是否存在可行走路径。
    用于生成 episode 时规避「起点/终点可站立但无路径」的情况（避免运行时 find_path 失败）。
    """
    if not HAS_HABITAT_SIM:
        return True  # 未安装 habitat_sim 时跳过校验
    try:
        start_arr = np.array(start, dtype=np.float32)
        end_arr = np.array(end, dtype=np.float32)
        if start_arr.size != 3 or end_arr.size != 3:
            return False
        path = habitat_sim.ShortestPath()
        path.requested_start = start_arr.reshape(3)
        path.requested_end = end_arr.reshape(3)
        return pathfinder.find_path(path)
    except Exception:
        return False


def generate_episode_from_door(
    base_episode: dict,
    door_start: List[float],
    door_end: List[float],
    pathfinder,
    episode_id: int,
    distance: Optional[float] = None,
    height: float = 0.16441,
    sample_method: str = "circle",
    radius_min: float = 1.5,
    radius_max: float = 2.0
) -> dict:
    """
    基于门的位置生成一个episode
    
    Args:
        base_episode: 基础episode模板
        door_start: 门起点 [x, y, z]
        door_end: 门终点 [x, y, z]
        pathfinder: habitat pathfinder对象
        episode_id: episode ID
        distance: 距离门中点的距离（米）。None 表示在 radius_min–radius_max 之间随机，本 episode 内统一
        height: 高度，默认0.16441
        sample_method: "circle" = 在门两侧圆上随机采样；"perpendicular" = 人/机器人起始点连线与门垂直
        radius_min, radius_max: 半径范围（米），distance 为 None 时使用。默认 1.5–2.0
    
    Returns:
        新的episode字典
    """
    np.random.seed(episode_id * 1000)
    if distance is None:
        distance = float(np.random.uniform(radius_min, radius_max))

    door_middle = get_door_middle(door_start, door_end)
    door_direction = get_door_direction(door_start, door_end)
    perp_vec = np.array([-door_direction[2], 0, door_direction[0]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    perp_vec_2d = np.array([perp_vec[0], perp_vec[2]])

    robot_start = None
    human_start = None
    robot_goal = None
    human_goal = None

    if sample_method == "perpendicular":
        # ========== 垂直采样：人/机器人起始点连线与门垂直 ==========
        max_start_attempts = 100
        for attempt in range(max_start_attempts):
            door_offset = np.random.uniform(-0.3, 0.3)
            robot_start_candidate = find_valid_position_along_perp(
                pathfinder, door_middle, distance, 1.0, perp_vec, door_direction, door_offset, height
            )
            human_start_candidate = find_valid_position_along_perp(
                pathfinder, door_middle, distance, -1.0, perp_vec, door_direction, door_offset, height
            )
            if robot_start_candidate is not None and human_start_candidate is not None:
                robot_start = robot_start_candidate
                human_start = human_start_candidate
                break

        max_goal_attempts = 100
        for attempt in range(max_goal_attempts):
            door_offset = np.random.uniform(-0.3, 0.3)
            robot_goal_candidate = find_valid_position_along_perp(
                pathfinder, door_middle, distance, -1.0, perp_vec, door_direction, door_offset, height
            )
            human_goal_candidate = find_valid_position_along_perp(
                pathfinder, door_middle, distance, 1.0, perp_vec, door_direction, door_offset, height
            )
            if robot_goal_candidate is None or human_goal_candidate is None:
                continue
            if not path_exists(pathfinder, robot_start, robot_goal_candidate):
                continue
            if not path_exists(pathfinder, human_start, human_goal_candidate):
                continue
            robot_goal = robot_goal_candidate
            human_goal = human_goal_candidate
            break

    else:
        # ========== 圆形生成（原方法）==========
        max_start_attempts = 100
        for attempt in range(max_start_attempts):
            angle_robot = np.random.uniform(0, 2 * math.pi)
            angle_human = np.random.uniform(0, 2 * math.pi)
            robot_start_candidate = find_valid_position_on_circle(
                pathfinder, door_middle, distance, angle_robot, perp_vec, door_direction, height
            )
            human_start_candidate = find_valid_position_on_circle(
                pathfinder, door_middle, distance, angle_human, perp_vec, door_direction, height
            )
            if robot_start_candidate is None or human_start_candidate is None:
                continue
            robot_vec = np.array([robot_start_candidate[0] - door_middle[0], robot_start_candidate[2] - door_middle[2]])
            human_vec = np.array([human_start_candidate[0] - door_middle[0], human_start_candidate[2] - door_middle[2]])
            robot_side = np.dot(robot_vec, perp_vec_2d)
            human_side = np.dot(human_vec, perp_vec_2d)
            if robot_side * human_side < 0 and abs(np.linalg.norm(robot_vec) - np.linalg.norm(human_vec)) <= 0.3:
                robot_start = robot_start_candidate
                human_start = human_start_candidate
                break

        max_goal_attempts = 100
        for attempt in range(max_goal_attempts):
            angle_robot = np.random.uniform(0, 2 * math.pi)
            angle_human = np.random.uniform(0, 2 * math.pi)
            robot_goal_candidate = find_valid_position_on_circle(
                pathfinder, door_middle, distance, angle_robot, perp_vec, door_direction, height
            )
            human_goal_candidate = find_valid_position_on_circle(
                pathfinder, door_middle, distance, angle_human, perp_vec, door_direction, height
            )
            if robot_goal_candidate is None or human_goal_candidate is None:
                continue
            robot_vec = np.array([robot_goal_candidate[0] - door_middle[0], robot_goal_candidate[2] - door_middle[2]])
            human_vec = np.array([human_goal_candidate[0] - door_middle[0], human_goal_candidate[2] - door_middle[2]])
            robot_side = np.dot(robot_vec, perp_vec_2d)
            human_side = np.dot(human_vec, perp_vec_2d)
            if robot_side * human_side < 0:
                if robot_start is not None and human_start is not None:
                    robot_start_vec = np.array([robot_start[0] - door_middle[0], robot_start[2] - door_middle[2]])
                    robot_start_side = np.dot(robot_start_vec, perp_vec_2d)
                    if robot_side * robot_start_side > 0:
                        continue
                if abs(np.linalg.norm(robot_vec) - np.linalg.norm(human_vec)) <= 0.3:
                    if not path_exists(pathfinder, robot_start, robot_goal_candidate):
                        continue
                    if not path_exists(pathfinder, human_start, human_goal_candidate):
                        continue
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
    
    # 生成随机旋转（前两个值为0，后两个值随机）
    # 旋转格式: [x, y, z, w] (quaternion)
    # 我们只随机化z轴旋转（yaw角）（种子已在函数开头设置，此处沿用同一随机序列）
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
