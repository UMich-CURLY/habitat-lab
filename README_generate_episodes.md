# 生成Episode脚本使用说明

## 功能
根据门的位置（door_start和door_end）生成多个episode，每个门生成3个episode。

## 使用方法

### 方法1: 从像素点输入（推荐）

如果你在play_rvo_agent的tensorboard中看到了top_down_map，可以点击两个像素点来指定门的位置：

```bash
cd /home/xinyuan/habicrowd/Simulator/habitat-lab
python3 generate_episodes_from_doors.py \
    --door-start-pixel 300 231 \
    --door-end-pixel 300 243 \
    --output-json scene_wise/test_dataset15_generated.json.gz
```

### 方法2: 直接输入world坐标

如果你已经知道门在world frame中的坐标：

```bash
python3 generate_episodes_from_doors.py \
    --door-start -4.57991 0.0 -10.84772 \
    --door-end -4.57097 0.0 -10.06371 \
    --output-json scene_wise/test_dataset15_generated.json.gz
```

### 方法3: 使用原始episode中的门位置

如果不指定门位置，脚本会使用原始JSON中的门位置：

```bash
python3 generate_episodes_from_doors.py \
    --output-json scene_wise/test_dataset15_generated.json.gz
```

## 参数说明

- `--input-json`: 输入的JSON数据集路径（默认: scene_wise/test_dataset15_finalest_only_8.json.gz）
- `--output-json`: 输出的JSON数据集路径（默认: scene_wise/test_dataset15_generated.json.gz）
- `--config`: Habitat配置文件路径（默认: habitat-lab/habitat/config/benchmark/multi_agent/hssd_fetch_human_social_nav_irl.yaml）
- `--door-start`: 门起点world坐标 [x, y, z]
- `--door-end`: 门终点world坐标 [x, y, z]
- `--door-start-pixel`: 门起点像素坐标 [pixel_x, pixel_y]（从tensorboard top_down_map）
- `--door-end-pixel`: 门终点像素坐标 [pixel_x, pixel_y]（从tensorboard top_down_map）
- `--radius`: 生成位置的圆半径（米），默认2.7
- `--episodes-per-door`: 每个门生成多少个episode，默认3

## 生成逻辑

1. **计算门的中点**: `door_middle = (door_start + door_end) / 2`

2. **生成起始位置**: 在门的一边，以door_middle为圆心，半径为r（默认2.7米）的圆上生成：
   - 机器人起始位置
   - 人的起始位置（稍微偏移避免重叠）

3. **生成目标位置**: 在门的另一边（对面），同样在圆上生成：
   - 机器人目标位置
   - 人的目标位置

4. **旋转设置**: 
   - 前两个值（x, y）固定为0
   - 后两个值（z, w）随机生成（z轴旋转）

5. **位置验证**: 所有生成的位置都会检查是否可导航（不被占用）

6. **每个门生成3个episode**: 使用不同的角度配置，确保多样性

## 输出

脚本会生成一个新的JSON文件，包含：
- 原始episode的所有信息（场景、物体等）
- 新生成的机器人起始位置和目标位置
- 新生成的人的起始位置和目标位置
- 新的门位置
- 新的旋转信息

## 示例输出

```
加载原始JSON: scene_wise/test_dataset15_finalest_only_8.json.gz
加载Habitat配置: habitat-lab/habitat/config/benchmark/multi_agent/hssd_fetch_human_social_nav_irl.yaml
创建Habitat环境...
从像素点转换门位置...
  门起点像素: [300, 231]
  门终点像素: [300, 243]
  门起点world: [-4.57991, 0.0, -10.84772]
  门终点world: [-4.57097, 0.0, -10.06371]
门中点: [-4.57544, 0.0, -10.45572]
半径: 2.7米
每个门生成 3 个episode

开始生成episodes...
生成episode 1/3...
  Episode 0:
    机器人起始: [-1.63043, 0.16441, -9.73827]
    机器人目标: [-7.28582, 0.16441, -10.56872]
    人起始: [-6.80863, 0.16441, -11.42159]
    人目标: [-1.61759, 0.16441, -11.3999]
...

保存新JSON到: scene_wise/test_dataset15_generated.json.gz
完成! 生成了 3 个episodes
```

