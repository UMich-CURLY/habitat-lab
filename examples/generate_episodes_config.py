# 生成 episodes 的配置（只改本文件即可，无需改 play_rvo_agent.py）
# GENERATE_ONLY: True = 只生成数据集并退出（不跑 step）；False = 不生成，直接跑 step
GENERATE_ONLY = True

# 模板数据集路径（用于取 episode 作模板）。相对 habitat-lab 根目录，或绝对路径。None = 默认 scene_wise/test_dataset15_finalest_only_8.json.gz
TEMPLATE_DATASET_PATH = "data/social_nav_episode_0415_570.json.gz"

# 采样方式: "circle" = 在门两侧的圆上随机采样（原方法）；"perpendicular" = 人/机器人起始点连线与门连线垂直
SAMPLE_METHOD = "circle"

# 距离门中点的半径范围（米），本 episode 内统一在该范围内随机一个半径。原为 1.5–2.5，现改为 1.5–2
RADIUS_MIN = 1
RADIUS_MAX = 1.5

# 生成数据集的输出路径。相对 habitat-lab 根目录，或绝对路径。None = 默认 data/data_xinyuan/test_dataset15_generated.json.gz
OUTPUT_PATH = "data/data_xinyuan/test_0220.json.gz"

# 每项：template_episode_index（读入文件中第几号 episode，下标从 0 开始）+ door_pixel_pairs（该模版下的多组门）
# 格式：每组门为 [[门起点 pixel], [门终点 pixel]]，即 [[x1,y1], [x2,y2]]

GENERATE_CONFIG = [

    {"template_episode_index": 1, "door_pixel_pairs": [
        [[723, 591],  [723, 603]],
        [[723, 428], [723, 443]],
        [[504, 572], [518, 572]],
    ]},

    {"template_episode_index": 11, "door_pixel_pairs": [
        [[245, 231], [245, 242]],
    ]},

    {"template_episode_index": 21, "door_pixel_pairs": [
        [[402, 124], [417, 124]],
        [[414, 215], [418, 215]],
    ]},

    {"template_episode_index": 31, "door_pixel_pairs": [
        [[268, 395], [268, 405]],
        [[340, 344], [340, 358]],
    ]},

    {"template_episode_index": 41, "door_pixel_pairs": [
        [[418, 301], [446, 301]],
    ]},

    {"template_episode_index": 51, "door_pixel_pairs": [
        [[558, 427], [577, 427]],
    ]},

    {"template_episode_index": 61, "door_pixel_pairs": [
        [[238, 435], [253, 435]],
        [[76, 436],  [93, 436]],
    ]},

    {"template_episode_index": 71, "door_pixel_pairs": [
        [[307, 744], [351, 744]],
        [[243, 274], [264, 274]],
        [[466, 279], [484, 279]],
        [[528, 280], [541, 280]],
        [[555, 779], [555, 795]],
    ]},

    {"template_episode_index": 81, "door_pixel_pairs": [
        [[149, 150], [137, 166]],
        [[177, 424], [184, 436]],
    ]},

    {"template_episode_index": 91, "door_pixel_pairs": [
        [[652, 444], [666, 445]],
        [[816, 595], [831, 598]],
        [[982, 629], [978, 650]],
    ]},
    # 第二个模版 + 它的门（取消注释使用）
    # {"template_episode_index": 1, "door_pixel_pairs": [
    #     [[500, 300], [500, 350]],
    # ]},


]


