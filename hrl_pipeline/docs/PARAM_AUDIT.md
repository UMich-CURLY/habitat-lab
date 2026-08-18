# 全栈阈值参数审计表(2026-08-04)

分类:
- **[T] teacher 策略参数** — 只影响 teacher 行为;僵硬是设计意图(论文里"学习超越规则"的空间),保留
- **[I] 共享基础设施参数** — teacher 和 student 共用;过紧会制造对双方都无解的场景,是数据集空间的隐形边界
- **[S] sensor/数据门控参数** — 决定演示/特征怎么产生

## 1. Teacher(hrl_pipeline/scripted_hl_eval.py)

| 参数 | 位置 | 值 | 类 | 说明 |
|---|---|---|---|---|
| `DIST_T`(RULE_DIST) | :26 | 3.0 m | [T] | conflict 触发距离 |
| bearing 门 | :107 | \|b\|<1.75 rad | [T] | 人在前方扇区 |
| rel_heading 门 | :108 | \|rh\|>π/2 | [T] | 人朝向我们 |
| **rel_speed 门** | :113 | >0.1 m/s | [T] | **死锁根因之一:用的量是相对速度(见 5.1),人停住时=机器人自身速度,导致 go↔backoff 极限环** |
| smart_wait 速度门 | :116 | 0.15 | [T] | |
| passed 门垂距 | :133 | 0.3 m | [T] | 当前对动作无效(死逻辑) |
| recede 单调 eps | :135 | +1e-3 | [T] | 全部 1813 条矛盾标签的来源 |
| recede 释放拍数 | :144 | 3 | [T] | 极限环的周期设定者 |
| RULE_EVALS / seed | :30/:249 | 8 / 7 | [T] | |

## 2. LL skills(habitat-baselines/.../skills/social_nav_skills.py)

| 参数 | 位置 | 值(v2 覆盖) | 类 | 说明 |
|---|---|---|---|---|
| GoToGoal `max_speed` | :541 | 1.0 m/s | [I] | 前进速度上限 |
| GoToGoal `turn_thresh` | :542 | 0.1(0.3) | [I] | 先转后走门限 |
| GoToGoal `goal_stop_radius` | :546 | 0.15 | [I] | 停止强制运动 |
| GoToGoal `goal_done_radius` | :548 | 0.3 | [I] | 交还 HL |
| GoToGoal `min_turn_rate` | :555 | 0(0.35) | [I] | |
| GoToGoal 人距检查 | :626-650 | **不存在** | — | **故意不加(注释明示);lin 永远≥0,无避让/无倒退。"避人半径太大"假设被证伪** |
| Yield `max_back_speed` | :202 | 1.0 | [I] | |
| Yield `start_stop/done_radius` | :206/:208 | 0.15/0.3 | [I] | |
| Yield/GoToGoal `lin_speed` 除数 | :201/:540 | 10.0 | [I] | 须等于 BaseVel longitudinal_lin_speed |
| `_TURN_GAIN` | :161 | 2.0 | [I] | 90° 饱和 |
| LearnedYield lidar 量程/归一 | :391/:415 | 3.0 / 6.0,±1.5 | [I][S] | |
| `max_skill_steps` | baselines cfg :237 | 200 | [I] | |

## 3. 导航/人驱动(oracle_social_nav_actions.py + two_agent_social_nav_task.py)

| 参数 | 位置 | 值 | 类 | 说明 |
|---|---|---|---|---|
| `firm_mode` | actions:473, v2 yaml:46 | true(agent_1) | [I] | 人被挡时冻结 |
| **`block_clearance`** | actions:475, yaml:47 | 0.8 m | [I] | 人"被物理挡住"半径;机器人进 0.8m 前方锥 → 人冻得更死,加深僵局 |
| blocked 锥半角 | actions:524 | ±75°(cos>0.2588) | [I] | 硬编码 |
| `block_hysteresis_steps` | actions:478 | 5 | [I] | |
| waypoint `dist_thresh` | actions:578, yaml | 0.1 | [I] | |
| ORCA `rvo_agent_radius` | task:72 | 0.2 | [I] | 人的自身盘 |
| ORCA `rvo_agent_0_radius` | task:102, v2:27 | 0.25(旧 yaml 0.5) | [I] | 机器人在人 ORCA 里的盘;有效避让距离=0.45m(v2) |
| **`_rvo_goal_stop_radius`** | task:92 | 0.3,**硬编码无 config** | [I] | 人到 goal 0.3m 内 pref-vel 清零 → 人停死;"人停在门口"场景的制造者 |
| `rvo_reciprocity` | task:116, v2:29 | 0(v2) | [I] | 人完全不让机器人 |
| `rvo_resume_speed_floor` | task:128, v2:30 | 0.3 | [I] | |
| RVO lookahead | task:30 | 1.0 s | [I] | |

## 4. 几何/成功判定(configs)

| 参数 | 位置 | 值 | 类 | 说明 |
|---|---|---|---|---|
| 机器人 navmesh 半径 | agents/spot.yaml:6 | 0.25 | [I] | 实际用 HSSD 烘焙 navmesh(`navmesh_agent_radius=None`) |
| 人 navmesh 半径 | agents/human.yaml:6 | 0.3 | [I] | **人不在 navmesh 里**——geodesic 会径直穿过停住的人 |
| 机器人体足迹 | v2 yaml:59 `navmesh_offset` | [[0,0],[0.225,0]] | [I] | 碰撞钳制用 |
| `success_distance` | twoagent.yaml:128 | 0.4 | [I] | social_nav_to_pos_succ |
| `end_on_collide` | cfg :1474 默认 true;overfit.yaml:20 false | — | [I] | eval 配置下撞人=判负 |
| HL `control_interval` | hierarchical.yaml:121 | 30 步(1 Hz) | [I] | 决策粒度 |
| `max_episode_steps` | hierarchical.yaml:43 | 1500(STEP_CAP 覆盖) | [I] | |
| sensor dt | ac_freq_ratio 4 / ctrl_freq 120 | 1/30 s | [I][S] | |

## 5. Sensors(social_nav_sensors.py)

| 参数 | 位置 | 值 | 类 | 说明 |
|---|---|---|---|---|
| **`human_rel_speed` 定义** | :1627-1631 | ‖Δhuman−Δrobot‖/dt | [S] | **根因:相对速度模长,无符号且混入自车运动;人停住时=机器人速度。修复=新增第 7 维 human_approach_speed(人绝对速度向机器人方向投影)** |
| `backoff_dist` | :985 | 2.0 | [S] | 袋搜索锚距 |
| `obstacle_clearance` | :986 | 0.3(松弛 0.15) | [S] | |
| **`corridor_clearance`** | :1178 | 1.1(松弛 0.825) | [S] | 离人类最短路走廊 |
| **`human_goal_clearance`** | :1183 | 1.5 m | [S] | **人停在自己 goal 时,走廊退化为点+此约束拒掉停人附近所有候选 → 退让点被推极远(观察到"退很远"的放大器);legacy fallback 里仍强制** |
| 候选扇/半径 | :1109-1144 | ±150°,0.5-3.5m | [S] | |
| `recompute_human_motion` | :1309 | 0.75 m | [S] | |
| spawn fallback | :1233-1242 | 无验证 | [S] | **静默垃圾演示来源,本轮删除** |
| 奖励 `safe_dis_min` | :213, yamls | 1.0-1.5 | [S] | |
| 奖励 `yield_dis` | :224, overfit:26 | 3.0 | [S] | |
| lidar | :1470-1472 | 16 rays / 3.0m | [S] | |

## 6. 死代码地雷

| 位置 | 内容 | 处置 |
|---|---|---|
| social_nav_neural_policy.py:535-597 | dist<0.5 或 closing>0.5 强制 backoff 的规则覆盖;观测 key 不存在,从不触发,但正是 ping-pong 逻辑 | 本轮删除 |

## 几何可行性下界(点场景前心算)

机器人沿走廊通过一个停住的人,可行 ⟺
```
缝隙宽(人体边缘到墙) ≥ 机器人体宽 0.45 m(navmesh_offset 0.225×2)+ 余量
```
- navmesh 半径 0.25 已烘焙进可走区域,geodesic 不会贴墙 0 距离;
- 人**不在** navmesh/geodesic 里 → go_to_goal 会直线穿人,end_on_collide=true 时物理接触即判负 → 人半径 ~0.3 m 要另算:人体中心到机器人路径的横距需 ≥ 0.3+0.225 ≈ 0.53 m;
- 人停在 human_goal 上 → 按 build 配方 human_goal 应在门后 4-6 m,若点得太靠门(<1 m),门口缝隙大概率 < 下界 → 死题,勿选。

## 结论
死锁不是 [I] 参数过紧,而是 [T]/[S] 层的信号语义错误(rel_speed)。[I] 参数本轮**不动**(go_to_goal 无避让 → 保留为 A 类失败空间);唯一接近危险的 [I] 是 `_rvo_goal_stop_radius` 硬编码+`firm_mode` 组合(人到 goal 即永久冻结),它定义了"停人"场景的物理形态——选场景时用上面的下界排除死题,而不是改参数。
