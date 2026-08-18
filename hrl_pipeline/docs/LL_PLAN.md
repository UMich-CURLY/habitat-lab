# LL(low-level)训练战役计划 — 2026-08-15

目标:把 LL 从脚本换成学生网络(先 Yield,后 GoToGoal),终局 = 全学习栈
(HL 学生 + LL 学生)不劣于脚本 LL 栈;同时检验 id2/3/25 碰撞族是否因
LL 自由度(让行口袋方向是 LL 决定的,HL 够不着)而松动。

判据与 HL 战役同构:retention / paired efficiency / capability,
对照 = NR1@1.5M(最优 HL + 脚本 LL),一次只换一个因素。

## 已有基建(2026-08-15 侦察)

- `LearnedYieldSkill`(social_nav_skills.py:357):lidar16+feat4+door2 → MLP
  128×128 → [lin,ang];`LL_MODEL` env 换权重;`LL_DAGGER` env = 学生驾驶时
  免费记 teacher 标签(特权 yield 传感器照常算 wp_delta);HL 训练时 LL 冻结。
- 脚本 `ClearCorridorYieldSkill` 带 `LL_LOG` 逐步演示采集(:226)。
- `bc_ll_yield.py`(22→128→128→2,tanh,归一化持久化)+ 旧 `ll_demos.jsonl`
  15352 行(出处/时代待认证)+ `es_ll_finetune.py`(ES 微调驱动,状态待查)。
- `FlatNavSkill(LearnedYieldSkill)`:GoToGoal 的学习版接口(第二期)。

## 必须带上的教训(逐条对应)

1. 窗口指标不可信 → 结论只认 evals×3 确定性评测(seed7, cap1200)
2. 禁 cherry-pick → 固定预算定停 + 预注册评测点
3. 单遍认证不可靠 → 干净 = 成功∧零碰撞∧重复一致;历史证据取并集
4. flaky 隔离训练/判据,终评放回
5. 演示干净可追溯:LL_LOG 先 `rm -f`(append 坑);段-stats 对齐验证;
   失败段剔除并回灌重分类
6. duration-coupled reward 会被 hack → LL 若上 RL,沿用已验证三层奖励
7. 熵奖励融化尖锐策略 → entropy 0
8. critic 冷启动拆台 → warmup + critic_lr 放大(LL PPO 若开)
9. 锚压制新能力 → 不加锚,短预算防遗忘
10. 新 loss 项必配梯度 probe(ratio+cos)
11. 上采样被 per-env split 稀释 → rollout 占比必实测;23 场景禁 SPLIT_SCENES=0(OOM)
12. DISABLE_CUDNN=1 保持
13. 链脚本 exit 守卫:训练不完整绝不产表;权重路径缺失=静默随机初始化,必先 ls
14. L0 探针先行,阈值按实测尺度定(release 0.01>0.008 之鉴)
15. 机制快检(小集短预算)先于正式跑
16. 一次一因素,对照臂固定
17. 维度探测 all() 坑 → 逐行 assert 22 维
18. 评测永远在固定全集(train36_v1/holdout2)上做
19. 串行单 GPU 进程 + 后台哨兵 + 死机可恢复(结果落盘即安全)

## Phase 0 结论(2026-08-16 盘点完成)

- **最短换 LL 路径**(无需新 yaml,当前栈 overfit_v2 已带 lidar_scan):
  `RULE_EXTRA="$SK.skill_name=LearnedYieldSkill +$SK.skill_data.model_path=<pth> +$SK.skill_data.lidar_max_range=3.0"`,
  其中 `SK=...hierarchical_policy.defined_skills.backoff`;对照臂 = 去掉
  RULE_EXTRA。全学习栈再追加 `+...pretrained_hl_weights=nr1_ck14_hl.pth`。
  (hierarchical_policy.py:96 `eval(skill_name)`;LL_MODEL env 也可覆盖。)
- **历史证据(7 月,旧 teacher,数字不可与 NR1 并列)**:纯 BC 灾难
  (succ 0.306/coll 0.694)→ +DAgger 0.861 ≈ teacher → **ES 0.882 超特权脚本**
  (cd17)。阶梯有效,flat-nav(GoToGoal 学生化)当时全超时失败。
- **旧演示判死**(全部重采):6 集/4 场景覆盖不足;rule_yield 旧 teacher 代际
  (无 wait 分布);8 月环境变更(branch 传感器 3→6 维、corridor_clearance
  0.5、human_stop_dist 冻结分支);append 不可追溯、无 ep 字段无法剔除失败段。
  旧权重仅两用途:Phase 0 基线读数 + ES warm start。
- **DAgger 标签 bug(Phase 3 前必修)**:LearnedYieldSkill._teacher_action
  (social_nav_skills.py:460-480)硬编码 v2 参数且**缺 human_stop_dist=1.0
  冻结分支** → 近距离教"继续倒车";需对齐脚本 LL(:268-274)并从 skill_data
  读参。DAgger 行格式与 demos 兼容(cat 合并),但按来源分文件记账。
- **Phase 1 干净采集用两遍法**(_new_teacher_chain 成熟做法):先跑拿
  EVAL_STATS → 构造 success-only 集 → 在其上采,天然整遍干净;同时给
  LL_LOG/DAgger 行加 ep/t 字段(或改用 flat_collect.py 式 step 补丁,唯一
  带 episode 对齐的采集器)。
- es_ll_finetune.py:ES 默认只扰最后一层(258 参),fitness =
  succ − 0.5·coll − 0.1·steps/cap,子进程评测,8 并发,禁与 PPO 抢卡。

## Phase 0|盘点与基线(半天,CPU 为主)

- 摸清 wiring:哪个 yaml/override 挂 LearnedYieldSkill 与 lidar_scan;
  es_ll_finetune.py 的机制与历史结果;FlatNavSkill 状态。
- 认证旧资产:ll_demos.jsonl 是哪个 teacher/场景集采的、有无 append 污染;
  旧 ll_yield_bc.pth 若在 → 直接闭环评一遍(换 LearnedYieldSkill,其余全同,
  HL=teacher)拿"旧学生"起点读数,决定续用还是重采重训。
- LL 侧对照基线 = NR1@1.5M 终表(已有,不重跑)。

## Phase 1|干净演示重采(~1 天)

- 驱动栈 = 当前 wait-teacher HL + 脚本 LL,在 stableok 28 集(23 场景)采
  `LL_LOG`;先 `rm -f`;配 EVAL_STATS;该遍非干净成功的 episode 整段剔除
  (对齐方式 Phase 0 确认:LL 日志是 env-step 粒度)。
- 量级目标 ≥3× 旧 15k 行;检查 lin/ang 分布(防全零/塌缩);逐行维度 assert。

## Phase 2|BC + L0 闭环探针(半天)

- bc_ll_yield.py 重训;开环验收(val MSE/动作分布)只是门票。
- **L0 闭环对拍**(真验收):l0set 5 集,脚本 LL vs 学生 LL(HL=teacher,
  其余全同)逐集 succ/coll/steps + 复用 analyze_yield_geometry 看让行几何
  是否同形。BC 复合误差(OOD 漂移)在此暴露。

## Phase 3|DAgger 迭代(1–2 轮,每轮 ~2h)

- `LL_DAGGER` 基建现成:学生驾驶、teacher 标签免费落在学生访问态上。
- 聚合重训 → 闭环评;判据:vs 脚本 LL paired(succ 不降、coll 不升、
  steps 差≤噪声)。两轮后仍显著劣 → 停,报告差距结构再议。

## Phase 4|全学习栈判据(半天)

- 最优 HL(nr1_ck14)+ 学生 LL:train36 evals×3 + holdout2 → by_group
  三判据,对照 NR1@1.5M。
- 重点:id2/3/25 是否松动(1/3 flaky 级信号也记录)。

## Phase 5|(门控)LL 微调

- 仅当 Phase 4 = "可用但差一口气"才开。顺序:es_ll_finetune.py(ES,无梯度,
  基建已有)→ 需要时才是 LL PPO(连续动作,本 repo 未验证的新水域,须先
  机制快检)。奖励沿用三层新奖励;教训 6-10 全套。

## 并行安排

GPU 现在空闲:先挂 **HL 多 seed 复核**(2-3 seed × 1.5M,~3.5h/seed 串行)
——LL Phase 0/1 是 CPU/代码工作,互不抢;Phase 2 需要 GPU 时 seeds 已完。
