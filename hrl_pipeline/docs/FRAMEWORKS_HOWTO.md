# 四条训练路线框架使用手册

本手册对应 2026-07-19 搭好的四条路线框架。目的:你在**大数据集**建好后,按需选择思路直接开跑。所有命令都在容器里执行:

```bash
docker exec -u root wxinyuan bash -lc 'cd /habitat-lab && . activate habitat && <命令>'
```

现有资产(小数据集验证过的基线,选型参照):
- 学习 LL(蒸馏):bcsolid6 上 0.861 ≈ 特权脚本 0.896(统计打平)
- HL 配方(BC→PPO):`checkpoints_multiscene_d20` 脚本LL下 multiscene4 100%
- 全零-oracle 栈(神经HL+学习LL):cleandev17 前身集上 0.75 / 0 撞
- 常用数据集:`data/social_nav_episode_cleandev17_v2.json.gz`(13 SWEET id0-12 + 4 trivial id13-16)、`..._bcsolid6_v2.json.gz`(6 铁集)、`..._multiscene4_v2.json.gz`

---

## 🏁 小数据集选型结果(2026-07-20,已实测)

**结论:路线① ES 微调 LL 是赢家。** 部署栈(神经 HL + 学习 LL,**无任何特权信息、测试期无 teacher**)在 multiscene4 上 **0.75 → 1.00**;在更广的 cleandev17(17 集 / 8 场景)上成功率 **0.71 → 0.88、碰撞减半、更快**。代价:对 LL 最后一层 258 个参数做 ~14 分钟 ES(gen1 就破了)。

**关键协议发现(以后必带):评测必须 `STEP_CAP=1200`。** harness 默认 750 会截断自然较长的 episode——历史上"学习 LL 0.861 vs 脚本 0.896"的差距其实是 **cap 假象**。cap1200 下 bcsolid6 两者逐位相同(都 6/6、同 1027 步),简单门上 LL 根本不是瓶颈。

**瓶颈定位:** 全部无特权差距只压在一个 episode——multiscene4 **ep0(=ep70,0.45m 最窄门)**。teacher HL、cap1200:脚本 LL(特权)4/4 但极限(1199/1200 步),蒸馏 BC-LL 3/4(ep0 差一口气超时)。即 ep0 可解,BC-LL 只是在那道窄门稍慢一点。ES 以闭环结果为目标(fitness 含步数惩罚)正好补上这口气。

| 数据集(评测协议) | 脚本 LL(特权·上界) | BC-LL(蒸馏) | **ES-LL(路线①)** |
|---|---|---|---|
| multiscene4 · teacher HL | 4/4 @1012 步 | 3/4 @1011 | **4/4 @916(更快)** |
| multiscene4 · **部署神经 HL** | — | **3/4 = 0.75** | **4/4 = 1.00**(ep0@1186) |
| bcsolid6(6 集)· teacher HL | 6/6 @1027 | 6/6 @1027 | **6/6 @898(更快)** |
| cleandev17(17 集/8 场景)· teacher HL | — | 0.71 / 撞 0.12 / 968 | **0.88 / 撞 0.06 / 893** |

*(cleandev17 绝对值偏低是因为 teacher 门几何 episodes[0]-only bug 对多门集释放计时错误,但对 BC/ES 一视同仁,比较仍公平——见附录 B①。)*

**其它三条路线(实测/结论,2026-07-20 补全):**
- **路线④ RoundA(HL 适配 LL):无收益。** ep0 是 LL 物理穿门速度不足,HL 重新计时救不了(1.4e5 步与重评都停在 0.75)。仅当出现真正的 HL-LL 时机错配才值得跑。
- **路线② flat(纯 RL 与 BC 双双失败,分层优势的完整证据):**
  - flat 纯 RL:**0.00**(2.5e5 步,躲避-超时局部最优);
  - **flat-BC:0/8**(24 维部署 obs + 目标极坐标,32k 步 teacher 示范,BC val MSE 0.04;上环全 1200 步超时、0 碰撞 = 复合误差漂移,连训练过的场景都到不了点);
  - **flat-BC + ES 微调:拉不动**(32 候选 × 4 代全 -0.1,fitness 无选择信号)。对照分层上 ES 第 1 代即破——**ES 只有配上分层给的"小靶子 + 能走的起点"才有效**。
  - 工具链:`FlatNavSkill` + `always_backoff` 模式 + `flat_collect.py`(VectorEnv.step 补丁记录)+ `bc_flat.py`(位姿差反演动作标签),大数据集可直接复用作论文基线。
- **路线③ topdown(带家具占据图):框架已完全跑通,小数据下无增益。**
  - 传感器目检 ✅(家具障碍块、自我中心旋转、门标记);
  - **init 等价性在环验证 ✅**:v3map + 零初始化 map 列,零训练 = ms4 **1.00**,与 scalar 完全等价;
  - **实测(lr1e-5×1.5e5,ES-LL)**:ms4 保持 **1.00**;cleandev17 **0.353/撞0.588 = 与无图栈几乎逐位相同**——map 列范数仅 0.0097(scalar 列 1.347),保守训练下 CNN 分支没学到东西。4 集/3 场景喂不出"几何→决策"的迁移映射;**map 的验证要等大数据集**(多场景训练 + 场景不相交 heldout)。
  - **两个硬坑(已修/已记)**:① **lr 5e-5 会立刻训崩 map-HL**(随机初始化 CNN 分支让 PPO 在 24 个 update 内把 1.00 打成 0,reward -82→-160;scalar 同 lr 稳定)——**map-HL 必须 lr 1e-5** 或冻结/热身 CNN;② **多智能体 eval 的 ckpt 直载是静默 no-op**(`should_load_agent_state=False`),必须走"抽权重 + `pretrained_hl_weights` + `should_load_ckpt=False`"路径,map 版抽取器 `_extract_hl_map.py`、加载器已支持整头+map 模块;③ 评测时必须 `eval.video_option=[]`,否则 4 通道 map 让渲染崩。

**大数据集推荐配方:** HL-BC+PPO(脚本 LL)→ 蒸馏 LL → **ES 微调 LL(决定性一步)**。ES 不但无特权追平特权脚本,还更快、碰撞更少,且改进能**迁移到部署神经 HL**。最优权重:`hrl_pipeline/ll_yield_es_ms4.pth`。

---

## 路线① LL RL 微调 —— ES 驱动器(已冒烟 ✅)

**是什么**:在 22→128→128→2 的学习 yield MLP 上做黑盒进化搜索(默认只扰最后一层 258 参),直接以**闭环 episode 结果**为目标(fitness = success − 0.5·collide − 0.1·步数占比),突破 BC 的 teacher 天花板。

**什么时候选**:学习 LL 已和 teacher 打平、想**超过** teacher 时;或想在 teacher 根本做不了的集(unsolv 池)上磨 LL 时。

**命令**(小集选型 / 大集正式):
```bash
# 正式跑(约一晚): 6铁集, pop16 x 15代, 8并发
python hrl_pipeline/es_ll_finetune.py \
  --episodes-gz /habitat-lab/data/social_nav_episode_bcsolid6_v2.json.gz \
  --pop 16 --gens 15 --workers 8 --evals 1
# 大数据集: 换 --episodes-gz, 酌情 --evals 2 抗噪, --full 扰全权重
# 管线自检: --smoke (pop2x1代x1集, 几分钟)
```

**输出看哪**:stdout 每代一行 `[es] genK fits: ... | best=... {'success':..,'collide':..}`;最优权重存 `--out`(默认 `hrl_pipeline/ll_yield_es.pth`);中间产物在 `hrl_pipeline/es_ll_*/`。用最优权重评测:`LL_MODEL=<pth> + RULE_CONFIG=social_nav/social_nav_hierarchical_v2_llyield.yaml` 跑 `scripted_hl_eval.py`。

**预期成本**:pop16×15代×6集 ≈ 一晚(8 并发);大集按集数线性放大。
**坑**:勿与任何 PPO 训练同时占卡;fitness 有评测噪声,重要结论用 `--evals 2+`;`--init random` 即"纯RL-LL from scratch"消融(预期弱,报告用)。

---

## 路线② 纯 RL 基线 —— flat 端到端 PPO(已冒烟 ✅)

**是什么**:agent_0 用单个平铺循环策略(PointNavResNetPolicy,连续 gaussian 动作)直接从观测出 base_vel,每步决策,无 HL/技能。配置文件 `social_nav_flat_e2e.yaml`(全局 updater/storage 已切回标准 PPO/RolloutStorage)。

**什么时候选**:作为**支撑分层论点的基线/消融**。先验(HL M2 教训):PPO 从零在此任务陷"躲避-超时"局部最优——本配置就是用来把这个负结果跑实的。

**命令**:
```bash
python -u habitat-baselines/habitat_baselines/run.py \
  --config-name=social_nav/social_nav_flat_e2e.yaml \
  habitat.dataset.data_path=/habitat-lab/data/<你的数据集>.json.gz \
  habitat_baselines.total_num_steps=5.0e6
# eval: 同 config + habitat_baselines.evaluate=True + eval_ckpt_path_dir=checkpoints_flat_e2e/latest.pth
```

**输出看哪**:训练日志 `update:` 行的 `social_nav_to_pos_success`/`did_collide` 窗口均值;终评用 `eval.episode_stats_path` 出逐集 json。
**预期成本**:5M 步一晚量级。
**坑**:agent_0 动作向量含 rearrange_stop / oracle_nav 维(stop 维若被策略触发会提前终止 episode)——跑正式基线前建议把 agent_0 动作裁剪到 base_velocity(在 benchmark yaml 里重定义 `habitat.task.actions` 只留 `agent_0_base_velocity` + agent_1 全部;Hydra 合并不删键,需完整重写 actions 块);深度相机在 obs 里会启用视觉编码器(变慢),不需要可从 obs_keys 去掉。

---

## 路线③ topdown image —— HL 地图分支(S7-lite,已冒烟 ✅)

**是什么**:HL 增加一路自我中心 64×64×4 顶视"冲突图"输入(通道:静态占据 / 人盘+30步渐隐轨迹 / 机器人目标 / 门线),经 ResNet18 编码 64 维,**post-GRU 融合**进 heads。关键设计:6-D GRU 原样保留 + **map 列零初始化** → 初始行为与标量策略完全一致,已验证的 BC(标量权重直接加载)+PPO 配方原封可用。对标组里 S-MEDIRL 的 topdown 输入叙事(角色不同:这里是 HL 的 when-to-yield 输入,非 IRL 奖励图)。

**什么时候选**:标量 6-D 观测怀疑表达不了时机/几何信息时;论文需要"标量 vs +图"消融时。**诚实预期**:小集(十几个 episode)CNN 会背图,打平或小赢即正常;泛化结论必须配场景不相交 heldout。

**命令**:
```bash
# 训练(BC 标量权重热启动, map 分支由 PPO 自己学):
python -u habitat-baselines/habitat_baselines/run.py \
  --config-name=social_nav/social_nav_hierarchical_v3map.yaml \
  habitat.dataset.data_path=/habitat-lab/data/<数据集>.json.gz \
  habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.hidden_dim=32 \
  habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.num_rnn_layers=1 \
  habitat_baselines.rl.ppo.hidden_size=32 \
  +habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.pretrained_hl_weights=<标量HL-BC.pth> \
  habitat_baselines.rl.ppo.lr=5e-5 habitat_baselines.total_num_steps=4.0e5 \
  habitat_baselines.checkpoint_folder=checkpoints_maphl
```

**首次使用必做**:目检地图朝向——跑一次带 `RULE_CONFIG=social_nav/social_nav_hierarchical_v3map.yaml` 的短 eval,在 `TopDownConflictMapSensor.get_observation` 临时加 `np.save('/habitat-lab/hrl_pipeline/map_dump.npy', out)`,画出 4 通道确认"机器人朝向=图上方、门线/人位置对齐"(cv2 旋转符号约定未做全量视觉验证)。
**预期成本**:比标量 HL 训练慢 ~2-4×(CNN);仍在 1 小时量级/4e5 步。
**坑**:传感器静态占据按场景缓存(首步会慢一拍);地图对 agent_1 返回全零;`map_feature_dim` 可调(默认 64)。

---

## 路线④ HL-LL 联合训练 —— 交替最优响应(已冒烟 ✅)

**是什么**:两轮交替:**RoundA** HL 在"冻结的学习 LL"环境里 PPO 微调(LL 参数已 `requires_grad_(False)`,且技能存 plain dict 天然不进优化器/DDP,安全);**RoundB** 适配后的 HL 下再收 LL 的 DAgger 纠错并重训 MLP。这是解决"全栈 0.75 < 各自单独最优"(HL 配脚本 LL 训的、和学习 LL 不默契)的直接手段。

**什么时候选**:HL 和 LL 各自训好后,合体成绩低于各自单独表现时——几乎总是值得跑 RoundA(30-60 分钟)。

**命令**:
```bash
# RoundA: HL 适配学习 LL
python -u habitat-baselines/habitat_baselines/run.py \
  --config-name=social_nav/social_nav_hierarchical_v2_llyield.yaml \
  habitat.dataset.data_path=/habitat-lab/data/<数据集>.json.gz \
  habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.hidden_dim=32 \
  habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.num_rnn_layers=1 \
  habitat_baselines.rl.ppo.hidden_size=32 \
  +habitat_baselines.rl.policy.agent_0.hierarchical_policy.high_level_policy.pretrained_hl_weights=<HL-BC.pth> \
  habitat_baselines.rl.ppo.lr=5e-5 habitat_baselines.total_num_steps=4.0e5 \
  habitat_baselines.checkpoint_folder=checkpoints_joint

# RoundB: 适配 HL 下 LL DAgger 再蒸馏
# 1) 从 RoundA ckpt 抽 HL 权重(多智能体 ckpt 结构 {0:{state_dict},1:{...}}):
#    torch.load(ckpt, weights_only=False)[0]["state_dict"] 里按 "._state_encoder."/"._policy."
#    切 key 存 {"state_encoder":..,"policy":..}(参考 hrl_pipeline/hl_from_msd20.pth 的做法)
# 2) LL_DAGGER=/habitat-lab/hrl_pipeline/ll_dagger2.jsonl + RULE_MODE=learned? 不——用
#    harness 以 teacher HL 或抽出的 HL 权重跑 v2_llyield, LearnedYieldSkill 自动记
#    (obs -> 脚本teacher动作) 到 LL_DAGGER 文件
# 3) cat 旧示范+新DAgger -> LL_DATA=<合并jsonl> python hrl_pipeline/bc_ll_yield.py
```

**停止判据**:最多 2 轮交替;全栈 success 增量 < ~0.06 即停。
**论文口径(诚实)**:交替优化的联合**适配**,不是端到端同时梯度(后者需把 LL 参数纳入优化器并跨 30 步宏动作做信用分配,列 future work)。

---

## 推荐组合顺序(拿到大数据集后)

1. **HL-BC + PPO(脚本LL)** → 得 HL 基线(配方见 memory/以往 session:BC 覆盖全训练集是铁律)
2. **LL 重蒸馏 + DAgger**(若门几何/袋逻辑没变可直接复用现有 `ll_yield_bc.pth`)
3. **路线④ RoundA**(几乎必赚)→ 全栈头条数字
4. **路线① ES** 冲超越 teacher → 若赚,回到 3 再 RoundA 一次
5. **路线③ 地图**、**路线② flat** 作消融行
Go/no-go:每步用同一评测协议(deterministic、evals_per_ep≥5、seed 7、统一 STEP_CAP),新步骤不优于上一步就停在上一步。

---

## 附录 A:LL-PPO 子任务完整版(需要时怎么建,本次未实现)

ES 之外的标准化路线,三步:
1. **yield 子任务奖励 Measure**:注册新 Measure(参考 `SocialNavReward` 的写法),密集项 = 离"人的真实路线(navmesh 最短路)"的 clearance + 不撞 + 位移代价;`habitat.task.reward_measure: <新uuid>` 挂为训练奖励(`habitat/core/environments.py` 每步读取)。
2. **子任务 episode**:从现有集生成"机器人在门前 1-1.5m、人正逼近"的短集(复用 `hrl_pipeline/scene_tools/gen_v3_datasets.py` 的放置原语),人过门且机器人 clear 即成功终止。
3. **flat 训练 + 包装技能**:用路线②的配置骨架(PointNavResNetPolicy gaussian + 标准 PPO/RolloutStorage)在子任务集上训;**不要做权重移植**——新写一个 `PPOYieldSkill`(仿 `LearnedYieldSkill`)直接加载训练出的 RNN actor(SkillPolicy 接口本就传递 rnn_hidden_states)。

## 附录 B:已知问题清单

1. **harness 门几何 bug**:`scripted_hl_eval.py` 只从 `episodes[0]` 加载门线 → 多集数据集上,其它集的"人已过门"释放判据用错门(退化到 recede 计数兜底)。单集/同门数据集不受影响;做多集 teacher 审计前应修(per-episode 重载)。
2. **flat 配置 stop 维**:见路线②"坑"。
3. **地图朝向**:见路线③"首次使用必做"。
4. **GPU 互斥**:ES(多进程 eval)与任何 PPO 训练不要同时跑。
5. **HL 尺寸**:所有加载 32 维 HL-BC/ckpt 的命令必须带 `hidden_dim=32 num_rnn_layers=1 ppo.hidden_size=32`(v2 默认 256,不带会 shape 不匹配)。
6. **多智能体 ckpt 读取**:`torch.load(..., weights_only=False)`,顶层 key 是 `{0:..., 1:..., 'config', 'extra_state'}`。

## 冒烟状态(2026-07-19,全部通过)

| 路线 | 冒烟 | 说明 |
|---|---|---|
| ① ES 驱动器 | ✅ | pop2×1代全环路:扰动→LL_MODEL 注入→评测→选择→落盘 |
| ② flat e2e | ✅ | trainer 完整短训,3 个 checkpoint 落盘 |
| ③ v3map | ✅ | sensor+HL map 分支前向 60 步无异常(修复过一处 maps 导入) |
| ④ RoundA 联合 | ✅ | HL PPO + 冻结学习 LL 完整短训,3 个 checkpoint |
| 回归(v2_llyield) | ✅ | 现有行为不受任何新改动影响 |
