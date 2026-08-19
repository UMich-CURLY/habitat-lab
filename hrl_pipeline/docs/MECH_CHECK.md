# 机制快检判读(2026-08-14,rev3 协议)

三臂:同一 BC 起点(bc_stable_hl,28 集干净演示,acc 0.985),G3/G4 配方(gamma
0.99,critic_lr 3e-2,warmup 60u,entropy 0,1024/1,B2 奖励),quick15 子集,
1.5M 定停取末尾 ckpt,评测 = 完整 train36_v1 evals×3(episode 级成功 = 3/3)。
臂 A(uniform)按用户指示取消;B/C 的 rollout fail 帧占比实测均 0.37
(SPLIT_SCENES=0)。宿主机 8/14 07:36 UTC 死机一次,B 为重训版本。

## 终表

| 策略 | stable-success (28) | stable-failure (4) | retention loss | 新碰撞 | paired Δsteps vs BC (inefficient 13集) |
|---|---|---|---|---|---|
| BC 起点 | 24/28,1 碰 | 0/4 | — | — | — |
| B(仅上采样) | 24/28,0 碰 | **1/4(id20,1099 步,3/3 稳)** | 1(id12) | 0 | **−31** |
| C(上采样+DAPG λ0=0.1) | **26/28,0 碰** | 0/4 | **0** | 0 | −22 |
| teacher(cert3) | 28/28,0 碰 | 0/4 | — | — | 参考系 |

B 的 u86 中段预览:id2 1/3、id20 1/3、id25 1/3——中期曾闪现三个 failure 的
不稳定解,终点只固化了 id20。B/C 都在 parked 场景(id32/34)变慢(仍成功)。

## 三判据结论

1. **Retention**:1.5M 预算 + B2 奖励 + G3 配方下,遗忘远比旧长跑温和——B 无
   DAPG 也只丢 1 集、零新碰撞。DAPG 的保护价值真实但有限(+2 集,还修回 BC
   的 1 个碰撞集)。
2. **Efficiency**:两臂都未整体超 teacher(+24/+25);难集(inefficient)上
   B −31 / C −22 优于 BC。效率主瓶颈与几何诊断一致:extra_backoff(~130 步)+
   extra_wait(~90 步)是 teacher 演示焊进去的结构性浪费,现行 reward 对它
   没有梯度。
3. **Capability(核心)**:**只有无 DAPG 的 B 长出新能力**(id20 死锁→穿越)。
   C 的 probe(cos 中位 −0.68,89% 负)是真实压制而非防噪。碰撞型失败
   (2/3/25)两臂皆未固化,但 B 中期闪现过 1/3——PPO 摸得到解,固化缺
   价值坡度。

## 决策分支(对应 rev3 预设)

命中"B 解出 failure 而 C 被压制"分支 → **下一轮去掉(或大幅弱化)DAPG,
走 reward shaping**:三层新奖励(终端排序 + success-conditioned 效率 +
走廊势函数/及时释放,阈值来自 YIELD_GEOM_REPORT.txt:stop-backoff =
plan_clear≥1.0,safe-go = mpd_go≥0.75),对照 = 本轮 B。若新奖励下
retention 恶化,再回补轻 DAPG(λ0≤0.03、u60 前归零)。

## 泛化读数(2026-08-14 补)

- **PPO-unseen(train36 内、quick15 外的 17 个 stable-success)**:B succ 15/17 与 BC
  持平、0 碰(BC 有 1 碰,被修掉)、唯一丢失 id12 在 unseen 侧(漂移非干扰);
  paired Δsteps(B−BC)=+22 —— **效率提升不迁移,安全性迁移**。C unseen 16/17。
- **场景级零泄漏 holdout 2 集**(106879080/107734479,train36 与 BC 演示均未见):
  ep0 teacher 3/3@840,BC 3/3@979,B 3/3@958;ep1 teacher 3/3@930,BC 与 B 完全
  相同 2/3+1 碰@614。→ 零泄漏迁移存在但不完美;PPO 对未见场景不增不减,
  学生的泛化来自 BC/观测结构。n=2 仅作轶事,正式轮需配大 held-out 集
  (doorcand_v3 未用门 + 点选流水线)。

## 资产索引

stats_mech{BC,B,C}.json、stats_mechB_mid.json、_by_group_mech.txt、
tb_mech{B,C}(C 含 dapg_grad_ratio/cos 全程)、rollout_counts.json(实测采样混合)、
trace_{teacher,BC,C,g3}.jsonl + YIELD_GEOM_REPORT.txt + yield_geom_episodes.csv、
TEACHER_CERT3.csv(三分类)、MECH_CHECK.md(本文)。
