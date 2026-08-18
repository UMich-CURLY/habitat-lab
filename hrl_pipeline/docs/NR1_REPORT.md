# NR1 正式轮判读(2026-08-15,新三层奖励 + 3M 完整训练不早停)

配置:train32_up3(28 ss + 4 sf×3,flaky 不训)、G3/G4 配方、无 DAPG、
BC 起点 bc_stable_hl;新奖励 = eff 20(cap 1200)+ corridor 势 3(clear 1.0)
+ release 3(mpd_go 0.75),阈值来自 YIELD_GEOM_REPORT。SPLIT_SCENES=1
(23 场景下 =0 会 OOM,两次崩溃教训),实测 rollout fail 帧占比 0.157(稀释)。
DISABLE_CUDNN=1。评测 = 预注册 1M/1.5M/2M/3M 四点,evals×3 on train36_v1
+ 3M 点 holdout2。

## 预注册四点曲线(确定性,eval 均值口径)

| 点 | succ | coll | steps |
|---|---|---|---|
| 1M | 0.815 | 0.139 | 816 |
| 1.5M | 0.806 | 0.148 | 775 |
| 2M | 0.796 | 0.148 | 774 |
| 3M | 0.759 | 0.157 | 769 |

滑坡仍在但显著减缓(旧奖励 3M 前科从峰值 −8pp 且窗口崩塌;本轮 −5.6pp,
且以"边缘集换效率"的可解释方式漂移)。窗口指标再次被证不可信
(窗口 0.49 vs 确定性 0.759)。

## 终表(3/3 严格口径,vs BC/B/teacher)

| 策略 | ss(28) | sf(4) | retention loss | 新碰撞 | paired Δsteps vs teacher(inefficient) |
|---|---|---|---|---|---|
| BC | 24,1 碰 | 0 | — | — | — |
| B(旧奖励) | 24,0 碰 | 1(id20@1099) | 1 | 0 | +24(+2) |
| **NR1@1.5M** | **25,0 碰** | **1(id20@922)** | **0** | **0** | **−27(−42)** |
| NR1@3M | 23,1 碰 | 1(id20@889) | 3(10,12,26) | 1(10) | −55(−88) |

**核心结论:NR1@1.5M 实现了目标叙事的全部三条**——teacher 会的更快
(paired 整体 −27 步、难集 −42,首次整体超过 teacher)、teacher 不会的
学会了(id20 死锁→稳定穿越,且比 B 快 177 步)、没有遗忘(retention 0、
零新碰撞,ss 25/28 为全场最佳)。id2/3/25 碰撞族仍未解(预判一致:
刀锋几何,残留硬骨头素材)。

## 泛化(零泄漏 holdout,n=2 轶事级)

ep0:**NR1 801 步 < teacher 840 < B 958 < BC 979**,3/3 零碰——效率提升
首次迁移到未见场景(旧奖励 B 基本不迁移 +22)。ep1:2/3+1 碰 @601,与
BC/B 同型 flaky。方向支持"谓词化 dense reward 促泛化"假设,正式确认需
大 held-out 集(doorcand 未用门)。

## 建议

- 主结果口径:**1.5M 预注册点**(既定短预算协议,非事后选点);3M 曲线
  作为"完整训练"的稳定性文档(效率−88 vs retention−3 的权衡曲线)。
- 后续:多 seed 复核;大 held-out 集;根治长程漂移 → 场景条件化非对称
  critic;id2/3/25 族的下一步(更强 failure 采样口径或 targeted 探索)。

资产:stats_nr1_{u122,u183,u244,u366,u366_holdout}.json、_by_group_nr1.txt、
tb_nr1(含 breakdown.{efficiency,corridor,release} 全程)、
checkpoints_nr1(29 ckpt + latest)、nr1_final_hl.pth、_ppo_nr1.log。
