# hrl_pipeline 清理方案(待确认,尚未删除任何东西)

现状:**537 项 / 5.9 GB**。其中 `.log` 占 5.3 GB(90%),而这些日志 97% 是
RVO/传感器刷屏——528 MB 的 `_ppo_prod1.log` 里只有 0.2 MB 是指标行。

分四档:**A 保留** / **B 瘦身(不删内容)** / **C 建议删除** / **D 需要你拍板**。

---

## A. 保留 —— 最终管线

### A1. HL(高层)核心脚本
| 文件 | 作用 |
|---|---|
| `scripted_hl_eval.py` | **规则 expert 本体** + 演示采集。`RULE_MODE=rule_markov` 是最终 teacher(含 wait) |
| `bc_pretrain_hl.py` | BC 预训练(feat7 + prev-action one-hot,宽度 10) |
| `_extract_hl_map.py` | 从多智能体 ckpt 抽 HL 权重(直接载 ckpt 评测是静默 no-op,必须走这步) |
| `analyze_hl_decisions.py` | 演示质量:标签冲突率 / 重放一致性 / 分支占比 |
| `probe_wait_prob.py` | 检查策略给各动作的概率(诊断 wait 是否真被学到) |
| `check_train_health.py` | PPO 训练健康一键检查 |

### A2. LL(底层)核心脚本
| 文件 | 作用 |
|---|---|
| `bc_ll_yield.py` | 让行技能蒸馏(脚本 → 神经网络) |
| `es_ll_finetune.py` | LL 的进化搜索微调 |
| `verify_ll_demos.py` | 过滤 LL 演示,只保留干净成功段 |
| `ll_dataset_stats.py` / `ll_mse_units.py` | LL 数据统计 / MSE 的物理量纲换算 |
| `reconstruct_es_center.py` | 重建 ES 的 center 权重 |

### A3. 数据集构建与认证
| 文件 | 作用 |
|---|---|
| `build_final_datasets.py` | **生成 TRAIN / EVAL** 的脚本 |
| `dataset_provenance.py` | **泄漏检查**(按 episode 内容溯源 + 场景重叠) |
| `certify_solvability.sh` + `certify_merge.py` | 可解性认证(每集哪些模式能过) |
| `build_pool_v1.py` | 76 集总池(带标签与溯源) |
| `make_subset.py` | 按索引切子集 |
| `classify_teacher.py` | 按 teacher 结局给集分类 |

### A4. 诊断 / 可视化
| 文件 | 作用 |
|---|---|
| `eval_to_tb.py` | 111 个评测结果 → TensorBoard(已修 evals_per_ep 去重 bug) |
| `logs_to_tb.py` | BC/ES/DAgger 的 stdout 日志 → TensorBoard |
| `dump_losses.py` / `dump_reward.py` | 训练曲线速览(终端表格) |
| `reward_table.py` | 六分量奖励分解(按 success/collision/timeout 分组) |
| `by_group.py` | 按标签分组的评测细分 |

### A5. Checkpoint(保留 11 个,共约 0.4 MB)
| 文件 | 是什么 | 成绩 |
|---|---|---|
| `nr1_ck14_hl.pth` ⭐ | HL 主结果(3M 跑的第 15 个存档 ≈1.5M) | 见 NR1_REPORT.md |
| `nr1_s200_hl.pth` / `nr1_s300_hl.pth` | 同配方换种子(复现证据) | 83% / 81% |
| `nr1_ck9/ck19/final_hl.pth` | 1M/2M/3M 曲线点(支撑"为什么停在 1.5M") | 81% / 80% / 76% |
| `bc_stable_hl.pth` | HL 的 BC 起点(**训练日志已丢失**) | 一切"进步了多少"的基准 |
| `ll_yield_es_v3.pth` ⭐ | LL 主结果(ES 微调) | EVAL 79.8% |
| `ll_yield_dagger1.pth` | LL 的 DAgger 热启动 | — |
| `mechB_hl.pth` / `mechC_hl.pth` | 三臂机制实验的对照臂 | 80% / 79% |

### A6. 文档
`PIPELINE_REFERENCE.md`(1199 行,总参考)、`DATASETS_FINAL.md`、`NR1_REPORT.md`、
`LL_TRAINING_CURVES.md`、`LL_PLAN.md`、`MECH_CHECK.md`、`PARAM_AUDIT.md`
(全栈阈值参数表)、`FRAMEWORKS_HOWTO.md`(踩坑手册)

### A7. 数据
- 评测 stats:`stats_nr1_*.json`、`stats_var_*.json`、`stats_teacher_cert3.json`、
  `bd_*.json`(带奖励分解)、`cert_*.csv/json`
- 演示语料:`ll_demos_v2_clean.jsonl`、`ll_dagger_r1.jsonl`、
  `hl_decisions_rule_markov_all.jsonl`
- `TEACHER_CERT3.csv`、`POOL_v1.csv`、`TRAIN36.csv`、`cert_*.csv`

---

## B. 瘦身(内容不丢,只砍刷屏) —— **省约 5.3 GB**

168 个 `.log`。做法:每个日志只保留 `update:` / `Average window` /
`Average episode` / `Traceback` / checkpoint 行,其余(RVO 初始化、
"missing required sensors"、tqdm 进度条)全部丢弃。

| 日志 | 现在 | 瘦身后 |
|---|---|---|
| `_ppo_prod1.log` | 528 MB | ~0.2 MB |
| `_ppo_g4.log` / `_ppo_g5.log` / `_ppo_nr1.log` | 各 ~525 MB | 各 ~0.2 MB |
| `_ppo_g1/g2/g3.log`、`_ppo_nr1_s200/s300.log`、`_ppo_mechC.log` | 各 ~260 MB | 各 ~0.2 MB |
| 其余 ~155 个 | 合计 ~2 GB | ~2 MB |

**注意**:这些日志里的指标行**已经在 `tb_*/` 里有了**,瘦身后主要保留异常/回溯,
用于事后排查。若你觉得没必要,可直接整包删除。

---

## C. 建议删除

### C1. 一次性分析脚本(21 个,共约 90 KB)
`an_es_v3_rep.py` `an_es_v3_noise.py` `an_es_v3_repro.py` `an_es_v3_dump.py`
`an_es_v3_mine.py` `an_es_v3_mine2~5.py` `an_es_v3_map.py` `an_es_v3_find_ds.py`
`_mine_es_v2.py` `_mine_es_v2b~e.py`
→ ES 调查期间的临时脚本,无 docstring、无复用价值,结论已进 `LL_TRAINING_CURVES.md`。

`summarize_cd13.py` `dump_tb.py` `dump_tb2.py` `tb_peaks.py` `traj.py`
`agg_eval.py` `_agg.py` `extract_frames.py` `dump_map.py`
→ 早期一次性脚本,功能已被 `eval_to_tb.py` / `dump_losses.py` / `dump_reward.py` 取代。

### C2. 废弃 checkpoint(21 个)
| 文件 | 是什么 | 为什么可删 |
|---|---|---|
| `bc_ep70_hl.pth` `hl_from_msd20.pth` `flat_bc.pth` | 7 月的单集/扁平基线 | 早已被 train36 系列取代 |
| `bc_markov_hl.pth` `bc_train36_hl.pth` `bc_train36_wait_hl.pth` | 8/4–8/13 的 BC 中间版 | 被 `bc_stable_hl.pth` 取代 |
| `b2_final_hl.pth` `g3_final_hl.pth` | 我这条诊断线的最佳模型(32/36) | **在 train36_v1 上训的,该集与 EVAL 95% 重合,结论不可用** |
| `ppo_ckpt1_hl.pth` `ppoD_ckpt7/8_hl.pth` `ppoE_ckpt7_hl.pth` | 崩塌轨迹里的抢救点 | 已证实不可复现(同位置两次 32/36 vs 28/36) |
| `prod1_ckpt6/final/ckpt29/u392_hl.pth` | 3M 退化曲线的采样点 | 曲线结论已记录,权重无用 |
| `mechB_mid_hl.pth` | 机制实验中间点 | 保留 B/C 终点即可 |
| `_tmp_bc_v2.pth` `_tmp_bc_v2_rep.pth` `_tmp_bc_dagger1.pth` | 临时文件 | 名字即说明 |
| `ll_yield_bc.pth` `ll_yield_bc_v2.pth` `ll_yield_es_ms4.pth` `ll_yield_es_v2.pth` | LL 早期版本 | v2 已证实过拟合(25/28→18/28),ms4 是 7 月旧版 |
| `ll_yield_es_seed1/seed2.pth` `ll_yield_center_r2.pth` | LL 方差实验的臂 | 结论已进 `LL_TRAINING_CURVES.md` |

### C3. 中间数据(约 120 MB)
- `hl_decisions_learned.jsonl`(34 MB)——learned 模式的原始决策流,分析已完成
- `trace_C/teacher/BC/g3.jsonl`(合计 37 MB)——四策略轨迹对比,结论已出
- `ll_train_r1.jsonl`(15 MB)、`ll_demos.jsonl`、`ll_demos_v2.jsonl`(未过滤版,
  已有 `_clean` 版)
- `hl_decisions_*.pre_markov.jsonl`(5.3 MB)——旧 teacher 的基线日志,
  7.5% 冲突率结论已记录
- `_bcll_ms4.json` `_cd17_*.json` `_bl_*.json` `_eval_*.json` `_flat*.json`
  `fa_*.json` `fix*.json` `audit_parked.json` 等约 40 个早期评测

### C4. 目录(约 5.4 GB 里的大头,不含日志)
| 目录 | 大小 | 内容 | 建议 |
|---|---|---|---|
| `expert_videos_train36/` | 90 MB | 36 集 expert 视频 | 删(可随时重渲) |
| `scene_rgb/` `scene_menu/` `scene_tiled/` | 70 MB | 点场景用的场景缩略图 | 删(点选已完成) |
| `video_diag/` `video_rule_markov/` `video_rule_yield/` `video_wait579/` `video_waitB579/` `video_manual_uncert/` | 60 MB | 各阶段诊断视频 | 删,**只留 `video_waitC579/`**(最终 teacher 的三集) |
| `_repro_v3/` | 27 MB | ES 复现实验的原始 txt | 删(结论已进文档) |
| `manual/` | 121 MB | 点选流程的 review PNG | 删(数据集已建好) |
| `archive/` | 106 MB | 更早的 frames/logs/stats/weights | 删 |
| `map_check/` `figures/` `tb/` `datasets/` `__pycache__/` `es_ll_*/` | ~9 MB | 杂项 | 删(`figures/` 若要出图可留) |

---

## D. 需要你拍板

1. **`b2_final_hl.pth` / `g3_final_hl.pth`** —— 我这条线的"最佳模型"(32/36、超过
   teacher)。但它们训练用的 `train36_v1` 与 EVAL 有 95% 重合,**结论不能用于论文**。
   删,还是留作过程记录?
2. **日志**:按 B 瘦身(留异常回溯),还是整包删掉?
3. **`figures/`**:是否已有出图需求?
4. **`manual/` 的 review PNG**:以后还要再点场景的话,这套流程要不要留?
5. **7 月的旧文档**(`HRL_DIAGNOSIS.md` `DIAGNOSIS_PROGRESS.md` `IMPLEMENTATION_PLAN.md`
   `METHODOLOGY.md` `STAGE_LOG.md` `SESSION_REVIEW.md` `INDEX.md` `dataset_review.md`)
   —— 历史记录,内容已被 `PIPELINE_REFERENCE.md` 覆盖。删还是归档?

---

## 预计效果

| | 现在 | 清理后 |
|---|---|---|
| 文件数 | 537 | ~90 |
| 体积 | 5.9 GB | **~30 MB**(日志瘦身)或 **~25 MB**(日志全删) |

清理后的 `hrl_pipeline/` 结构:
```
hrl_pipeline/
  README.md              ← 新写:这个目录是什么、怎么跑完整管线
  PIPELINE_REFERENCE.md  ← 总参考
  DATASETS_FINAL.md      PARAM_AUDIT.md      NR1_REPORT.md
  LL_TRAINING_CURVES.md  LL_PLAN.md          MECH_CHECK.md
  FRAMEWORKS_HOWTO.md
  hl/       scripted_hl_eval.py  bc_pretrain_hl.py  _extract_hl_map.py ...
  ll/       bc_ll_yield.py  es_ll_finetune.py  verify_ll_demos.py ...
  data/     build_final_datasets.py  dataset_provenance.py  certify_*.{sh,py} ...
  diag/     eval_to_tb.py  logs_to_tb.py  dump_*.py  reward_table.py ...
  weights/  11 个 checkpoint
  results/  评测 stats + csv
  demos/    3 个演示语料
  video_waitC579/
```
(是否要真的建子目录、还是保持平铺但删干净,也请你定——建子目录会改脚本里的相对路径,
我会一并改掉。)
