# P007 最终策略报告

## 方法与提交内容

本次提交是确定性的纯规则策略。每个机器人只接收协议规定的单机器人局部观测，在回合内独立保存可见目标的位置历史，估算目标运动并追踪短暂离开视野的目标。没有可追踪目标时，机器人前往按编号分散的固定航点；近距离可见队友产生避碰斥力。

当本机同时看见多个目标和队友时，策略结合公开动力学、本机状态、可见队友状态及目标历史，估计剩余回合内的覆盖价值，并枚举小规模局部分配。这个估计只决定本机追踪哪个可见目标；真实覆盖与碰撞始终由官方环境计算。策略不读取训练全局状态、隐藏目标位置或其他机器人的私有观测，也不加载训练模型。

正式入口为 `entry.py:build_policy`，由 `candidate_policy.py` 实现局部覆盖价值分配。唯一登记产物是 `artifacts/rule-params.json` 中的固定追踪参数；`submission.yaml` 登记其 SHA-256 和大小。推理只使用官方环境已提供的 NumPy，无额外推理依赖。纯规则策略没有训练步数和训练模型。

## 关键对照

使用此前稳定规则提交 `acd0d13d869fe931b1ddb7d90ad1b313f8042c77` 作为对照。两种策略在相同公开任务配置及相同场景种子上分别闭环运行。三批种子互不重叠，四个用例每批各 1,000 个回合，共 12,000 个成对场景；按公开评分规则对四个用例等权折算。第一批在候选选择后进行，后两批在策略冻结后复核。无训练预算，成对评测实际运行 24,000 个策略回合。

| 批次 | 四个用例的种子起点 | 对照策略折算分 | 提交策略折算分 | 增益 95% 区间 | 碰撞 agent-step（对照/提交） |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 3210001 / 3220001 / 3230001 / 3240001 | 166.80 | 168.05 | [0.91, 1.61] | 118 / 148 |
| 2 | 3310001 / 3320001 / 3330001 / 3340001 | 166.69 | 167.78 | [0.74, 1.46] | 110 / 142 |
| 3 | 3510001 / 3520001 / 3530001 / 3540001 | 166.56 | 168.09 | [1.11, 2.00] | 128 / 134 |
| **合并** | **12,000 个场景** | **166.69** | **167.98** | **[1.07, 1.52]** | **356 / 424** |

三批合并后，各用例的本地逐回合 `J` 均值如下。每行有 3,000 个成对场景。

| 用例 | 对照策略 | 提交策略 | 对照/提交覆盖目标步 | 对照/提交碰撞 agent-step |
| --- | ---: | ---: | ---: | ---: |
| basic-0 | 0.16758 | 0.16865 | 15103 / 15199 | 106 / 102 |
| basic-1 | 0.16624 | 0.16758 | 14984 / 15106 | 112 / 118 |
| coop-0 | 0.16755 | 0.16873 | 15091 / 15201 | 58 / 76 |
| coop-1 | 0.16538 | 0.16694 | 14900 / 15050 | 80 / 128 |

提交策略合计多覆盖 478 个目标步；逐回合比较为 225 胜、32 负、11,743 平。零覆盖回合仅从 3,174 降到 3,173，说明发现目标仍是主要限制。额外碰撞发生在 25 个回合，其中 9 个没有覆盖收益。退化最重的种子 `3240250` 和 `3510515` 各少 3 个覆盖目标步，单回合 `J` 下降 0.10。增益是均值结论，不保证每回合都有改善。

## 官方脚本本地预览

在 `enter`（Python 3.12）环境，对最终提交树运行 `scripts/check_submission.py`：0 拒绝项、0 扫描命中，`working_tree_dirty=false`。再从干净目录单独运行 `scripts/evaluate_one.py` 固定公开 8 回合：8/8 `ok`、`errors=[]`、`performance_score=225.00`、零碰撞，`working_tree_dirty=false`。对照策略在同一固定公开套件也为 225.00。上述公开成绩的 `provenance=local_preview`；最终以组织方核验为准。

## 复现方法

以下命令从仓库根目录运行。每次的输出路径在运行前应不存在；评测脚本会创建 `participant/P007/dev/`，该目录是本地临时结果，不属于提交文件。

```text
python participant/P007/paired_eval.py --per-case 1000 --starts 3210001 3220001 3230001 3240001 --output participant/P007/dev/paired-batch-1.json
python participant/P007/paired_eval.py --per-case 1000 --starts 3310001 3320001 3330001 3340001 --output participant/P007/dev/paired-batch-2.json
python participant/P007/paired_eval.py --per-case 1000 --starts 3510001 3520001 3530001 3540001 --output participant/P007/dev/paired-batch-3.json
python participant/P007/analyze_paired_eval.py
python scripts/check_submission.py --submission participant/P007 --output participant/P007/dev/final-check
python scripts/evaluate_one.py --submission participant/P007 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output participant/P007/dev/final-public8
```

`rule_final_eval.py` 在相同种子上分别运行对照与提交策略，按官方逐步覆盖和碰撞定义计算 `J`；`paired_eval.py` 对四案等权折算；`analyze_paired_eval.py` 汇总逐组得失和分层 bootstrap 区间。预检和公开预览须分别在干净目录运行，才能得到 `working_tree_dirty=false` 的记录。先前实验记录可从已有提交历史查看；12,000 回合的原始逐回合 JSON 未纳入最终目录，可按以上命令重建。当前目录仅保存最终方案及复现所需文件。
