# P007 最终策略报告

## 提交方法

当前入口是确定性的纯规则策略。`entry.py` 创建 `CoverageAssignmentPolicy`，参数清单只包含 `artifacts/rule-params.json`，推理不加载训练模型。每个机器人只使用协议提供的自身状态、可见目标和可见队友的局部观测；实例之间不共享隐藏状态。

策略保留短期目标轨迹估计、固定搜索航点、可见目标追踪和队友避碰。机器人同时看到多个目标和队友时，根据公开动力学估算每个局部机器人与可见目标在剩余步数内的覆盖价值，枚举小规模局部分配，再选择自身目标。局部估算只用于决策，实际成绩由评测环境计算。

## 验证结果

H016 稳定提交 `acd0d13d869fe931b1ddb7d90ad1b313f8042c77` 是同种子对照。四个公开任务用例分别使用三批相互独立的新种子，每批每案 1,000 回合，共 12,000 个成对回合。按官方权重折算，H016 为 **166.69**，当前策略为 **167.98**；分层 bootstrap 的增益 95% 区间为 **[1.07, 1.52]**。当前策略多覆盖 478 个目标步，碰撞 agent-step 从 356 增至 424；225 胜、32 负、11,743 平。零覆盖回合从 3,174 减至 3,173，目标发现仍是主要限制。具体种子、逐组结果、失败回合与复现命令见 `H031_FINAL_EVIDENCE.md`。

使用 `enter` 环境运行官方提交预检，拒绝项为 0。官方脚本固定公开 8 回合的本地预览全部 `ok`，`errors=[]`，`performance_score=225.00`，零碰撞；同套件 H016 也是 225.00。该结果的 `provenance=local_preview`，最终成绩以组织方核验为准。

## 文件与复现

正式推理入口及参数为 `entry.py`、`candidate_policy.py`、`artifacts/rule-params.json`、`submission.yaml` 和 `requirements-infer.lock`。`rule_final_eval.py`、`h031_full_eval.py`、`h035_analyze.py` 用于复核同种子对照；`H031_FINAL_EVIDENCE.md` 记录完整命令。`LOG.md` 和 `experiments.csv` 保存实验过程，旧实验脚本已从当前提交目录移除，其历史仍可通过 Git 查看。

从仓库根目录执行官方入口检查：

```text
python scripts/check_submission.py --submission participant/P007 --output participant/P007/dev/final-check
python scripts/evaluate_one.py --submission participant/P007 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output participant/P007/dev/final-public8
```

每次运行前，输出目录应不存在或为空。以上命令使用官方代码与公开配置，不修改官方文件。
