# P007 个人开发测试集

本目录用于在公开测试之外检查策略泛化，不属于官方成绩。

- `dev-suite-v1.yaml`：40 个固定场景，其中 20 个 `uniform`、20 个 `crossing`。
- `dev-seeds-v1.json`：与套件匹配的固定策略种子安排。
- 场景种子分别为 31001–31020 与 32001–32020，不使用公开套件种子。
- 物理、观测、奖励、3v3 规模和 10 步回合与公开套件保持一致。
- 两组各占开发分数的 50%，每个场景运行一次。

在仓库根目录运行：

```powershell
python scripts/evaluate_one.py `
  --submission participant/P007 `
  --suite participant/P007/dev/dev-suite-v1.yaml `
  --seeds participant/P007/dev/dev-seeds-v1.json `
  --output outputs/P007/dev-v1
```

每次评测应使用不存在或为空的新输出目录。比较方案时至少检查总分、两个组的
`mean_j`、碰撞率，以及 `episodes.csv` 中最差场景和零覆盖场景数量。

## 首次基准

策略提交：`e157d3b`（按轴增强目标追踪驱动）。

```text
status = ok
provenance = local_preview
performance_score = 161.6667

basic:
  mean_j = 0.1800
  mean_collision_rate = 0
  zero_coverage_cases = 1 / 20

cooperation:
  mean_j = 0.1433333
  mean_collision_rate = 0
  zero_coverage_cases = 7 / 20
```

本次本地输出位于 `outputs/P007/dev-v1-e157d3b/`，不加入 Git。开发集成绩只用于
方案比较，不能描述为官方公开测试成绩或组织方核验成绩。

## 主动搜索实验结论

E008–E011 依次测试了固定顺序网格搜索、新颖度—距离效用、可见队友重叠惩罚和
五步新颖度恢复。四个版本的开发集总分均为 161.6667，crossing 零覆盖场景均为
7/20，没有超过首次基准。因此当前策略未保留 5×5 网格搜索实现。
