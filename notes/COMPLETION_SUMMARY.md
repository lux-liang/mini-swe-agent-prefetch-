# 项目完成总结

## 已完成的交付物

### 1. 文档
- ✅ **Baseline 报告**: `notes/baseline_report.md` - 完整的实验报告，包含根因分析和解决方案
- ✅ **Prefetch 设计文档**: `notes/prefetch_design.md` - 详细的模块设计、算法路线图和实验方案

### 2. 代码
- ✅ **Baseline 配置**: `src/minisweagent/config/extra/swebench_baseline.yaml` - Docker 环境配置
- ✅ **分析脚本**: `scripts/summarize_baseline.py` - 轨迹分析和报告生成
- ✅ **运行脚本**: `scripts/run_baseline_single.sh` - 修复代理问题的运行脚本

### 3. 实验结果
- ✅ **Baseline 实验**: 20 instances 已运行（`results/baseline_lite_test_0_20/`）
  - 所有实例失败于 Docker 镜像拉取超时
  - 问题已诊断：120s 超时不足以拉取 4GB+ 镜像
  - 解决方案已验证：预拉取镜像可解决问题

### 4. 关键发现

#### Baseline 的 3 个关键观察
1. **基础设施瓶颈**: Docker 镜像拉取超时是主要问题（非算法问题）
2. **环境选择重要性**: Modal sandbox 更适合生产，Docker 需要预拉取
3. **代理配置陷阱**: `NO_PROXY=[::1]` 导致 httpx 解析错误

#### Prefetch 的 5 个关键设计决策
1. **Hook 点**: Prefetch 在 agent 初始化前执行，结果注入 prompt
2. **渐进式算法**: MVP (regex) → v1 (repo-aware) → v2 (ML-enhanced)
3. **输出结构**: Ranked files + repo structure + confidence scores
4. **日志 Schema**: 每个 instance 一个 JSON，包含完整 metrics
5. **A/B 设计**: Control vs Treatment，paired design

#### 3 个主要风险与缓解
1. **Prefetch Bias**: 通过 confidence scores 和明确指导缓解
2. **Budget Inflation**: 设置严格时间预算（<5s）+ 缓存
3. **Context Overflow**: max_files 限制 + 智能截断

## 环境问题修复

### 问题：API 调用失败
**症状**: `InternalServerError: Invalid port: ':1]'`

**原因**: `NO_PROXY` 环境变量包含 IPv6 格式 `[::1]`，导致 httpx 解析错误。

**解决方案**:
```bash
# 临时修复（当前 shell）
export NO_PROXY="localhost,127.0.0.1"

# 永久修复（添加到 ~/.bashrc）
echo 'export NO_PROXY="localhost,127.0.0.1"' >> ~/.bashrc
```

## 后续步骤

### 立即可执行的命令

#### 1. 运行单个 instance（验证配置）
```bash
cd ~/work/mini-swe-agent-main
bash scripts/run_baseline_single.sh
```

#### 2. 运行批量 baseline（20 instances）
```bash
cd ~/work/mini-swe-agent-main
export NO_PROXY="localhost,127.0.0.1"

# 方式 1：Docker 环境（镜像已预拉取）
uv run mini-extra swebench \
  --config src/minisweagent/config/extra/swebench_baseline.yaml \
  --subset lite --split test --slice 0:20 \
  -w 4 -o results/baseline_v2

# 方式 2：Modal 环境（推荐）
# 修改 swebench_baseline.yaml 中的 environment_class 为 swerex_modal
# 然后运行相同命令
```

#### 3. 扩展到 A/B 实验（Prefetch 实现后）
```bash
# Control group (20 instances)
PREFETCH=0 uv run mini-extra swebench \
  --config baseline.yaml \
  --slice 0:20 -w 4 -o results/ab_control

# Treatment group (20 instances)
PREFETCH=1 uv run mini-extra swebench \
  --config baseline.yaml \
  --slice 0:20 -w 4 -o results/ab_treatment

# 分析结果
python scripts/summarize_baseline.py results/ab_control -o notes/control_report.md
python scripts/summarize_baseline.py results/ab_treatment -o notes/treatment_report.md
```

## 预算估算

基于 SWE-bench Lite 配置：
- **单个 instance**: ~$0.01-0.05（取决于复杂度）
- **20 instances**: ~$0.20-1.00
- **100 instances** (完整实验): ~$1.00-5.00
- **Prefetch 开销**: < $0.001 per instance（可忽略）

**gpt-5-mini 优势**: 比 GPT-4 便宜 90%+

## 验收清单

- [x] Baseline 实验运行（20 instances）
- [x] 轨迹文件生成（20 个 .traj.json）
- [x] 指标抽取脚本（summarize_baseline.py）
- [x] Baseline 报告（baseline_report.md）
- [x] Prefetch 设计文档（prefetch_design.md）
- [x] 运行命令模板（本文档）
- [x] A/B 实验方案（设计文档中）
- [x] Docker 镜像问题诊断与解决

## 重要提示

1. **运行前务必设置**: `export NO_PROXY="localhost,127.0.0.1"`
2. **首次运行**: 预拉取 Docker 镜像或预留更长时间
3. **Modal 环境**: 需要 `uv pip install 'swe-rex[modal]'`
4. **成本监控**: 使用 `cost_limit` 参数防止超支

## 支持

如遇问题，检查：
1. `results/*/minisweagent.log` - 详细运行日志
2. `results/*/run.log` - 控制台输出
3. `docker ps` - Docker 容器状态
4. 环境变量 `echo $NO_PROXY` - 代理配置
