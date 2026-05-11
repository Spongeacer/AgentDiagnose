# AgentDiagnose

一个用于评估和诊断 LLM 智能体轨迹的开源工具包。支持编码智能体（Claude Code / Ducc 格式）和网页浏览智能体，结合基于规则的行为分析与 LLM 评判，实现可落地、可复现的评估。

## 架构

`reasoning_quality` 评分器采用三层流水线：

```
Trajectory.actions（原始工具调用序列）
         │
         ▼
[第一层] behavioral_metrics.py
          • 基于 SHA256 哈希的回溯检测
          • exploration_ratio（唯一状态数 / 总步数）
          • verification_rate（Edit → Bash/读取同一文件）
          • 循环检测（重复动作序列模式）
          → TrajectoryMetrics（health_tag: healthy / blind_edit / dead_loop）
         │
         ▼
[第二层] LLM（通过 LiteLLM 调用 Gemini API）
          • 将行为证据作为 ground truth 注入提示词
          • LLM 对 4 个子维度按 1–4 分评分
         │
         ▼
[第三层] 后处理规则
          • 基于行为事实对 LLM 分数进行封顶/覆盖
          • 将 verification_rate 融合进 self_verification 分数
          → ScorerResult（0–1 分数 + 完整详情）
```

该设计可防止 LLM 被“听起来合理但实际行为不符”的推理文本欺骗（例如声称已验证、却从未在编辑后读取文件）。

---

## 快速开始

### 评估轨迹

```bash
# 编码智能体轨迹（Ducc/Claude Code JSONL 格式）
python evaluate_trajectories.py \
  --input path/to/data.jsonl \
  --input-format ducc_jsonl \
  --scorers reasoning_quality objective_quality navigation_path_code_type \
  --output-json results.json

# 网页浏览智能体轨迹（JSON 文件目录）
python evaluate_trajectories.py \
  --input examples/sample_trajectories \
  --scorers reasoning_quality objective_quality navigation_path \
  --output-json results.json

# 试运行 —— 仅统计 token 用量，不实际调用模型
python evaluate_trajectories.py \
  --input path/to/data.jsonl \
  --input-format ducc_jsonl \
  --scorers reasoning_quality \
  --dry-run
```

### 一站式仪表盘

```bash
./launch_dashboard.sh
```

自动运行完整流水线（动词-名词提取 → 标签云 → 嵌入 → 评估），并启动交互式 Web 仪表盘，默认地址为 `http://localhost:8080`。

---

## 评分器

| 评分器 | 类型 | 适用对象 | 说明 |
|--------|------|--------|-------------|
| `reasoning_quality` | LLM + 规则 | 编码 & 网页智能体 | 基于行为证据的 4 子维度质量评分 |
| `objective_quality` | LLM | 通用 | 任务描述质量（具体性 / 可执行性） |
| `navigation_path` | 基于规则 | 网页浏览智能体 | 导航路径诊断指标 |
| `navigation_path_code_type` | 基于规则 | 编码智能体 | 工具调用模式分析 |

---

### reasoning_quality

结合 LLM 评判与基于实际工具调用提取的规则约束。

#### 子维度

| 维度 | 评分方 | 说明 |
|-----------|-----------|-------------|
| `task_decomposition` | 仅 LLM | 智能体将复杂任务拆分为可管理步骤的能力 |
| `observation_reading` | 仅 LLM | 智能体对工具输出理解的准确程度 |
| `self_verification` | 规则 + LLM 融合 | 智能体是否通过读取/测试验证编辑结果 |
| `strategic_backtracking` | 规则 + LLM | 智能体是否识别错误路径并纠正。不适用时为 `null` |

#### 后处理规则

**strategic_backtracking（策略性回溯）**

| 条件 | 操作 |
|-----------|--------|
| `health_tag == "死循环"` | 封顶 0.25（行为事实覆盖 LLM） |
| `health_tag == "盲打"` 且 `backtrack_count == 0` | 规则分数：`min(1 − continue_edit_rate, 0.40)` |
| `health_tag == "盲打"` 且 `backtrack_count ≥ 2` | 封顶 0.50 |
| `health_tag == "健康"` 且 `backtrack_count == 0` 且 `loop_detected == 0` 且 `exploration_ratio ≥ 0.8` | `null`（N/A — 干净线性路径，维度不适用） |
| 其他情况 | 直接使用 LLM 分数 |

**self_verification（自我验证）**

```
rule_score = min(1.0, verification_rate × 1.25)   当 edit_total ≥ 3
           = 0.5（中性）                            当 edit_total < 3
final      = 0.7 × rule_score + 0.3 × llm_score
           → 若 health_tag == "盲打" 则封顶 0.50
```

#### 行为指标（存储在 `details.behavioral_metrics` 中）

| 指标 | 说明 |
|--------|-------------|
| `health_tag` | `healthy` / `blind_edit` / `dead_loop` |
| `total_steps` | 总工具调用数 |
| `backtrack_count` | 智能体回到此前见过的文件状态（SHA256 哈希）的次数 |
| `backtrack_score` | 浪费步数比例：回溯步数 / 总步数 |
| `exploration_ratio` | 访问的唯一状态数 / 总步数 |
| `loop_detected` | 重复动作序列模式的计数 |
| `explicit_undo_count` | `git restore` / `git checkout` / 回退命令数 |
| `edit_total` | 总文件写入操作数 |
| `verification_rate` | 编辑后 3 步内执行 Bash/读取同一文件的比率 |
| `continue_edit_rate` | 编辑后无验证继续编辑的比率 |
| `max_edit_chain` | 对同一文件最长连续编辑链长度 |
| `quadrant` | 任务难度：`linear_progress` / `deep_iteration` / `state_maze` / `compound_difficult` |

#### health_tag 分类

| 标签 | 条件 |
|-----|-----------|
| `dead_loop` | `backtrack_count ≥ 20` 且 `max_edit_chain ≤ 1`，或 `loop_detected ≥ 3` 且 `max_edit_chain ≤ 2` |
| `blind_edit` | `continue_edit_rate ≥ 0.5` 且 `verification_rate < 0.3` |
| `healthy` | 以上均不满足 |

---

### navigation_path_code_type

编码智能体工具调用序列的规则分析。分数固定为 `0.0`，价值完全体现在 `details` 中。

| 指标 | 说明 |
|--------|-------------|
| `tool_usage_pattern` | `explore_act` / `explore_act_explore` / `explore_only` / `act_only` / `empty` |
| `explore_to_act_ratio` | EXPLORE 工具调用 / ACT 工具调用 |
| `verification_rate` | 编辑后执行 Bash/读取同一文件的比率 |
| `search_read_chain_rate` | Grep/Glob 后 3 步内出现 Read 的比率 |
| `post_write_read_count` | 文件写入后被读取的次数 |
| `tool_type_distribution` | `EXPLORE` / `ACT` / `THINK` / `MCP` 计数 |

---

## 输入格式

| 格式 | 标志 | 说明 |
|--------|------|-------------|
| JSON 目录 | `--input-format json`（默认） | 独立轨迹 JSON 文件目录 |
| Ducc JSONL | `--input-format ducc_jsonl` | 单条 JSONL 文件，Claude Code / Ducc 工具使用格式 |

`Trajectory` 类还支持其他格式：BrowserGym pickle、CUGA、Synatra、AgentTrek。

---

## 输出格式

每条轨迹为每个评分器生成一个 `ScorerResult`：

```json
{
  "traj_001": {
    "ReasoningQuality": {
      "score": 0.808,
      "confidence": 0.9,
      "weight": 1.0,
      "details": {
        "reasoning_quality": {
          "task_decomposition": 1.0,
          "observation_reading": 1.0,
          "self_verification": 0.425,
          "strategic_backtracking": null,
          "raw_scores": { ... },
          "llm_scores": { ... }
        },
        "justifications": { ... },
        "behavioral_metrics": {
          "health_tag": "healthy",
          "backtrack_count": 0,
          "verification_rate": 0.0,
          "exploration_ratio": 0.95,
          ...
        }
      }
    }
  }
}
```

---

## 配置

在环境变量中设置 LLM API 密钥：
```bash
export LLM_API_KEY="your-api-key-here"
```

---

## 项目结构

```
AgentDiagnose/
├── evaluate_trajectories.py          # CLI 入口
├── launch_dashboard.sh               # 一站式流水线 + 仪表盘
├── generate_verb_nouns.py            # 从轨迹提取动词-名词对
├── generate_embeddings.py            # 生成 FAISS 语义嵌入
├── generate_tag_cloud.py             # 生成 TF-IDF 标签云
├── evaluator/
│   ├── trajectory.py                 # 轨迹数据模型 + 格式解析器
│   ├── evaluator.py                  # TrajectoryJudge 编排器
│   ├── tool_verb_noun_extractor.py   # 动作短语提取（基于工具调用）
│   └── scorers/
│       ├── base.py                   # BaseScorer、LLMScorer、ScorerResult
│       ├── behavioral_metrics.py     # 第一层：基于规则的行为分析
│       ├── reasoning_quality.py      # 第二+三层：ReasoningQualityScorer
│       ├── objective_quality.py      # ObjectiveQualityScorer
│       ├── navigation_path_scorer.py            # 网页浏览导航评分器
│       ├── navigation_path_scorer_code_type.py  # 编码智能体导航评分器
│       └── prompts/
│           ├── prompt.py             # ObjectiveQuality 提示词
│           └── prompt_enhanced.py    # ReasoningQuality 提示词（含行为证据）
└── dashboard/
    ├── web_dashboard.py
    └── backend/main.py
```

---

## Web 仪表盘

交互式界面，用于探索评估结果：

- **汇总页签**：分数分布与整体统计
- **查看轨迹页签**：逐步轨迹检查
- **嵌入页签**：动词/名词/组合嵌入的 t-SNE 可视化
- **标签云页签**：TF-IDF 加权推理与动作短语云
- **评分器页签**：各评分器分数分布与筛选

跨页签联动：嵌入散点图中的选择会同步传递到标签云和分数直方图。

默认访问地址 `http://localhost:8080`，启动时也会打印隧道 URL。
