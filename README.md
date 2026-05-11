# AgentDiagnose

An open toolkit for evaluating and diagnosing LLM agent trajectories. Supports coding agents (Claude Code / Ducc format) and web-browsing agents, combining rule-based behavioral analysis with LLM judgment for grounded, reproducible evaluation.

## Architecture

Evaluation uses a three-layer pipeline for the `reasoning_quality` scorer:

```
Trajectory.actions (raw tool call sequence)
         │
         ▼
[Layer 1] behavioral_metrics.py
          • SHA256 hash-based backtrack detection
          • exploration_ratio (unique states / total steps)
          • verification_rate (Edit → Bash/Read-same-file)
          • loop detection (repeated action-sequence patterns)
          → TrajectoryMetrics (health_tag: healthy / blind_edit / dead_loop)
         │
         ▼
[Layer 2] LLM (Gemini API via LiteLLM)
          • Behavioral evidence injected into prompt as ground truth
          • LLM scores 4 sub-dimensions on 1–4 scale
         │
         ▼
[Layer 3] Post-processing rules
          • Cap/override LLM scores based on behavioral facts
          • Blend verification_rate into self_verification score
          → ScorerResult (0–1 score + full details)
```

This design prevents the LLM from being fooled by fluent reasoning text that doesn't match actual behavior (e.g., an agent that claims to verify but never reads files after editing).

---

## Quick Start

### Evaluate trajectories

```bash
# Coding agent trajectories (Ducc/Claude Code JSONL format)
python evaluate_trajectories.py \
  --input path/to/data.jsonl \
  --input-format ducc_jsonl \
  --scorers reasoning_quality objective_quality navigation_path_code_type \
  --output-json results.json

# Web-browsing agent trajectories (directory of JSON files)
python evaluate_trajectories.py \
  --input examples/sample_trajectories \
  --scorers reasoning_quality objective_quality navigation_path \
  --output-json results.json

# Dry run — estimate token usage without calling the model
python evaluate_trajectories.py \
  --input path/to/data.jsonl \
  --input-format ducc_jsonl \
  --scorers reasoning_quality \
  --dry-run
```

### All-in-one dashboard

```bash
./launch_dashboard.sh
```

Runs the full pipeline (verb-noun extraction → tag clouds → embeddings → evaluation) and launches an interactive web dashboard at `http://localhost:8080`.

---

## Scorers

| Scorer | Type | Target | Description |
|--------|------|--------|-------------|
| `reasoning_quality` | LLM + rules | Coding & web agents | 4 sub-dimension quality score with behavioral grounding |
| `objective_quality` | LLM | Any | Task description quality (specificity / actionability) |
| `navigation_path` | Rule-based | Web-browsing agents | Navigation path diagnostic metrics |
| `navigation_path_code_type` | Rule-based | Coding agents | Tool call pattern analysis |

---

### reasoning_quality

Combines LLM judgment with rule-based constraints derived from actual tool calls.

#### Sub-dimensions

| Dimension | Scored by | Description |
|-----------|-----------|-------------|
| `task_decomposition` | LLM only | How well the agent broke the task into manageable steps |
| `observation_reading` | LLM only | How accurately the agent interpreted tool outputs |
| `self_verification` | Rules + LLM blend | Whether the agent verified edits with reads/tests |
| `strategic_backtracking` | Rules + LLM | Whether the agent recognized wrong paths and corrected course. `null` when not applicable. |

#### Post-processing rules

**strategic_backtracking**

| Condition | Action |
|-----------|--------|
| `health_tag == "死循环"` | Cap at 0.25 (behavioral ground truth overrides LLM) |
| `health_tag == "盲打"` and `backtrack_count == 0` | Rule score: `min(1 − continue_edit_rate, 0.40)` |
| `health_tag == "盲打"` and `backtrack_count ≥ 2` | Cap at 0.50 |
| `health_tag == "健康"` and `backtrack_count == 0` and `loop_detected == 0` and `exploration_ratio ≥ 0.8` | `null` (N/A — clean linear path, dimension not applicable) |
| All other cases | LLM score as-is |

**self_verification**

```
rule_score = min(1.0, verification_rate × 1.25)   if edit_total ≥ 3
           = 0.5 (neutral)                          if edit_total < 3
final      = 0.7 × rule_score + 0.3 × llm_score
           → capped at 0.50 if health_tag == "盲打"
```

#### Behavioral metrics (stored in `details.behavioral_metrics`)

| Metric | Description |
|--------|-------------|
| `health_tag` | `healthy` / `blind_edit` / `dead_loop` |
| `total_steps` | Total tool calls |
| `backtrack_count` | Times the agent revisited a previously-seen file state (SHA256 hash) |
| `backtrack_score` | Wasted steps ratio: backtrack steps / total steps |
| `exploration_ratio` | Unique states visited / total steps |
| `loop_detected` | Count of repeated action-sequence patterns |
| `explicit_undo_count` | `git restore` / `git checkout` / revert commands |
| `edit_total` | Total file-write operations |
| `verification_rate` | Edit → Bash/Read-same-file within 3 steps |
| `continue_edit_rate` | Edit → Edit (no verification in between) |
| `max_edit_chain` | Longest consecutive-edit chain on the same file |
| `quadrant` | Task difficulty: `linear_progress` / `deep_iteration` / `state_maze` / `compound_difficult` |

#### health_tag classification

| Tag | Condition |
|-----|-----------|
| `dead_loop` | `backtrack_count ≥ 20` and `max_edit_chain ≤ 1`, OR `loop_detected ≥ 3` and `max_edit_chain ≤ 2` |
| `blind_edit` | `continue_edit_rate ≥ 0.5` and `verification_rate < 0.3` |
| `healthy` | Neither of the above |

---

### navigation_path_code_type

Rule-based analysis of coding agent tool call sequences. Score is always `0.0` — value is in `details`.

| Metric | Description |
|--------|-------------|
| `tool_usage_pattern` | `explore_act` / `explore_act_explore` / `explore_only` / `act_only` / `empty` |
| `explore_to_act_ratio` | EXPLORE tool calls / ACT tool calls |
| `verification_rate` | Edit → Bash/Read-same-file rate |
| `search_read_chain_rate` | Grep/Glob followed by Read within 3 steps |
| `post_write_read_count` | Times a file was read after being written |
| `tool_type_distribution` | `EXPLORE` / `ACT` / `THINK` / `MCP` counts |

---

## Input Formats

| Format | Flag | Description |
|--------|------|-------------|
| JSON directory | `--input-format json` (default) | Directory of individual trajectory JSON files |
| Ducc JSONL | `--input-format ducc_jsonl` | Single JSONL file in Claude Code / Ducc tool-use format |

Other supported formats (via the `Trajectory` class): BrowserGym pickle, CUGA, Synatra, AgentTrek.

---

## Output Format

Each trajectory produces a `ScorerResult` per scorer:

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

## Configuration

Set your LLM API key in the environment:
```bash
export LLM_API_KEY="your-api-key-here"
```

---

## Project Structure

```
AgentDiagnose/
├── evaluate_trajectories.py          # CLI entry point
├── launch_dashboard.sh               # All-in-one pipeline + dashboard
├── generate_verb_nouns.py            # Extract verb-noun pairs from trajectories
├── generate_embeddings.py            # Generate FAISS semantic embeddings
├── generate_tag_cloud.py             # Generate TF-IDF tag clouds
├── evaluator/
│   ├── trajectory.py                 # Trajectory data model + format parsers
│   ├── evaluator.py                  # TrajectoryJudge orchestrator
│   ├── tool_verb_noun_extractor.py   # Action phrase extraction (tool-call-based)
│   └── scorers/
│       ├── base.py                   # BaseScorer, LLMScorer, ScorerResult
│       ├── behavioral_metrics.py     # Layer 1: rule-based behavioral analysis
│       ├── reasoning_quality.py      # Layer 2+3: ReasoningQualityScorer
│       ├── objective_quality.py      # ObjectiveQualityScorer
│       ├── navigation_path_scorer.py            # Web-browsing nav scorer
│       ├── navigation_path_scorer_code_type.py  # Coding agent nav scorer
│       └── prompts/
│           ├── prompt.py             # ObjectiveQuality prompts
│           └── prompt_enhanced.py    # ReasoningQuality prompts (with behavioral evidence)
└── dashboard/
    ├── web_dashboard.py
    └── backend/main.py
```

---

## Web Dashboard

Interactive interface for exploring evaluation results:

- **Summary tab**: score distributions and overall statistics
- **View Trajectory tab**: step-by-step trajectory inspection
- **Embeddings tab**: t-SNE visualization of verb/noun/pair embeddings
- **Tag Cloud tabs**: TF-IDF weighted reasoning and action phrase clouds
- **Scorer tabs**: per-scorer score distributions and filtering

Cross-tab filtering: selections in the embedding scatter plot propagate to tag clouds and score histograms.

Access at `http://localhost:8080` (default) or the tunnel URL printed on startup.
