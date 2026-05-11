"""
Behavioral metrics adapter for coding-agent trajectories.

Bridges Trajectory.actions (one Action per tool_use) to the backtrack_analyzer
analysis pipeline, producing objective behavioral facts that can ground LLM scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional

from evaluator.trajectory import Action, Trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_NAME_MAP: dict[str, tuple[str, ...]] = {
    "read":        ("Read", "read_file", "read"),
    "edit":        ("Edit", "edit_file"),
    "write":       ("Write", "write_file"),
    "bash":        ("Bash", "bash", "run_command"),
    "glob":        ("Glob", "glob_path", "find"),
    "grep":        ("Grep", "grep_content", "codebase_search"),
    "todo":        ("TodoWrite", "todo_write"),
    "agent":       ("Agent", "delegate_subtask"),
    "list_dir":    ("list_dir",),
    "delete_file": ("delete_file",),
    "web_search":  ("web_search", "web_fetch"),
}

UNDO_CMD_PREFIXES: tuple[str, ...] = (
    "git checkout", "git restore", "git reset", "git clean",
    "rm -rf", "rm -r", "mv --",
)

READONLY_BASH_PREFIXES: tuple[str, ...] = (
    "git status", "git log", "git diff", "git show", "git branch", "git remote",
    "ls", "ll", "find ", "cat ", "head ", "tail ", "grep ", "wc", "du ", "pwd", "echo ",
    "which ", "file ", "stat ", "id", "uname", "whoami", "df", "ps ", "netstat", "lsof ",
    "pip3 show", "pip show", "npm list", "yarn list", "npm view", "cargo search",
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict
    tool_use_id: str = ""


@dataclass
class Step:
    step_idx: int
    actions: list[ToolCall]
    observations: list = field(default_factory=list)
    state_snapshot: str = ""
    action_signature: str = ""


@dataclass
class TrajectoryMetrics:
    total_steps: int = 0
    backtrack_count: int = 0
    backtrack_depths: list[int] = field(default_factory=list)
    backtrack_details: list[dict] = field(default_factory=list)
    backtrack_score: float = 0.0
    exploration_ratio: float = 0.0
    explicit_undo_count: int = 0
    loop_detected: int = 0
    effective_backtracks: int = 0
    max_edit_chain: int = 0
    action_repeat_count: int = 0
    edit_total: int = 0
    edit_followed_by_bash: int = 0
    edit_followed_by_read_same: int = 0
    edit_followed_by_edit: int = 0
    verification_rate: float = 0.0
    continue_edit_rate: float = 0.0
    quadrant: str = "linear_progress"
    health_tag: str = "healthy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "backtrack_count": self.backtrack_count,
            "backtrack_depths": self.backtrack_depths,
            "backtrack_score": round(self.backtrack_score, 4),
            "exploration_ratio": round(self.exploration_ratio, 4),
            "explicit_undo_count": self.explicit_undo_count,
            "loop_detected": self.loop_detected,
            "effective_backtracks": self.effective_backtracks,
            "max_edit_chain": self.max_edit_chain,
            "action_repeat_count": self.action_repeat_count,
            "edit_total": self.edit_total,
            "edit_followed_by_bash": self.edit_followed_by_bash,
            "edit_followed_by_read_same": self.edit_followed_by_read_same,
            "edit_followed_by_edit": self.edit_followed_by_edit,
            "verification_rate": round(self.verification_rate, 4),
            "continue_edit_rate": round(self.continue_edit_rate, 4),
            "quadrant": self.quadrant,
            "health_tag": self.health_tag,
        }


# ---------------------------------------------------------------------------
# Tool utilities
# ---------------------------------------------------------------------------

def is_tool(name: str, canonical: str) -> bool:
    return name in TOOL_NAME_MAP.get(canonical, ())


def extract_file_path(inp: dict) -> str:
    raw = ""
    if "edits" in inp and isinstance(inp["edits"], list):
        for edit in inp["edits"]:
            if isinstance(edit, dict):
                path = edit.get("path") or edit.get("file_path")
                if path:
                    raw = path
                    break
    if not raw:
        raw = inp.get("file_path") or inp.get("target_file") or inp.get("path") or ""
    if not raw:
        return ""
    return os.path.normpath(raw).replace("\\", "/")


def get_bash_signature(cmd: str) -> str:
    cmd = cmd.strip()
    if len(cmd) > 40:
        return hashlib.sha256(cmd.encode()).hexdigest()[:16]
    return cmd.replace("\n", " ")


def _parse_tool_input(action: Action) -> dict:
    """Parse tool input dict from Action.action string (ToolName({...}) format)."""
    tool_name = action.action_type or ""
    action_str = action.action or ""
    prefix = f"{tool_name}("
    if action_str.startswith(prefix) and action_str.endswith(")"):
        json_str = action_str[len(prefix):-1]
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        return json.loads(action_str)
    except Exception:
        return {}


def _build_action_signature(actions: list[ToolCall]) -> str:
    sig_parts: list[str] = []
    for a in actions:
        name = a.name
        inp = a.input
        fp = extract_file_path(inp)

        if is_tool(name, "read") and fp:
            limit = inp.get("limit", "")
            offset = inp.get("offset", "")
            extra = f"[o={offset},l={limit}]" if (limit or offset) else ""
            sig_parts.append(f"Read:{fp}{extra}")
        elif is_tool(name, "edit") and fp:
            sig_parts.append(f"Edit:{fp}")
        elif is_tool(name, "write") and fp:
            sig_parts.append(f"Write:{fp}")
        elif is_tool(name, "bash") and "command" in inp:
            sig_parts.append(f"Bash:{get_bash_signature(inp['command'])}")
        elif is_tool(name, "glob") and "pattern" in inp:
            sig_parts.append(f"Glob:{inp['pattern']}")
        elif is_tool(name, "grep") and "pattern" in inp:
            sig_parts.append(f"Grep:{inp['pattern']}")
        elif is_tool(name, "todo") and "todos" in inp:
            count = len(inp["todos"]) if isinstance(inp["todos"], list) else 0
            sig_parts.append(f"TodoWrite:{count}")
        elif is_tool(name, "agent") and "query" in inp:
            sig_parts.append(f"Agent:{inp['query'][:30]}")
        else:
            first_val = next(
                (str(v)[:30] for v in inp.values() if v and isinstance(v, str)), ""
            )
            sig_parts.append(f"{name}:{first_val}")

    return "|".join(sorted(sig_parts))


# ---------------------------------------------------------------------------
# Adapter: Trajectory.actions → Step list
# ---------------------------------------------------------------------------

def trajectory_to_steps(trajectory: Trajectory) -> list[Step]:
    steps: list[Step] = []
    for idx, action in enumerate(trajectory.actions):
        tc = ToolCall(
            name=action.action_type or "",
            input=_parse_tool_input(action),
        )
        step = Step(
            step_idx=idx,
            actions=[tc],
            action_signature=_build_action_signature([tc]),
        )
        steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

def _compute_step_snapshot(
    step: Step, prev_acc: dict[str, Any]
) -> tuple[str, dict[str, Any], bool]:
    acc: dict[str, Any] = {
        "files_read":     set(prev_acc.get("files_read", set())),
        "files_modified": set(prev_acc.get("files_modified", set())),
        "commands":       list(prev_acc.get("commands", [])),
        "searches":       set(prev_acc.get("searches", set())),
        "edit_counter":   dict(prev_acc.get("edit_counter", {})),
    }

    step_focus_files: list[str] = []
    step_search_sigs: list[tuple] = []
    has_progress = False

    for action in step.actions:
        name = action.name
        inp = action.input
        fp = extract_file_path(inp)

        if is_tool(name, "read") and fp:
            limit = inp.get("limit", "")
            offset = inp.get("offset", "")
            fp_key = f"{fp}[o={offset},l={limit}]" if (limit or offset) else fp
            step_focus_files.append(fp_key)
            if fp not in acc["files_read"]:
                has_progress = True
            acc["files_read"].add(fp)

        elif (is_tool(name, "edit") or is_tool(name, "write")) and fp:
            acc["edit_counter"][fp] = acc["edit_counter"].get(fp, 0) + 1
            fp_key = f"{fp}#edit{acc['edit_counter'][fp]}"
            step_focus_files.append(fp_key)
            acc["files_modified"].add(fp)
            has_progress = True

        elif is_tool(name, "list_dir") and fp:
            step_focus_files.append(fp)

        elif is_tool(name, "delete_file") and fp:
            acc["files_modified"].add(fp)
            has_progress = True

        elif is_tool(name, "bash") and "command" in inp:
            cmd = inp["command"].strip()
            if cmd:
                acc["commands"].append(cmd.split()[0])

        elif is_tool(name, "glob") and "pattern" in inp:
            sig = ("Glob", inp["pattern"])
            step_search_sigs.append(sig)
            if sig not in acc["searches"]:
                has_progress = True
            acc["searches"].add(sig)

        elif is_tool(name, "grep") and "pattern" in inp:
            sig = ("Grep", inp["pattern"], inp.get("path", ""))
            step_search_sigs.append(sig)
            if sig not in acc["searches"]:
                has_progress = True
            acc["searches"].add(sig)

        elif is_tool(name, "web_search"):
            has_progress = True

    snapshot_obj = {
        "focus_files":   tuple(sorted(set(step_focus_files))),
        "search_sigs":   tuple(sorted(set(step_search_sigs))),
        "action_types":  tuple(a.name for a in step.actions),
        "action_sig":    step.action_signature,
    }
    snapshot_str = hashlib.sha256(
        json.dumps(snapshot_obj, sort_keys=True, default=str).encode()
    ).hexdigest()

    return snapshot_str, acc, has_progress


def _is_todo_only(step: Step) -> bool:
    return bool(step.actions) and all(is_tool(a.name, "todo") for a in step.actions)


def _is_readonly_bash(step: Step) -> bool:
    bash_actions = [a for a in step.actions if is_tool(a.name, "bash")]
    if not bash_actions or len(bash_actions) != len(step.actions):
        return False
    danger = (
        "rm ", "mv ", "cp ", "> ", ">> ", "| tee", "chmod ", "chown ",
        "mkdir ", "touch ", "rmdir ", "git checkout", "git reset",
        "git restore", "git clean", "git commit", "git push", "git pull",
    )
    for a in bash_actions:
        cmd = a.input.get("command", "").strip()
        if not cmd:
            return False
        lower = cmd.lower()
        if any(m in lower for m in danger):
            return False
        if "&&" in cmd or ";" in cmd or "||" in cmd:
            return False
        main_cmd = lower.split("|")[0].strip()
        if not any(main_cmd.startswith(p.lower()) for p in READONLY_BASH_PREFIXES):
            return False
    return True


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def _detect_backtracks(
    steps: list[Step],
) -> tuple[int, list[int], list[dict], float, float]:
    acc_state: dict[str, Any] = {
        "files_read": set(), "files_modified": set(),
        "commands": [], "searches": set(), "edit_counter": {},
    }
    step_progress: list[bool] = []
    for step in steps:
        snapshot, acc_state, has_progress = _compute_step_snapshot(step, acc_state)
        step.state_snapshot = snapshot
        step_progress.append(has_progress)

    visited: dict[str, int] = {}
    bt_count = 0
    bt_depths: list[int] = []
    bt_details: list[dict] = []

    for i, step in enumerate(steps):
        snap = step.state_snapshot
        is_new_state = snap not in visited
        if is_new_state:
            visited[snap] = i

        if _is_todo_only(step) or _is_readonly_bash(step):
            continue

        if is_new_state:
            continue
        prev_idx = visited[snap]
        skip = i - prev_idx
        if skip <= 2:
            continue
        if any(step_progress[j] for j in range(prev_idx + 1, i)):
            continue
        bt_depths.append(skip)
        bt_count += 1
        bt_details.append({
            "step": i,
            "previous_step": prev_idx,
            "skip_steps": skip,
            "action_types": tuple(a.name for a in step.actions),
        })

    total = len(steps)
    score = sum(bt_depths) / total if total else 0.0
    ratio = len(visited) / total if total else 0.0
    return bt_count, bt_depths, bt_details, score, ratio


def _detect_explicit_undo(steps: list[Step]) -> int:
    count = 0
    for step in steps:
        for a in step.actions:
            if is_tool(a.name, "bash") and "command" in a.input:
                cmd = a.input["command"].strip().lower()
                if any(cmd.startswith(u) for u in UNDO_CMD_PREFIXES):
                    count += 1
    return count


def _detect_loops(steps: list[Step]) -> int:
    total = len(steps)
    if total < 4:
        return 0
    sigs = [s.action_signature for s in steps]
    loop_count = 0
    max_cycle = min(6, total // 2 + 1)
    for cycle_len in range(2, max_cycle):
        for start in range(total - cycle_len * 2 + 1):
            first = tuple(sigs[start:start + cycle_len])
            second = tuple(sigs[start + cycle_len:start + cycle_len * 2])
            if first == second and len(set(first)) > 1:
                loop_count += 1
    return loop_count


def _compute_edit_chain(steps: list[Step]) -> int:
    max_chain = 0
    current_file: Optional[str] = None
    current_chain = 0
    for step in steps:
        edit_files = {
            extract_file_path(a.input)
            for a in step.actions
            if is_tool(a.name, "edit") or is_tool(a.name, "write")
        }
        edit_files.discard("")
        if len(edit_files) == 1:
            fp = next(iter(edit_files))
            if fp == current_file:
                current_chain += 1
            else:
                max_chain = max(max_chain, current_chain)
                current_file = fp
                current_chain = 1
        else:
            max_chain = max(max_chain, current_chain)
            current_file = None
            current_chain = 0
    return max(max_chain, current_chain)


def _compute_action_repeat(steps: list[Step]) -> int:
    if not steps:
        return 0
    sigs = [s.action_signature for s in steps]
    count = 0
    for i in range(len(sigs)):
        window = [sigs[j] for j in range(max(0, i - 3), i)]
        if sum(1 for s in window if s == sigs[i]) >= 2:
            count += 1
    return count


def _compute_verification_behavior(
    steps: list[Step],
) -> tuple[int, int, int, int, float, float]:
    edit_total = 0
    edit_followed_by_bash = 0
    edit_followed_by_read_same = 0
    edit_followed_by_edit = 0

    for i, step in enumerate(steps):
        edit_actions = [
            a for a in step.actions
            if is_tool(a.name, "edit") or is_tool(a.name, "write")
        ]
        if not edit_actions:
            continue
        edit_fps = {extract_file_path(a.input) for a in edit_actions}
        edit_fps.discard("")
        has_bash_same_step = any(is_tool(a.name, "bash") for a in step.actions)
        edit_total += 1
        if i + 1 >= len(steps):
            continue
        next_actions = steps[i + 1].actions
        next_has_bash = any(is_tool(a.name, "bash") for a in next_actions)
        if next_has_bash and not has_bash_same_step:
            edit_followed_by_bash += 1
            continue
        next_has_edit = any(
            is_tool(a.name, "edit") or is_tool(a.name, "write") for a in next_actions
        )
        if next_has_edit:
            edit_followed_by_edit += 1
            continue
        next_read_fps = {
            extract_file_path(a.input)
            for a in next_actions if is_tool(a.name, "read")
        }
        next_read_fps.discard("")
        if edit_fps and next_read_fps and (edit_fps & next_read_fps):
            edit_followed_by_read_same += 1

    vr = (edit_followed_by_bash + edit_followed_by_read_same) / edit_total if edit_total else 0.0
    cr = edit_followed_by_edit / edit_total if edit_total else 0.0
    return edit_total, edit_followed_by_bash, edit_followed_by_read_same, edit_followed_by_edit, vr, cr


def _grade_difficulty(m: TrajectoryMetrics) -> None:
    bt_high = m.effective_backtracks >= 2
    ec_high = m.max_edit_chain >= 2
    if not bt_high and not ec_high:
        m.quadrant = "linear_progress"
    elif not bt_high and ec_high:
        m.quadrant = "deep_iteration"
    elif bt_high and not ec_high:
        m.quadrant = "state_maze"
    else:
        m.quadrant = "compound_difficult"

    if (m.backtrack_count >= 20 and m.max_edit_chain <= 1) or (m.loop_detected >= 3 and m.max_edit_chain <= 2):
        m.health_tag = "dead_loop"
    elif m.max_edit_chain >= 4 and m.verification_rate < 0.1:
        m.health_tag = "blind_edit"
    else:
        m.health_tag = "healthy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_behavioral_metrics(trajectory: Trajectory) -> TrajectoryMetrics:
    """
    Compute behavioral metrics for a trajectory.
    Returns a TrajectoryMetrics with all fields populated.
    """
    steps = trajectory_to_steps(trajectory)
    m = TrajectoryMetrics()
    m.total_steps = len(steps)

    (
        m.backtrack_count, m.backtrack_depths, m.backtrack_details,
        m.backtrack_score, m.exploration_ratio,
    ) = _detect_backtracks(steps)

    m.explicit_undo_count = _detect_explicit_undo(steps)
    m.loop_detected = _detect_loops(steps)
    m.effective_backtracks = m.backtrack_count + m.explicit_undo_count + min(m.loop_detected, 3)

    m.max_edit_chain = _compute_edit_chain(steps)
    m.action_repeat_count = _compute_action_repeat(steps)

    (
        m.edit_total, m.edit_followed_by_bash,
        m.edit_followed_by_read_same, m.edit_followed_by_edit,
        m.verification_rate, m.continue_edit_rate,
    ) = _compute_verification_behavior(steps)

    _grade_difficulty(m)
    return m
