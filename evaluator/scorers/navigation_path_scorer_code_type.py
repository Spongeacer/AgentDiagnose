import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from evaluator.scorers.base import BaseScorer, ScorerResult
from evaluator.trajectory import Action, Trajectory

# Tool classification — pure structural grouping, no cognitive interpretation
_EXPLORE_TOOLS = {"Read", "Glob", "Grep", "WebFetch", "WebSearch"}
_ACT_TOOLS     = {"Write", "Edit", "Bash", "NotebookEdit"}
_THINK_TOOLS   = {"Agent", "TodoWrite", "Skill"}


class NavigationPathScorer_CodeType(BaseScorer):
    """
    Behavioral pattern scorer for coding agents (Claude Code, OpenClaw, etc.).

    Boundary: this scorer measures WHAT the agent did — tool call sequences,
    file access patterns, structural phase composition. It produces raw
    observable counts and ratios only. It does NOT judge cognitive quality
    (that is the responsibility of ReasoningQualityScorer) and does NOT
    assess task description quality (that is ObjectiveQualityScorer).

    score is fixed at 0.0; diagnostic value is entirely in `details`.
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "NavigationPath_CodeType",
        description: str = (
            "Behavioral pattern analysis of coding-agent trajectories: "
            "file access patterns, tool-call sequences, phase structure, "
            "and tool-usage completeness metrics."
        ),
        **kwargs,
    ) -> None:
        super().__init__(weight=weight, name=name, description=description)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, trajectory: Trajectory) -> ScorerResult:
        if not trajectory.actions:
            return ScorerResult(
                score=0.0,
                name=self.name,
                description=self.description,
                details={"error": "No actions found"},
                confidence=0.0,
                weight=self.weight,
            )
        metrics = self._analyze(trajectory.actions)
        return ScorerResult(
            score=0.0,
            name=self.name,
            description=self.description,
            details=metrics,
            confidence=0.8,
            weight=self.weight,
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analyze(self, actions: List[Action]) -> Dict[str, Any]:
        tool_sequence: List[str] = []
        tool_type_sequence: List[str] = []

        visited_files: set = set()
        read_files: set = set()
        written_files: set = set()
        revisited_files: set = set()

        last_op_per_file: Dict[str, str] = {}
        repeated_reads = 0
        post_write_reads = 0

        file_access_sequence: List[str] = []
        files_per_dir: Dict[str, set] = defaultdict(set)

        bash_total = 0
        bash_with_desc = 0
        glob_patterns: List[str] = []
        mcp_tools_used: List[str] = []

        search_positions: List[int] = []

        for step_idx, action in enumerate(actions):
            tool = action.action_type or ""
            tool_input = self._parse_tool_input(action)
            tool_type = self._classify_tool(tool)

            tool_sequence.append(tool)
            tool_type_sequence.append(tool_type)

            if tool_type == "MCP":
                mcp_tools_used.append(tool)

            if tool == "Bash":
                bash_total += 1
                if tool_input.get("description", "").strip():
                    bash_with_desc += 1

            if tool == "Glob":
                pattern = tool_input.get("pattern", "")
                if pattern:
                    glob_patterns.append(pattern)
                search_positions.append(step_idx)

            if tool == "Grep":
                search_positions.append(step_idx)

            file_path = self._extract_file_path(tool, tool_input)
            if not file_path:
                continue
            file_path = os.path.normpath(file_path)
            dir_path = os.path.dirname(file_path) or "/"

            if tool == "Read":
                prev_op = last_op_per_file.get(file_path)
                if prev_op == "read":
                    repeated_reads += 1
                elif prev_op == "write":
                    post_write_reads += 1
                last_op_per_file[file_path] = "read"

                if file_path in visited_files:
                    revisited_files.add(file_path)
                visited_files.add(file_path)
                read_files.add(file_path)
                file_access_sequence.append(file_path)
                files_per_dir[dir_path].add(file_path)

            elif tool in ("Write", "Edit", "NotebookEdit"):
                last_op_per_file[file_path] = "write"
                visited_files.add(file_path)
                written_files.add(file_path)
                files_per_dir[dir_path].add(file_path)

            elif tool in ("Grep", "Glob"):
                files_per_dir[dir_path].add(file_path)

        # ---- aggregation ----
        type_counts = Counter(tool_type_sequence)
        explore_count = type_counts.get("EXPLORE", 0)
        act_count     = type_counts.get("ACT", 0)
        think_count   = type_counts.get("THINK", 0)
        mcp_count     = type_counts.get("MCP", 0)
        other_count   = type_counts.get("OTHER", 0)

        dirs_accessed = {os.path.dirname(p) or "/" for p in visited_files}
        phase_runs = self._compute_phase_runs(tool_type_sequence)

        return {
            # --- overview ---
            "total_steps": len(actions),
            # --- file access ---
            "unique_files_accessed": len(visited_files),
            "unique_files_read": len(read_files),
            "unique_files_written": len(written_files),
            "file_revisit_count": len(revisited_files),
            "file_revisit_ratio": (
                round(len(revisited_files) / len(visited_files), 3)
                if visited_files else 0.0
            ),
            "repeated_read_count": repeated_reads,
            "file_access_sequence": file_access_sequence[:60],
            "files_per_directory": {d: len(f) for d, f in sorted(files_per_dir.items())},
            "directory_spread": len(dirs_accessed),
            # --- tool calls ---
            "tool_sequence": tool_sequence,
            "tool_type_distribution": {
                "EXPLORE": explore_count,
                "ACT":     act_count,
                "THINK":   think_count,
                "MCP":     mcp_count,
                "OTHER":   other_count,
            },
            "explore_to_act_ratio": (
                round(explore_count / act_count, 3) if act_count > 0 else None
            ),
            "bash_total": bash_total,
            "bash_with_description_ratio": (
                round(bash_with_desc / bash_total, 3) if bash_total > 0 else None
            ),
            "max_consecutive_bash": self._max_consecutive(tool_sequence, "Bash"),
            "search_operations": explore_count,
            "glob_patterns": glob_patterns,
            "mcp_tools_used": sorted(set(mcp_tools_used)),
            # --- phase structure ---
            "phase_runs": phase_runs,
            # --- coverage ---
            "coverage": self._compute_coverage(
                tool_type_sequence=tool_type_sequence,
                phase_runs=phase_runs,
                read_files=read_files,
                written_files=written_files,
                tool_sequence=tool_sequence,
                search_positions=search_positions,
                post_write_reads=post_write_reads,
            ),
        }

    # ------------------------------------------------------------------
    # Coverage computation
    # ------------------------------------------------------------------

    def _compute_coverage(
        self,
        tool_type_sequence: List[str],
        phase_runs: List[Dict[str, Any]],
        read_files: set,
        written_files: set,
        tool_sequence: List[str],
        search_positions: List[int],
        post_write_reads: int,
    ) -> Dict[str, Any]:
        has_explore = any(t == "EXPLORE" for t in tool_type_sequence)
        has_act     = any(t == "ACT"     for t in tool_type_sequence)

        last_act_idx = max(
            (i for i, t in enumerate(tool_type_sequence) if t == "ACT"), default=-1
        )
        has_read_after_last_act = any(
            t == "EXPLORE" for t in tool_type_sequence[last_act_idx + 1:]
        ) if last_act_idx >= 0 else False

        explore_act_transitions = sum(
            1 for i in range(len(phase_runs) - 1)
            if phase_runs[i]["type"] == "EXPLORE" and phase_runs[i + 1]["type"] == "ACT"
        )

        if not has_explore and not has_act:
            pattern_label = "empty"
        elif has_explore and not has_act:
            pattern_label = "explore_only"
        elif has_act and not has_explore:
            pattern_label = "act_only"
        elif has_act and has_explore and has_read_after_last_act:
            pattern_label = "explore_act_explore"
        else:
            pattern_label = "explore_act"

        write_without_read = written_files - read_files
        file_write_ratio = (
            round(len(written_files) / len(read_files), 3) if read_files else None
        )

        search_total = len(search_positions)
        search_read_chain_count = sum(
            1 for pos in search_positions
            if "Read" in tool_sequence[pos + 1: pos + 4]
        )
        search_read_chain_rate = (
            round(search_read_chain_count / search_total, 3)
            if search_total > 0 else None
        )

        used_tool_types = sorted({t for t in tool_type_sequence if t != "OTHER"})

        return {
            "tool_usage_pattern": pattern_label,
            "has_explore_phase": has_explore,
            "has_act_phase": has_act,
            "has_read_after_last_act": has_read_after_last_act,
            "explore_act_transitions": explore_act_transitions,
            "file_write_ratio": file_write_ratio,
            "write_without_prior_read_count": len(write_without_read),
            "write_without_prior_read_files": sorted(write_without_read),
            "post_write_read_count": post_write_reads,
            "search_total": search_total,
            "search_read_chain_count": search_read_chain_count,
            "search_read_chain_rate": search_read_chain_rate,
            "used_tool_types": used_tool_types,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_tool_input(self, action: Action) -> dict:
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

    def _extract_file_path(self, tool: str, tool_input: dict) -> Optional[str]:
        if tool in ("Read", "Edit", "Write"):
            return tool_input.get("file_path") or tool_input.get("notebook_path")
        if tool == "NotebookEdit":
            return tool_input.get("notebook_path")
        if tool == "Grep":
            return tool_input.get("path") or ""
        if tool == "Glob":
            return tool_input.get("path") or ""
        if tool == "Bash":
            cmd = tool_input.get("command", "")
            matches = re.findall(r'(?:^|\s)(/[\w./\-_]+)', cmd)
            return matches[0] if matches else None
        return None

    def _classify_tool(self, tool: str) -> str:
        if tool in _EXPLORE_TOOLS:
            return "EXPLORE"
        if tool in _ACT_TOOLS:
            return "ACT"
        if tool in _THINK_TOOLS:
            return "THINK"
        if tool.startswith("mcp__"):
            return "MCP"
        return "OTHER"

    def _max_consecutive(self, sequence: List[str], target: str) -> int:
        max_run = cur_run = 0
        for item in sequence:
            if item == target:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        return max_run

    def _compute_phase_runs(self, type_seq: List[str]) -> List[Dict[str, Any]]:
        if not type_seq:
            return []
        runs = []
        cur_type, cur_len = type_seq[0], 1
        for t in type_seq[1:]:
            if t == cur_type:
                cur_len += 1
            else:
                runs.append({"type": cur_type, "length": cur_len})
                cur_type, cur_len = t, 1
        runs.append({"type": cur_type, "length": cur_len})
        return runs
