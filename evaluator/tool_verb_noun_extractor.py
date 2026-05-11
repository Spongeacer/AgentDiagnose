"""
Tool-call-based verb-noun extraction for coding agent (Ducc/Claude Code) trajectories.

Instead of parsing natural-language reasoning with spaCy (which assumes English text),
this module extracts structured (verb, noun) pairs directly from the tool_name and
tool_input of each Action in a Trajectory.

  Verb  ← tool name  (e.g. Edit → "edit",  Bash/git commit → "git")
  Noun  ← most informative argument  (e.g. file basename, git subcommand, pattern)

Output fills the same Action fields used by the existing pipeline:
    action.output_root_verb         str
    action.output_root_noun         str
    action.output_verb_noun_pairs   List[Tuple[str, str]]
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public lookup tables  (importable by callers who need verb/noun metadata)
# ---------------------------------------------------------------------------

# Structured tool name → verb
TOOL_VERB_MAP: dict[str, str] = {
    "Read":             "read",
    "Edit":             "edit",
    "Write":            "write",
    "Glob":             "find",
    "Grep":             "search",
    "TodoWrite":        "plan",
    "Agent":            "delegate",
    "Skill":            "invoke",
    "WebFetch":         "fetch",
    "WebSearch":        "search_web",
    "NotebookEdit":     "edit",
    # Older / aliased tool names sometimes seen in Ducc logs
    "read_file":        "read",
    "edit_file":        "edit",
    "write_file":       "write",
    "bash":             "run",
    "run_command":      "run",
    "find":             "find",
    "codebase_search":  "search",
}

# Bash command first-token → verb  (None = skip this command entirely)
BASH_VERB_MAP: dict[str, Optional[str]] = {
    # Version-control
    "git":              "git",
    # Python runtimes
    "python3":          "run",
    "python":           "run",
    "python2":          "run",
    "pytest":           "test",
    "py.test":          "test",
    # JS/Node package managers
    "npm":              "npm",
    "yarn":             "npm",
    "pnpm":             "npm",
    "npx":              "npm",
    # Compiled-language toolchains
    "go":               "go",
    "cargo":            "cargo",
    "rustc":            "build",
    "javac":            "build",
    "make":             "build",
    "cmake":            "build",
    "mvn":              "build",
    "gradle":           "build",
    # Container / orchestration
    "kubectl":          "kubectl",
    "helm":             "helm",
    "docker":           "docker",
    "docker-compose":   "docker",
    # HTTP clients
    "curl":             "fetch",
    "wget":             "fetch",
    # Search utilities
    "grep":             "search",
    "rg":               "search",
    "ag":               "search",
    "find":             "find",
    # File-system inspection
    "ls":               "list",
    "ll":               "list",
    "la":               "list",
    "cat":              "read_file",
    "head":             "read_file",
    "tail":             "read_file",
    "less":             "read_file",
    "more":             "read_file",
    # Text processing
    "sed":              "transform",
    "awk":              "transform",
    "jq":               "transform",
    # Remote access
    "ssh":              "connect",
    "sshpass":          "connect",
    "scp":              "transfer",
    "rsync":            "transfer",
    "sftp":             "transfer",
    # Package managers
    "pip":              "pip",
    "pip3":             "pip",
    "uv":               "pip",
    # File-system mutation
    "rm":               "delete",
    "mkdir":            "create",
    "touch":            "create",
    "cp":               "copy",
    "mv":               "move",
    # Introspection
    "wc":               "count",
    "which":            "check",
    "type":             "check",
    "diff":             "compare",
    # Shell runners / multiplexers
    "tmux":             "run",
    "screen":           "run",
    "bash":             "run",
    "sh":               "run",
    "zsh":              "run",
    "nohup":            "run",
    "xargs":            "run",
    # Android / mobile
    "adb":              "adb",
    # System package managers
    "brew":             "brew",
    "apt":              "apt",
    "apt-get":          "apt",
    "yum":              "apt",
    # Navigation
    "cd":               "navigate",
    # Noise — skip entirely
    "echo":             None,
    "printf":           None,
    "export":           None,
    "source":           None,
    "sleep":            None,
    "wait":             None,
    "true":             None,
    "false":            None,
    "chmod":            None,
    "chown":            None,
    "#":                None,
}

# File extension → human-readable category (used by downstream tag-cloud / TF-IDF)
EXT_CATEGORY: dict[str, str] = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".tsx":   "typescript",
    ".jsx":   "javascript",
    ".go":    "go",
    ".java":  "java",
    ".kt":    "kotlin",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".c":     "c",
    ".h":     "header",
    ".sh":    "shell",
    ".yaml":  "config",
    ".yml":   "config",
    ".toml":  "config",
    ".json":  "json",
    ".xml":   "xml",
    ".md":    "markdown",
    ".txt":   "text",
    ".html":  "html",
    ".css":   "css",
    ".sql":   "sql",
    ".proto": "proto",
}

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Tokens that are shell metacharacters — strip from the argument list
_SHELL_OPERATORS: frozenset[str] = frozenset({
    "<<", "<", ">", ">>", "|", "||", "&&", "&", ";",
    "2>", "2>>", "&>", "2>&1", ">&2", "1>&2",
})

# Operators that introduce a *new* command — truncate the token list here
_CHAIN_OPS: frozenset[str] = frozenset({"&&", "||", ";"})

# SSH flags that are immediately followed by a value
_SSH_SKIP_VALUE_FLAGS: frozenset[str] = frozenset({
    "-p", "-i", "-l", "-o", "-e", "-b", "-c",
    "-D", "-E", "-F", "-I", "-J", "-L", "-m",
    "-Q", "-R", "-S", "-w", "-W",
})

# Tokens in an ssh command that are not hostnames
_SSH_NON_HOST_TOKENS: frozenset[str] = frozenset({
    "ssh", "sshpass", "StrictHostKeyChecking=no", "BatchMode=yes", "no", "yes",
})

# Regex for a bare environment-variable assignment line
_BARE_ENV_RE = re.compile(r'^[A-Za-z_]\w*=')


# ---------------------------------------------------------------------------
# Lightweight noun helpers
# ---------------------------------------------------------------------------

def _file_noun(path: str) -> str:
    """Return a meaningful noun for a file path."""
    basename = os.path.basename(path.rstrip("/"))
    if not basename:
        return ""
    _generic = {
        "__init__.py", "index.js", "index.ts", "index.tsx",
        "main.py", "mod.rs", "lib.rs", "app.py", "app.ts", "app.js",
    }
    if basename in _generic:
        parent = os.path.basename(os.path.dirname(path))
        return f"{parent}/{basename}" if parent and parent not in ("", ".") else basename
    return basename


def _pattern_noun(pattern: str) -> str:
    """Trim a glob/regex pattern to a compact, readable noun (≤ 40 chars)."""
    cleaned = re.sub(r"^\*\*/", "", pattern)
    return cleaned[:40].strip() if cleaned else ""


def _first_non_flag(tokens: List[str]) -> str:
    """Return the first token that is not a flag."""
    return next((t for t in tokens if not t.startswith("-")), "")


# ---------------------------------------------------------------------------
# Bash extraction — helpers
# ---------------------------------------------------------------------------

def _first_command_line(command: str) -> Tuple[str, bool]:
    """Return (first_meaningful_line, is_heredoc)."""
    is_heredoc = bool(re.search(r"<<\s*['\"]?\w+['\"]?", command))

    for raw_line in re.split(r"[;\n]", command):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _BARE_ENV_RE.match(line):
            remainder = re.sub(r'(?:^|\s)[A-Za-z_]\w*=[^\s]*', '', line).strip()
            if not remainder:
                continue
        return line, is_heredoc

    return "", is_heredoc


def _build_rest(tokens: List[str]) -> List[str]:
    """Return argument tokens after the command name, truncated at chain operators."""
    rest: List[str] = []
    for t in tokens[1:]:
        if t in _CHAIN_OPS:
            break
        if t not in _SHELL_OPERATORS and not t.startswith(("<", ">", "|")):
            rest.append(t)
    return rest


def _connect_noun(rest: List[str]) -> str:
    """Extract remote host from ssh / sshpass arguments."""
    clean: List[str] = []
    skip_next = False
    for t in rest:
        if skip_next:
            skip_next = False
            continue
        if t in _SSH_SKIP_VALUE_FLAGS:
            skip_next = True
            continue
        clean.append(t)

    for t in clean:
        if "@" in t and not t.startswith("-"):
            return t.split("@")[-1][:40]
    for t in clean:
        if re.match(r'\d+\.\d+\.\d+\.\d+', t):
            return t[:40]
    for t in clean:
        if not t.startswith("-") and "." in t and not t.startswith(".") \
                and t not in _SSH_NON_HOST_TOKENS:
            return t[:40]
    return "host"


# ---------------------------------------------------------------------------
# Bash extraction — main dispatch
# ---------------------------------------------------------------------------

def _bash_extract(command: str) -> List[Tuple[str, str]]:
    """Extract [(verb, noun)] from a raw bash command string."""
    first_line, is_heredoc = _first_command_line(command)
    if not first_line:
        return []

    first_line = re.sub(r"^\s*([A-Za-z_]\w*=\S*\s+)+", "", first_line).strip()

    try:
        tokens = shlex.split(first_line)
    except ValueError:
        tokens = first_line.split()

    if not tokens:
        return []

    raw_token = tokens[0]
    first_token = os.path.basename(raw_token.lstrip("./"))

    if not first_token and raw_token.startswith("/"):
        return [("run", os.path.basename(raw_token))]

    verb = BASH_VERB_MAP.get(first_token)
    if verb is None:
        if raw_token.startswith("/") or raw_token.startswith("$"):
            return [("run", os.path.basename(raw_token.rstrip("/")))]
        if first_token.endswith((".sh", ".py", ".js", ".ts", ".rb")):
            return [("run", first_token)]
        return []

    rest_raw = tokens[1:]
    rest = _build_rest(tokens)

    if verb == "adb":
        subcmd = _first_non_flag(rest)
        return [("adb", subcmd or "device")]

    elif verb in ("apt", "brew"):
        non_flag = [t for t in rest if not t.startswith("-")]
        subcmd = non_flag[0] if non_flag else "install"
        pkg    = non_flag[1] if len(non_flag) > 1 else ""
        noun   = f"{subcmd}_{pkg}" if pkg else subcmd
        return [(verb, noun[:40])]

    elif verb == "build":
        target = _first_non_flag(rest)
        return [("build", target or ".")]

    elif verb == "check":
        return [("check", _first_non_flag(rest))]

    elif verb == "compare":
        non_flag = [t for t in rest if not t.startswith("-")]
        files = [os.path.basename(t) for t in non_flag[:2]]
        return [("compare", " vs ".join(files) if files else "files")]

    elif verb == "connect":
        return [("connect", _connect_noun(rest))]

    elif verb in ("copy", "create", "move"):
        non_flag = [t for t in rest if not t.startswith("-")]
        target = non_flag[-1] if non_flag else ""
        return [(verb, os.path.basename(target.rstrip("/")))] if target else []

    elif verb == "count":
        target = _first_non_flag(rest)
        return [("count", os.path.basename(target))] if target else [("count", "output")]

    elif verb == "delete":
        target = _first_non_flag(rest)
        return [("delete", os.path.basename(target.rstrip("/")))] if target else []

    elif verb == "fetch":
        http_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        url = next(
            (t for t in rest if t.startswith("http") or ("/" in t and not t.startswith("-"))),
            ""
        )
        if not url:
            url = next((t for t in rest if not t.startswith("-") and t not in http_methods), "")
        m = re.search(r"https?://([^/]+)", url)
        noun = m.group(1) if m else os.path.basename(url)
        return [("fetch", noun[:30])] if noun else [("fetch", "url")]

    elif verb == "find":
        name_match = re.search(r"-name\s+['\"]?(\S+?)['\"]?(?:\s|$)", first_line)
        if name_match:
            return [("find", name_match.group(1))]
        target = _first_non_flag(rest)
        return [("find", os.path.basename(target.rstrip("/")))] if target else []

    elif verb == "git":
        subcmd = _first_non_flag(rest)
        return [("git", subcmd)]

    elif verb in ("go", "cargo", "helm", "docker", "npm", "pip"):
        subcmd = _first_non_flag(rest)
        return [(verb, subcmd or "")]

    elif verb == "kubectl":
        subcmd   = _first_non_flag(rest)
        resource = next((t for t in rest if not t.startswith("-") and t != subcmd), "")
        noun     = f"{subcmd}_{resource}".strip("_") if resource else subcmd
        return [("kubectl", noun)] if noun else []

    elif verb == "list":
        target = _first_non_flag(rest) or "."
        return [("list", os.path.basename(target.rstrip("/")) or ".")]

    elif verb == "navigate":
        target = _first_non_flag(rest)
        noun = os.path.basename(target.rstrip("/")) or target or "."
        return [("navigate", noun)]

    elif verb == "query":
        target = _first_non_flag(rest)
        return [("query", target[:40])] if target else [("query", first_token)]

    elif verb == "read_file":
        if is_heredoc:
            out_match = re.search(r">+\s*(\S+)", first_line)
            if out_match:
                return [("write", os.path.basename(out_match.group(1)))]
            return [("write", "inline_file")]
        target = _first_non_flag(rest)
        return [("read_file", os.path.basename(target))] if target else []

    elif verb == "run":
        if "-c" in rest_raw or is_heredoc:
            return [("run", "inline_script")]
        if "-m" in rest:
            idx    = rest.index("-m")
            module = rest[idx + 1] if idx + 1 < len(rest) else ""
            if module in ("pytest", "unittest", "doctest"):
                target = next((t for t in rest[idx + 2:] if not t.startswith("-")), "tests")
                return [("test", os.path.basename(target.rstrip("/")))]
            return [("run", module or "module")]
        arg = _first_non_flag(rest)
        return [("run", os.path.basename(arg))] if arg else []

    elif verb == "search":
        non_flag = [t for t in rest if not t.startswith("-")]
        pattern  = non_flag[0] if non_flag else ""
        target   = non_flag[1] if len(non_flag) > 1 else ""
        if target:
            return [("search", os.path.basename(target.rstrip("/")))]
        return [("search", pattern[:30])] if pattern else []

    elif verb == "test":
        if "-m" in rest:
            idx    = rest.index("-m")
            module = rest[idx + 1] if idx + 1 < len(rest) else ""
            if module in ("pytest", "unittest", "doctest"):
                target = next((t for t in rest[idx + 2:] if not t.startswith("-")), "tests")
                return [("test", os.path.basename(target.rstrip("/")))]
        arg = _first_non_flag(rest)
        return [("test", os.path.basename(arg.rstrip("/")) or "tests")]

    elif verb == "transfer":
        non_flag = [t for t in rest if not t.startswith("-")]
        dest = non_flag[-1] if non_flag else ""
        return [("transfer", os.path.basename(dest.rstrip("/")))] if dest else [("transfer", "remote")]

    elif verb == "transform":
        non_flag = [t for t in rest if not t.startswith("-")]
        target = non_flag[-1] if non_flag else ""
        return [("transform", os.path.basename(target))] if target else []

    noun = os.path.basename(_first_non_flag(rest))
    return [(verb, noun)]


# ---------------------------------------------------------------------------
# Tool-level handlers
# ---------------------------------------------------------------------------

def _parse_tool_input(action_str: str, tool_name: str) -> dict:
    """Parse the JSON payload from a ToolName({...}) action string."""
    prefix = f"{tool_name}("
    if action_str.startswith(prefix) and action_str.endswith(")"):
        try:
            return json.loads(action_str[len(prefix):-1])
        except Exception:
            pass
    return {}


def _handle_file_tool(
    action_str: str,
    action_type: str,
    verb: str,
    *path_keys: str,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Generic handler for Read / Edit / Write and their aliases."""
    inp  = _parse_tool_input(action_str, action_type)
    path = next((inp[k] for k in path_keys if inp.get(k)), "")
    if not path and inp.get("_i"):
        noun = str(inp["_i"])[:50]
    else:
        noun = _file_noun(path)
    pairs = [(verb, noun)] if noun else []
    return verb, noun, pairs


def _handle_mcp_tool(
    action_type: str,
    action_str: str,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Handler for mcp__server__tool_name(...) style tools."""
    parts = action_type.split("__")
    verb  = parts[-1] if len(parts) >= 3 else "mcp"
    inp   = _parse_tool_input(action_str, action_type)
    noun  = next(
        (str(v)[:40] for k, v in inp.items() if k not in ("timeout", "session_id") and v),
        "",
    )
    pairs = [(verb, noun)] if noun else [(verb, "")]
    return verb, noun, pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_verb_noun_from_action(
    action_type: str,
    action_str: str,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Return (root_verb, root_noun, pairs) for a single Action."""
    if action_type.startswith("mcp__"):
        return _handle_mcp_tool(action_type, action_str)

    if action_type in ("Bash", "bash", "run_command"):
        inp     = _parse_tool_input(action_str, action_type)
        command = inp.get("command", "")
        if not command:
            return "", "", []
        pairs = _bash_extract(command)
        if not pairs:
            return "", "", []
        root_verb, root_noun = pairs[0]
        return root_verb, root_noun, pairs

    if action_type in ("Read", "read_file"):
        return _handle_file_tool(action_str, action_type, "read", "file_path", "path")

    if action_type in ("Edit", "edit", "edit_file", "NotebookEdit"):
        return _handle_file_tool(action_str, action_type, "edit", "file_path", "notebook_path", "path")

    if action_type in ("Write", "write", "write_file"):
        return _handle_file_tool(action_str, action_type, "write", "file_path", "path")

    if action_type in ("Glob", "find", "glob_path"):
        inp     = _parse_tool_input(action_str, action_type)
        noun    = _pattern_noun(inp.get("pattern", ""))
        pairs   = [("find", noun)] if noun else []
        return "find", noun, pairs

    if action_type in ("Grep", "grep", "grep_content", "codebase_search"):
        inp     = _parse_tool_input(action_str, action_type)
        pattern = inp.get("pattern", "")
        path    = inp.get("path", "")
        noun    = _pattern_noun(pattern) or os.path.basename(path.rstrip("/"))
        pairs   = [("search", noun)] if noun else []
        return "search", noun, pairs

    if action_type == "TodoWrite":
        return "plan", "task_list", [("plan", "task_list")]

    if action_type == "Agent":
        inp   = _parse_tool_input(action_str, action_type)
        desc  = inp.get("description", inp.get("prompt", ""))
        words = desc.split()[:4]
        noun  = " ".join(words) if words else "subtask"
        return "delegate", noun, [("delegate", noun)]

    if action_type in ("Skill", "invoke_skill"):
        inp  = _parse_tool_input(action_str, action_type)
        noun = inp.get("skill", inp.get("name", "")) or "skill"
        return "invoke", noun, [("invoke", noun)]

    if action_type in ("WebFetch", "web_fetch"):
        inp = _parse_tool_input(action_str, action_type)
        m   = re.search(r"https?://([^/]+)", inp.get("url", ""))
        noun = m.group(1) if m else inp.get("url", "")[:30]
        return "fetch", noun, [("fetch", noun)]

    if action_type in ("WebSearch", "web_search"):
        inp  = _parse_tool_input(action_str, action_type)
        noun = inp.get("query", "")[:40]
        return "search_web", noun, [("search_web", noun)]

    verb = TOOL_VERB_MAP.get(action_type) or action_type.lower().replace("__", "_")
    if not verb:
        return "", "", []

    inp  = _parse_tool_input(action_str, action_type)
    noun = next((str(v).strip()[:40] for v in inp.values() if str(v).strip()), "")
    pairs = [(verb, noun)] if noun else [(verb, "")]
    return verb, noun, pairs
