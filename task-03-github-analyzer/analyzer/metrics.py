"""Numbers for the dashboard, computed from the snapshot.

Kept separate from the GitHub client so the charts can be tested without the
network, and so the LLM prompt can be given a compact summary instead of the
whole tree.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .github_client import RepoSnapshot

# Extensions grouped into the buckets a reviewer cares about.
EXTENSION_GROUPS = {
    "Python": {".py", ".pyi", ".ipynb"},
    "JavaScript/TypeScript": {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"},
    "Web": {".html", ".css", ".scss", ".vue", ".svelte"},
    "Data/Config": {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".parquet"},
    "Docs": {".md", ".rst", ".txt", ".adoc"},
    "Shell": {".sh", ".bash", ".zsh", ".ps1"},
    "Other languages": {
        ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs",
        ".rb", ".php", ".swift", ".scala", ".r", ".jl",
    },
}

SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "dist", "build", "__pycache__", ".next"}


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def _is_vendored(path: str) -> bool:
    return any(part in SKIP_DIRS for part in path.split("/")[:-1])


def file_type_counts(tree_paths: List[str]) -> Dict[str, int]:
    """Count files per language/type group, skipping vendored directories."""
    counts: Dict[str, int] = {}
    for path in tree_paths:
        if _is_vendored(path):
            continue
        ext = _extension(path)
        if not ext:
            continue
        for group, extensions in EXTENSION_GROUPS.items():
            if ext in extensions:
                counts[group] = counts.get(group, 0) + 1
                break
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def top_directories(tree_paths: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """The busiest top-level directories: where the code actually lives."""
    counter: Counter = Counter()
    for path in tree_paths:
        parts = path.split("/")
        if len(parts) > 1 and parts[0] not in SKIP_DIRS:
            counter[parts[0]] += 1
    return [{"directory": name, "files": count} for name, count in counter.most_common(limit)]


def language_breakdown(languages: Dict[str, int]) -> List[Dict[str, Any]]:
    """GitHub's byte counts as percentages, largest first."""
    total = sum(languages.values()) or 1
    return [
        {"language": name, "bytes": size, "percent": round(size * 100 / total, 1)}
        for name, size in sorted(languages.items(), key=lambda item: item[1], reverse=True)
    ]


def commit_activity(recent_commits: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Commits per day for the days present in the recent-commit window."""
    counter: Counter = Counter()
    for commit in recent_commits:
        date = (commit.get("date") or "")[:10]
        if date:
            counter[date] += 1
    return [{"date": date, "commits": count} for date, count in sorted(counter.items())]


def summary_for_prompt(snapshot: RepoSnapshot, max_paths: int = 120) -> str:
    """A compact, token-cheap description of the repository for the model."""
    types = file_type_counts(snapshot.tree_paths)
    dirs = top_directories(snapshot.tree_paths, limit=8)
    langs = language_breakdown(snapshot.languages)
    manifests = ", ".join(sorted(snapshot.manifests)) or "none found"

    language_text = ", ".join(
        "{} {}%".format(item["language"], item["percent"]) for item in langs[:6]
    ) or "unknown"
    type_text = ", ".join("{} {}".format(name, count) for name, count in types.items()) or "unknown"
    dir_text = ", ".join(
        "{} ({})".format(item["directory"], item["files"]) for item in dirs
    ) or "flat repo"
    commit_text = "; ".join(
        "{} {}".format(c["date"][:10], c["message"][:60]) for c in snapshot.recent_commits[:8]
    ) or "none"
    contributor_text = ", ".join(
        "{} ({})".format(c["login"], c["contributions"]) for c in snapshot.contributors[:5]
    ) or "unknown"

    lines = [
        "Repository: {}/{}".format(snapshot.owner, snapshot.name),
        "Description: {}".format(snapshot.description or "(none)"),
        "Stars: {}, Forks: {}, Open issues: {}".format(
            snapshot.stars, snapshot.forks, snapshot.open_issues
        ),
        "License: {}, Archived: {}".format(snapshot.license_name or "none", snapshot.archived),
        "Created: {}, Last push: {}".format(snapshot.created_at[:10], snapshot.pushed_at[:10]),
        "Size: {} KB, files in tree: {}".format(snapshot.size_kb, len(snapshot.tree_paths)),
        "Languages (bytes): {}".format(language_text),
        "File types: {}".format(type_text),
        "Top directories: {}".format(dir_text),
        "Manifests: {}".format(manifests),
        "Recent commits: {}".format(commit_text),
        "Top contributors: {}".format(contributor_text),
        "File sample:",
    ]
    lines.extend("- {}".format(path) for path in snapshot.tree_paths[:max_paths])
    if snapshot.readme:
        lines.append("README (truncated):")
        lines.append(snapshot.readme[:6000])
    for name, text in list(snapshot.manifests.items())[:4]:
        lines.append("{} (truncated):".format(name))
        lines.append(text[:1500])
    return "\n".join(lines)
