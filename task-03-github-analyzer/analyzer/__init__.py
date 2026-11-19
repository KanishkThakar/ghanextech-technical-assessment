"""GitHub repository analyzer: fetch, measure, then explain."""

from .github_client import GitHubClient, GitHubError, NotFoundError, RateLimitError, RepoSnapshot
from .insights import InsightError, analyse
from .metrics import (
    commit_activity,
    file_type_counts,
    language_breakdown,
    summary_for_prompt,
    top_directories,
)

__all__ = [
    "GitHubClient",
    "GitHubError",
    "NotFoundError",
    "RateLimitError",
    "RepoSnapshot",
    "InsightError",
    "analyse",
    "commit_activity",
    "file_type_counts",
    "language_breakdown",
    "summary_for_prompt",
    "top_directories",
]
