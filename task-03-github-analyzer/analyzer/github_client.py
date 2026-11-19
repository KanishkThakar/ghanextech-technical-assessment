"""GitHub REST client for the analyzer.

Everything the dashboard shows comes from these calls: repository metadata, the
README, the file tree, the dependency manifests, recent commits and the
languages breakdown. The client turns GitHub's error codes into messages a user
can act on (not found, private, rate limited) instead of raw HTTP errors.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

API_ROOT = "https://api.github.com"
TIMEOUT_SECONDS = 20
USER_AGENT = "ghanex-repo-analyzer/1.0"

# Manifests worth reading: they tell the model what the project is built with.
MANIFEST_NAMES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Pipfile",
    "environment.yml",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
}

REPO_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/", re.IGNORECASE
)


class GitHubError(Exception):
    """A failure worth showing the user as-is."""


class NotFoundError(GitHubError):
    """The repository or file does not exist, or is private."""


class RateLimitError(Exception):
    def __init__(self, message: str, reset_at: Optional[str] = None):
        super().__init__(message)
        self.reset_at = reset_at


@dataclass
class RepoSnapshot:
    """Everything read from GitHub for one repository."""

    owner: str
    name: str
    html_url: str
    description: str = ""
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    watchers: int = 0
    default_branch: str = "main"
    license_name: str = ""
    topics: List[str] = field(default_factory=list)
    created_at: str = ""
    pushed_at: str = ""
    size_kb: int = 0
    archived: bool = False
    languages: Dict[str, int] = field(default_factory=dict)
    readme: str = ""
    tree_paths: List[str] = field(default_factory=list)
    manifests: Dict[str, str] = field(default_factory=dict)
    recent_commits: List[Dict[str, str]] = field(default_factory=list)
    contributors: List[Dict[str, Any]] = field(default_factory=list)
    truncated_tree: bool = False


def parse_repo_input(raw: str) -> tuple[str, str]:
    """Accept 'owner/repo' or a github.com URL and return (owner, repo)."""
    text = (raw or "").strip()
    if not text:
        raise GitHubError("Enter a repository as owner/repo or a GitHub URL.")
    text = REPO_URL_RE.sub("", text)
    text = text.split("?")[0].split("#")[0].strip("/")
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        raise GitHubError("That does not look like a repository. Use owner/repo, e.g. pandas-dev/pandas.")
    owner, name = parts[0], parts[1]
    if name.endswith(".git"):
        name = name[:-4]
    return owner, name


class GitHubClient:
    """Small wrapper over the REST endpoints the analyzer needs."""

    def __init__(self, token: str = ""):
        self.session = requests.Session()
        headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)
        self.authenticated = bool(token)

    # -- plumbing ------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = path if path.startswith("http") else f"{API_ROOT}{path}"
        try:
            response = self.session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise GitHubError(f"Could not reach GitHub: {exc}") from exc

        if response.status_code == 404:
            raise NotFoundError(
                "Repository not found. Check the spelling, or it may be private "
                "(add a token to read private repositories)."
            )
        if response.status_code == 401:
            raise GitHubError("GitHub rejected the token. Check it has not expired.")
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset = response.headers.get("X-RateLimit-Reset")
            raise RateLimitError(
                "GitHub's rate limit is used up for this connection. "
                "Add a personal access token to get a much higher limit.",
                reset_at=reset,
            )
        if response.status_code >= 400:
            raise GitHubError(f"GitHub returned {response.status_code} for {url}.")
        return response.json()

    def _get_text(self, path: str) -> str:
        """Read a file's raw text through the contents API."""
        data = self._get(path)
        if isinstance(data, dict) and data.get("content"):
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except (ValueError, TypeError):
                return ""
        return ""

    # -- reads ---------------------------------------------------------------
    def fetch(self, owner: str, name: str, commit_limit: int = 20) -> RepoSnapshot:
        """Read everything the dashboard needs, tolerating partial failures.

        A missing README or an empty tree should not sink the whole analysis, so
        each optional call is guarded; only the repository lookup itself must
        succeed.
        """
        meta = self._get(f"/repos/{owner}/{name}")
        snapshot = RepoSnapshot(
            owner=meta.get("owner", {}).get("login", owner),
            name=meta.get("name", name),
            html_url=meta.get("html_url", f"https://github.com/{owner}/{name}"),
            description=meta.get("description") or "",
            stars=meta.get("stargazers_count", 0),
            forks=meta.get("forks_count", 0),
            open_issues=meta.get("open_issues_count", 0),
            watchers=meta.get("subscribers_count", 0),
            default_branch=meta.get("default_branch", "main"),
            license_name=(meta.get("license") or {}).get("spdx_id", "") or "",
            topics=meta.get("topics") or [],
            created_at=meta.get("created_at", ""),
            pushed_at=meta.get("pushed_at", ""),
            size_kb=meta.get("size", 0),
            archived=bool(meta.get("archived")),
        )

        try:
            snapshot.languages = self._get(f"/repos/{owner}/{name}/languages") or {}
        except GitHubError:
            snapshot.languages = {}

        try:
            snapshot.readme = self._get_text(f"/repos/{owner}/{name}/readme")
        except GitHubError:
            snapshot.readme = ""

        try:
            tree = self._get(
                f"/repos/{owner}/{name}/git/trees/{snapshot.default_branch}",
                params={"recursive": "1"},
            )
            entries = tree.get("tree") or []
            snapshot.truncated_tree = bool(tree.get("truncated"))
            snapshot.tree_paths = [entry["path"] for entry in entries if entry.get("type") == "blob"]
        except GitHubError:
            snapshot.tree_paths = []

        snapshot.manifests = self._read_manifests(owner, name, snapshot.tree_paths)

        try:
            commits = self._get(
                f"/repos/{owner}/{name}/commits", params={"per_page": commit_limit}
            )
            snapshot.recent_commits = [
                {
                    "sha": (item.get("sha") or "")[:7],
                    "date": ((item.get("commit") or {}).get("author") or {}).get("date", ""),
                    "author": ((item.get("commit") or {}).get("author") or {}).get("name", ""),
                    "message": ((item.get("commit") or {}).get("message") or "").split("\n")[0][:120],
                }
                for item in commits
            ]
        except GitHubError:
            snapshot.recent_commits = []

        try:
            snapshot.contributors = [
                {"login": item.get("login", ""), "contributions": item.get("contributions", 0)}
                for item in self._get(
                    f"/repos/{owner}/{name}/contributors", params={"per_page": 10}
                )
            ]
        except GitHubError:
            snapshot.contributors = []

        return snapshot

    def _read_manifests(self, owner: str, name: str, paths: List[str]) -> Dict[str, str]:
        """Read the dependency manifests at the repository root, capped in size."""
        found: Dict[str, str] = {}
        for path in paths:
            if "/" in path:
                continue
            if path not in MANIFEST_NAMES:
                continue
            try:
                text = self._get_text(f"/repos/{owner}/{name}/contents/{path}")
            except GitHubError:
                continue
            if text:
                found[path] = text[:4000]
        return found
