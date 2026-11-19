"""Streamlit dashboard for the GitHub repository analyzer.

Flow: read the repository (REST), compute the numbers, show the charts, then ask
the model for the written review. The two halves are separate so the dashboard
still shows something useful when the model is unavailable.
"""

from __future__ import annotations

import streamlit as st

from analyzer.github_client import (
    GitHubClient,
    GitHubError,
    NotFoundError,
    RateLimitError,
    parse_repo_input,
)
from analyzer.insights import InsightError, analyse
from analyzer.metrics import (
    commit_activity,
    file_type_counts,
    language_breakdown,
    summary_for_prompt,
    top_directories,
)

st.set_page_config(page_title="GitHub Repo Analyzer", page_icon="📊", layout="wide")

st.title("GitHub Repository Analyzer")
st.caption(
    "Reads a public repository through the GitHub REST API, charts what is in it, and "
    "writes a review: summary, strengths, recommendations and where a new contributor can start."
)

with st.sidebar:
    st.subheader("Repository")
    repo_input = st.text_input("owner/repo or URL", value="pallets/flask")
    token = st.text_input(
        "GitHub token (optional)",
        type="password",
        help="Only needed for private repositories, or to raise the API rate limit.",
    )
    commit_count = st.slider("Recent commits to read", 5, 50, 20, 5)
    run = st.button("Analyse", type="primary")
    st.caption(
        "Without a token GitHub allows 60 requests an hour per address; "
        "a token raises that to 5,000."
    )

if not run:
    st.info("Enter a repository and press Analyse.")
    st.stop()

# --- read from GitHub --------------------------------------------------------
try:
    owner, name = parse_repo_input(repo_input)
except GitHubError as exc:
    st.error(str(exc))
    st.stop()

client = GitHubClient(token=token.strip())
with st.spinner("Reading {}/{} from GitHub…".format(owner, name)):
    try:
        snapshot = client.fetch(owner, name, commit_limit=commit_count)
    except NotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except RateLimitError as exc:
        st.error(str(exc))
        if exc.reset_at:
            st.caption("The limit resets at Unix time {}.".format(exc.reset_at))
        st.stop()
    except GitHubError as exc:
        st.error(str(exc))
        st.stop()

# --- header ------------------------------------------------------------------
st.subheader("{}/{}".format(snapshot.owner, snapshot.name))
if snapshot.description:
    st.write(snapshot.description)
if snapshot.archived:
    st.warning("This repository is archived: it is read-only and no longer maintained.")

metrics = st.columns(5)
metrics[0].metric("Stars", "{:,}".format(snapshot.stars))
metrics[1].metric("Forks", "{:,}".format(snapshot.forks))
metrics[2].metric("Open issues", "{:,}".format(snapshot.open_issues))
metrics[3].metric("Files", "{:,}".format(len(snapshot.tree_paths)))
metrics[4].metric("Size", "{} KB".format(snapshot.size_kb))

st.caption(
    "License: {} · Default branch: {} · Last push: {} · Created: {}".format(
        snapshot.license_name or "none",
        snapshot.default_branch,
        snapshot.pushed_at[:10] or "unknown",
        snapshot.created_at[:10] or "unknown",
    )
)
if snapshot.topics:
    st.caption("Topics: " + ", ".join(snapshot.topics))

def bar_chart(title: str, rows: list[tuple[str, float]], unit: str = "") -> None:
    """Draw a horizontal bar chart with plain HTML.

    Streamlit's st.bar_chart pulls in Altair, which breaks on some Python
    versions (3.14 removed the TypedDict `closed` argument Altair 5 passes).
    A few lines of HTML keep the dashboard dependency-free and readable.
    `rows` is (label, value) pairs; the longest bar is the largest value.
    """
    st.markdown("**{}**".format(title))
    if not rows:
        st.caption("Nothing to chart here.")
        return
    top = max(value for _, value in rows) or 1
    bars = []
    for label, value in rows:
        width = max(2.0, 100.0 * value / top)
        shown = "{:,.0f}{}".format(value, unit)
        bars.append(
            '<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            '<div style="width:32%;font-size:12px;color:#555;text-align:right;'
            'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{}</div>'
            '<div style="flex:1;background:#eef1f5;border-radius:4px">'
            '<div style="width:{:.1f}%;background:#4c78a8;height:14px;border-radius:4px"></div>'
            "</div>"
            '<div style="width:15%;font-size:12px;color:#333">{}</div>'
            "</div>".format(label, width, shown)
        )
    st.markdown("".join(bars), unsafe_allow_html=True)


# --- charts ------------------------------------------------------------------
left, right = st.columns(2)

with left:
    bar_chart(
        "Languages by bytes",
        [(row["language"], row["percent"]) for row in language_breakdown(snapshot.languages)],
        unit="%",
    )
    bar_chart("Files by type", list(file_type_counts(snapshot.tree_paths).items()))

with right:
    bar_chart(
        "Busiest directories",
        [(row["directory"], row["files"]) for row in top_directories(snapshot.tree_paths)],
    )
    bar_chart(
        "Commits per day (recent window)",
        [(row["date"], row["commits"]) for row in commit_activity(snapshot.recent_commits)],
    )

