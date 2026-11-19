"""Streamlit dashboard for the GitHub repository analyzer.

Flow: read the repository (REST), compute the numbers, show the charts, then ask
the model for the written review. The two halves are separate so the dashboard
still shows something useful when the model is unavailable.
"""

from __future__ import annotations

import pandas as pd
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

# --- charts ------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("**Languages by bytes**")
    languages = language_breakdown(snapshot.languages)
    if languages:
        frame = pd.DataFrame(languages)
        st.bar_chart(frame.set_index("language")["percent"], height=260)
    else:
        st.caption("GitHub reported no language breakdown.")

    st.markdown("**Files by type**")
    types = file_type_counts(snapshot.tree_paths)
    if types:
        frame = pd.DataFrame(list(types.items()), columns=["type", "files"])
        st.bar_chart(frame.set_index("type")["files"], height=260)
    else:
        st.caption("No recognised source files in the tree.")

with right:
    st.markdown("**Busiest directories**")
    directories = top_directories(snapshot.tree_paths)
    if directories:
        frame = pd.DataFrame(directories)
        st.bar_chart(frame.set_index("directory")["files"], height=260)
    else:
        st.caption("The repository is flat: no subdirectories.")

    st.markdown("**Commits per day (recent window)**")
    activity = commit_activity(snapshot.recent_commits)
    if activity:
        frame = pd.DataFrame(activity)
        st.bar_chart(frame.set_index("date")["commits"], height=260)
    else:
        st.caption("No recent commits were returned.")

# --- recent commits and contributors ----------------------------------------
with st.expander("Recent commits"):
    if snapshot.recent_commits:
        st.dataframe(
            pd.DataFrame(snapshot.recent_commits)[["sha", "date", "author", "message"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("None returned.")

with st.expander("Top contributors"):
    if snapshot.contributors:
        st.dataframe(pd.DataFrame(snapshot.contributors), use_container_width=True, hide_index=True)
    else:
        st.caption("None returned.")

with st.expander("Dependency manifests found"):
    if snapshot.manifests:
        for manifest_name, text in snapshot.manifests.items():
            st.markdown("**{}**".format(manifest_name))
            st.code(text[:1500], language="text")
    else:
        st.caption("No known manifest at the repository root.")

# --- the written review ------------------------------------------------------
st.divider()
st.subheader("Review")
with st.spinner("Asking the model for the review…"):
    try:
        review = analyse(summary_for_prompt(snapshot))
    except InsightError as exc:
        st.error(str(exc))
        st.stop()

st.markdown("**Summary**")
st.write(review["summary"] or "The model returned no summary.")

st.markdown("**Strengths**")
for item in review["strengths"] or ["None listed."]:
    st.markdown("- {}".format(item))

st.markdown("**Recommendations**")
for item in review["recommendations"] or ["None listed."]:
    st.markdown("- {}".format(item))

st.markdown("**Where a new contributor could start**")
if review["contribution_opportunities"]:
    for item in review["contribution_opportunities"]:
        title = item.get("title", "Untitled")
        where = item.get("where", "")
        why = item.get("why", "")
        suffix = " · `{}`".format(where) if where else ""
        st.markdown("**{}**{}".format(title, suffix))
        if why:
            st.caption(why)
else:
    st.caption("None listed.")

if review["risks"]:
    st.markdown("**Risks and maintenance notes**")
    for item in review["risks"]:
        st.write("- " + item)
