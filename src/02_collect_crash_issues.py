"""Step 2: collect crash-related issue counts from the GitHub Search API."""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
PROGRESS = DATA / "github_progress.json"
OUT = DATA / "pkgbloat_dataset.csv"
SLEEP = 2.2

TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

QUERIES = {
    "issues_total": "repo:{r} is:issue",
    "crash_issues": "repo:{r} is:issue traceback OR crash OR segfault OR exception",
    "dep_error_issues": 'repo:{r} is:issue ImportError OR ModuleNotFoundError OR "version conflict" OR incompatible',
}


def search(endpoint, q):
    """Return the parsed JSON of a search call, with backoff on 403/429."""
    wait = 15
    for _ in range(6):
        time.sleep(SLEEP)
        r = requests.get(f"https://api.github.com/search/{endpoint}", params={"q": q, "per_page": 1},
                         headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset")
            retry = r.headers.get("Retry-After")
            delay = int(retry) if retry else wait
            if r.headers.get("X-RateLimit-Remaining") == "0" and reset:
                delay = max(delay, int(reset) - int(time.time()) + 1)
            print(f"  rate limited, sleeping {delay}s")
            time.sleep(min(delay, 90))
            wait *= 2
            continue
        print(f"  HTTP {r.status_code} for {q}: {r.text[:100]}")
        return None
    return None


def collect(repo):
    row = {}
    for key, tmpl in QUERIES.items():
        j = search("issues", tmpl.format(r=repo))
        row[key] = j["total_count"] if j else None
    j = search("repositories", f"repo:{repo}")
    row["stars"] = j["items"][0]["stargazers_count"] if j and j.get("items") else None
    return row


def write_dataset(pkgs, progress):
    rows = []
    for p, info in pkgs.items():
        g = progress.get(p)
        if not g or g.get("issues_total") is None:
            continue
        tot = g["issues_total"]
        rows.append({
            "package": p, "version": info["version"], "github_repo": info["github_repo"],
            "direct_deps": info["n_direct"], "transitive_deps": info["n_transitive"],
            "issues_total": tot, "crash_issues": g["crash_issues"], "dep_error_issues": g["dep_error_issues"],
            "crash_rate": g["crash_issues"] / tot if tot else None,
            "dep_error_rate": g["dep_error_issues"] / tot if tot else None,
            "stars": g["stars"],
        })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only process the first N packages")
    args = ap.parse_args()
    if not TOKEN:
        print("WARNING: GITHUB_TOKEN not set; unauthenticated search is limited to 10 requests/min")
    pkgs = json.loads((DATA / "packages_deps.json").read_text())
    names = [p for p, v in pkgs.items() if v["github_repo"]]
    if args.limit:
        names = names[:args.limit]
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    for i, p in enumerate(names, 1):
        if p in progress and progress[p].get("issues_total") is not None and progress[p].get("stars") is not None:
            continue
        repo = pkgs[p]["github_repo"]
        print(f"[{i}/{len(names)}] {p} ({repo})")
        progress[p] = collect(repo)
        PROGRESS.write_text(json.dumps(progress, indent=1))
        write_dataset({n: pkgs[n] for n in names}, progress)
    n = write_dataset({n: pkgs[n] for n in names}, progress)
    print(f"wrote {OUT} with {n} rows")


if __name__ == "__main__":
    main()
