"""Step 3: statistics + static charts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, CHARTS = ROOT / "data", ROOT / "charts"
CHARTS.mkdir(exist_ok=True)
ACCENT, GREY = "#2563eb", "#9ca3af"
BUCKETS = [(0, 0, "0"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 25, "11-25"), (26, 10**9, "26+")]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 11, "figure.dpi": 130})


def bucket(n):
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label


def spearman(x, y):
    rho, p = stats.spearmanr(x, y)
    return {"rho": round(float(rho), 3), "p": float(p)}


def partial_spearman(x, y, controls):
    """Rank-residual method: rank everything, regress out controls, correlate residuals."""
    rk = lambda v: stats.rankdata(v)
    C = np.column_stack([np.ones(len(x))] + [rk(c) for c in controls])
    res = lambda v: rk(v) - C @ np.linalg.lstsq(C, rk(v), rcond=None)[0]
    r, _ = stats.pearsonr(res(x), res(y))
    n, k = len(x), len(controls)
    t = r * np.sqrt((n - 2 - k) / (1 - r**2))
    p = 2 * stats.t.sf(abs(t), n - 2 - k)
    return {"rho": round(float(r), 3), "p": float(p)}


def desc(s):
    return {"median": round(float(s.median()), 4), "mean": round(float(s.mean()), 4), "max": round(float(s.max()), 4)}


def main():
    raw = pd.read_csv(DATA / "pkgbloat_dataset.csv")
    df = raw[raw.issues_total >= 50].dropna(subset=["stars"]).copy()
    df["crash_rate"] = df.crash_issues / df.issues_total
    df["dep_error_rate"] = df.dep_error_issues / df.issues_total
    print(f"{len(df)} packages kept of {len(raw)} (issues_total >= 50)")

    res = {"n_packages": len(df),
           "descriptive": {c: desc(df[c]) for c in ["direct_deps", "transitive_deps", "crash_rate", "dep_error_rate"]},
           "spearman": {}, "partial_spearman": {}, "buckets": {}}
    for dep in ["direct_deps", "transitive_deps"]:
        for tgt in ["crash_issues", "crash_rate", "dep_error_rate"]:
            res["spearman"][f"{dep} vs {tgt}"] = spearman(df[dep], df[tgt])
            res["partial_spearman"][f"{dep} vs {tgt} | stars, issues_total"] = partial_spearman(
                df[dep], df[tgt], [df.stars, df.issues_total])
    res["size_confound"] = {"issues_total vs crash_issues": spearman(df.issues_total, df.crash_issues),
                            "stars vs crash_issues": spearman(df.stars, df.crash_issues)}
    df["bucket"] = df.transitive_deps.map(bucket)
    for _, _, label in BUCKETS:
        s = df[df.bucket == label]
        res["buckets"][label] = {"n": len(s),
                                 "median_crash_rate": round(float(s.crash_rate.median()), 4) if len(s) else None,
                                 "median_dep_error_rate": round(float(s.dep_error_rate.median()), 4) if len(s) else None}
    (CHARTS / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # scatter
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x, y = df.transitive_deps + 1, df.crash_rate * 100
    ax.scatter(x, y, color=ACCENT, alpha=0.75, s=40)
    slope, icpt, *_ = stats.linregress(np.log10(x), y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, icpt + slope * np.log10(xs), color="#111827", lw=1.5, ls="--", label="log-linear trend")
    for _, r in df.nlargest(5, "transitive_deps").iterrows():
        ax.annotate(r.package, (r.transitive_deps + 1, r.crash_rate * 100), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Transitive dependencies (+1, log scale)")
    ax.set_ylabel("Crash-related issues (% of all issues)")
    ax.set_title("Dependency count vs crash-issue rate")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(CHARTS / "scatter_crash.png"); plt.close(fig)

    # buckets
    labels = [b[2] for b in BUCKETS]
    vals = [(res["buckets"][l]["median_dep_error_rate"] or 0) * 100 for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=ACCENT)
    for b, l in zip(bars, labels):
        ax.annotate(f"n={res['buckets'][l]['n']}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", xytext=(0, 3), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Transitive dependencies"); ax.set_ylabel("Median dependency-error issues (%)")
    ax.set_title("Dependency-error rate by dependency bucket"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(CHARTS / "buckets.png"); plt.close(fig)

    # histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df.transitive_deps, bins=15, color=ACCENT, edgecolor="white")
    ax.set_xlabel("Transitive dependencies"); ax.set_ylabel("Packages")
    ax.set_title("Distribution of transitive dependency counts")
    fig.tight_layout(); fig.savefig(CHARTS / "hist_transitive.png"); plt.close(fig)

    # network
    edges = pd.read_csv(DATA / "dependency_edges.csv")
    G = nx.from_pandas_edgelist(edges, "package", "depends_on", create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42, k=0.35, iterations=80)
    indeg = dict(G.in_degree())
    analysed = set(df.package)
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.5, arrows=False)
    nodes = list(G.nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=[20 + 25 * indeg[n] for n in nodes],
                           node_color=[ACCENT if n in analysed else GREY for n in nodes], alpha=0.85)
    top = sorted(indeg, key=indeg.get, reverse=True)[:5]
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top}, ax=ax, font_size=10, font_weight="bold")
    ax.set_title("PyPI dependency network (node size = in-degree; blue = analysed package)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(CHARTS / "network.png"); plt.close(fig)
    print("charts saved")


if __name__ == "__main__":
    main()
