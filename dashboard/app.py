"""Step 4: PkgBloat interactive dashboard.  Run: python dashboard/app.py"""
from pathlib import Path

import dash
import dash_cytoscape as cyto
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, ctx, dash_table, dcc, html
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data"
ACCENT = "#2563eb"
BUCKETS = [(0, 0, "0"), (1, 3, "1-3"), (4, 10, "4-10"), (11, 25, "11-25"), (26, 10**9, "26+")]

df = pd.read_csv(DATA / "pkgbloat_dataset.csv")
df = df[df.issues_total >= 50].copy()
df["crash_pct"] = df.crash_rate * 100
df["dep_error_pct"] = df.dep_error_rate * 100
df["x"] = df.transitive_deps + 1
df["bucket"] = df.transitive_deps.map(lambda n: next(l for lo, hi, l in BUCKETS if lo <= n <= hi))
edges = pd.read_csv(DATA / "dependency_edges.csv")
analysed = set(df.package)
indeg = edges.depends_on.value_counts().to_dict()
nodes = set(edges.package) | set(edges.depends_on)
deps_of = edges.groupby("package").depends_on.apply(sorted).to_dict()
rho, pval = stats.spearmanr(df.transitive_deps, df.crash_rate)
MAXX = int(df.transitive_deps.max())

STYLE = [
    {"selector": "node", "style": {"background-color": "#9ca3af", "width": "data(size)", "height": "data(size)", "opacity": 0.8}},
    {"selector": "node.analysed", "style": {"background-color": ACCENT, "opacity": 1}},
    {"selector": "node.hub", "style": {"label": "data(id)", "font-size": 11}},
    {"selector": "node.dim", "style": {"opacity": 0.12}},
    {"selector": "node.hl", "style": {"background-color": "#f59e0b", "opacity": 1, "label": "data(id)", "font-size": 12, "z-index": 10}},
    {"selector": "edge", "style": {"width": 0.6, "line-color": "#d1d5db", "opacity": 0.5, "curve-style": "haystack"}},
    {"selector": "edge.dim", "style": {"opacity": 0.05}},
]
hubs = set(sorted(indeg, key=indeg.get, reverse=True)[:8])
elements = [{"data": {"id": n, "size": 8 + 4 * indeg.get(n, 0) ** 0.9},
             "classes": ("analysed " if n in analysed else "") + ("hub" if n in hubs else "")} for n in nodes]
elements += [{"data": {"source": a, "target": b}} for a, b in zip(edges.package, edges.depends_on)]


def kpi(label, value, sub=""):
    return html.Div([html.Div(label, style={"fontSize": 12, "color": "#6b7280"}),
                     html.Div(value, style={"fontSize": 26, "fontWeight": 600}),
                     html.Div(sub, style={"fontSize": 12, "color": "#6b7280"})],
                    style={"flex": 1, "padding": 14, "border": "1px solid #e5e7eb", "borderRadius": 8, "background": "#fff"})


CARD = {"flex": 1, "background": "#fff", "border": "1px solid #e5e7eb", "borderRadius": 8, "padding": 8}
app = dash.Dash(__name__)
app.title = "PkgBloat"
app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": 20, "background": "#f9fafb"}, children=[
    html.H2("PkgBloat: Does Dependency Count Predict Software Crash Risk?"),
    html.Div(style={"display": "flex", "gap": 12}, children=[
        kpi("Packages analysed", f"{len(df)}"), kpi("Dependency nodes", f"{len(nodes)}"),
        kpi("Dependency edges", f"{len(edges)}"),
        kpi("Spearman ρ (transitive vs crash rate)", f"{rho:.2f}", f"p = {pval:.3g}"),
    ]),
    html.Div(style={"display": "flex", "gap": 24, "margin": "16px 0", "alignItems": "flex-end"}, children=[
        html.Div([html.Label("Metric"), dcc.Dropdown(
            id="metric", clearable=False, value="crash_pct", style={"width": 260},
            options=[{"label": "Crash rate (%)", "value": "crash_pct"},
                     {"label": "Dependency-error rate (%)", "value": "dep_error_pct"}])]),
        html.Div([html.Label("Transitive dependency count"),
                  dcc.RangeSlider(id="range", min=0, max=MAXX, value=[0, MAXX], step=1, allowCross=False,
                                  marks={i: str(i) for i in range(0, MAXX + 1, max(1, MAXX // 8))},
                                  tooltip={"placement": "bottom"})], style={"width": 420}),
        html.Div([html.Label("Search package"),
                  dcc.Input(id="search", type="text", placeholder="e.g. flask", style={"display": "block"})]),
    ]),
    html.Div(style={"display": "flex", "gap": 12}, children=[
        html.Div(style=CARD, children=[
            html.B("Dependency network (blue = analysed packages, size = in-degree)"),
            cyto.Cytoscape(id="net", elements=elements, stylesheet=STYLE,
                           layout={"name": "cose", "animate": False, "idealEdgeLength": 60, "nodeRepulsion": 8000},
                           style={"height": 520, "width": "100%"}, minZoom=0.2, maxZoom=3),
            html.Div(id="detail", style={"padding": 8, "fontSize": 14}, children="Click a node for details."),
        ]),
        html.Div(style=CARD, children=[dcc.Graph(id="scatter", style={"height": 620})]),
    ]),
    html.Div(style={"display": "flex", "gap": 12, "marginTop": 12}, children=[
        html.Div(dcc.Graph(id="buckets"), style=CARD),
        html.Div(style=CARD, children=[
            dash_table.DataTable(
                id="table", sort_action="native", page_size=10, style_table={"overflowX": "auto"},
                columns=[{"name": n, "id": c} for n, c in [
                    ("Package", "package"), ("Direct", "direct_deps"), ("Transitive", "transitive_deps"),
                    ("Issues", "issues_total"), ("Crash %", "crash_pct"), ("Dep-error %", "dep_error_pct"),
                    ("Stars", "stars")]]),
        ]),
    ]),
])


@app.callback(Output("scatter", "figure"), Output("buckets", "figure"), Output("table", "data"),
              Input("metric", "value"), Input("range", "value"), Input("search", "value"), Input("net", "tapNodeData"))
def update(metric, rng, search, tapped):
    d = df[(df.transitive_deps >= rng[0]) & (df.transitive_deps <= rng[1])]
    if search:
        d = d[d.package.str.contains(search.strip().lower(), regex=False)]
    label = "Crash rate (%)" if metric == "crash_pct" else "Dependency-error rate (%)"
    fig = px.scatter(d, x="x", y=metric, log_x=True, custom_data=["package"], hover_name="package",
                     hover_data={"direct_deps": True, "transitive_deps": True, "issues_total": True,
                                 metric: ":.2f", "x": False},
                     labels={"x": "Transitive deps + 1 (log)", metric: label}, color_discrete_sequence=[ACCENT])
    fig.update_traces(marker={"size": 9, "opacity": 0.75})
    sel = tapped["id"] if tapped else None
    if sel in set(d.package):
        r = d[d.package == sel].iloc[0]
        fig.add_scatter(x=[r.x], y=[r[metric]], mode="markers+text", text=[sel], textposition="top center",
                        marker={"size": 16, "color": "#f59e0b", "line": {"width": 2, "color": "#111"}},
                        showlegend=False, hoverinfo="skip")
    fig.update_layout(dragmode="select", template="plotly_white", title="Transitive dependencies vs " + label,
                      margin={"t": 50})
    b = d.groupby("bucket")[metric].median().reindex([l for _, _, l in BUCKETS]).reset_index()
    bf = px.bar(b, x="bucket", y=metric, labels={"bucket": "Transitive deps", metric: "Median " + label},
                color_discrete_sequence=[ACCENT])
    bf.update_layout(template="plotly_white", title="Median " + label + " by dependency bucket")
    return fig, bf, d.sort_values("crash_pct", ascending=False).round(2).to_dict("records")


@app.callback(Output("net", "elements"), Input("scatter", "selectedData"), Input("net", "tapNodeData"),
              Input("search", "value"), State("net", "elements"))
def highlight(selected, tapped, search, els):
    hl = set()
    if ctx.triggered_id == "scatter" and selected and selected.get("points"):
        hl = {p["customdata"][0] for p in selected["points"]}
    elif ctx.triggered_id == "net" and tapped:
        hl = {tapped["id"]}
    elif ctx.triggered_id == "search" and search:
        hl = {n for n in analysed if search.strip().lower() in n}
    out = []
    for e in els:
        c = set((e.get("classes") or "").split()) - {"hl", "dim"}
        if "source" in e["data"]:
            if hl and not (e["data"]["source"] in hl or e["data"]["target"] in hl):
                c.add("dim")
        elif hl:
            c.add("hl" if e["data"]["id"] in hl else "dim")
        out.append({**e, "classes": " ".join(sorted(c))})
    return out


@app.callback(Output("detail", "children"), Input("net", "tapNodeData"))
def detail(node):
    if not node:
        return "Click a node for details."
    n = node["id"]
    ds = deps_of.get(n, [])
    if n not in analysed:
        return [html.B(n), html.Div(f"Not an analysed package. Depended on by {indeg.get(n, 0)} packages."),
                html.Div("Depends on: " + (", ".join(ds) or "nothing"))]
    r = df[df.package == n].iloc[0]
    return [html.B(n), html.Span(f"  v{r.version}"),
            html.Div(f"Direct deps: {r.direct_deps} | Transitive: {r.transitive_deps} | Stars: {int(r.stars):,}"),
            html.Div(f"Issues: {int(r.issues_total):,} | Crash rate: {r.crash_pct:.1f}% | "
                     f"Dep-error rate: {r.dep_error_pct:.1f}%"),
            html.Div("Direct dependencies: " + (", ".join(ds) or "none")),
            html.A("GitHub: " + r.github_repo, href="https://github.com/" + r.github_repo, target="_blank")]


if __name__ == "__main__":
    app.run(debug=False, port=8050)
