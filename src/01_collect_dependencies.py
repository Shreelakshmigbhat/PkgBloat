"""Step 1: collect direct + transitive dependencies from PyPI."""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

PACKAGES = """requests flask django fastapi numpy pandas scipy matplotlib seaborn scikit-learn pytest
black pylint flake8 mypy sphinx notebook ipython tornado aiohttp httpx urllib3 boto3 sqlalchemy
celery redis pillow pydantic click rich typer jinja2 werkzeug gunicorn uvicorn scrapy lxml pyyaml
paramiko fabric ansible-core poetry pipenv tox nltk spacy gensim transformers plotly dash bokeh
streamlit gradio networkx sympy statsmodels xgboost lightgbm keras tensorflow torch opencv-python
pymongo psycopg2 alembic marshmallow cryptography pyjwt arrow pendulum loguru tqdm docker
kubernetes""".split()

GITHUB_FIXES = {"pydantic": "pydantic/pydantic", "uvicorn": "encode/uvicorn"}
GH_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.I)
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

session = requests.Session()
_cache = {}


def norm(name):
    return re.sub(r"[-_.]+", "-", name.lower())


def skip_marker(marker):
    m = marker.lower().replace(" ", "")
    if "extra==" in m:
        return True
    if "sys_platform==\"win32\"" in m or "sys_platform=='win32'" in m or "platform_system==\"windows\"" in m \
            or "platform_system=='windows'" in m or "os_name==\"nt\"" in m or "os_name=='nt'" in m:
        return True
    if "implementation_name==\"pypy\"" in m or "implementation_name=='pypy'" in m \
            or "platform_python_implementation==\"pypy\"" in m or "platform_python_implementation=='pypy'" in m:
        return True
    if re.search(r"python_version(<|<=)[\"']?3\.(\d+)", m):
        mm = re.search(r"python_version(<|<=)[\"']?3\.(\d+)", m)
        minor = int(mm.group(2))
        if (mm.group(1) == "<" and minor <= 9) or (mm.group(1) == "<=" and minor < 9):
            return True
    if re.search(r"python_full_version<[\"']?3\.(\d+)", m):
        if int(re.search(r"python_full_version<[\"']?3\.(\d+)", m).group(1)) <= 9:
            return True
    return False


def fetch(name):
    name = norm(name)
    if name in _cache:
        return _cache[name]
    info = None
    for attempt in range(3):
        try:
            r = session.get(f"https://pypi.org/pypi/{name}/json", timeout=30)
            if r.status_code == 200:
                info = r.json()
            break
        except requests.RequestException:
            continue
    if info is None:
        res = {"direct": [], "version": None, "github": None, "found": False}
    else:
        i = info["info"]
        direct = set()
        for req in i.get("requires_dist") or []:
            spec, _, marker = req.partition(";")
            if marker and skip_marker(marker):
                continue
            m = NAME_RE.match(spec)
            if m:
                direct.add(norm(m.group(1)))
        direct.discard(name)
        gh = None
        urls = list((i.get("project_urls") or {}).values()) + [i.get("home_page") or ""]
        for u in urls:
            m = GH_RE.search(u or "")
            if m:
                repo = re.sub(r"\.git$", "", m.group(2))
                gh = m.group(1) + "/" + repo
                break
        res = {"direct": sorted(direct), "version": i.get("version"), "github": gh, "found": True}
    _cache[name] = res
    return res


def main():
    targets = [norm(p) for p in PACKAGES]
    edges = set()
    seen = set()
    frontier = list(targets)
    with ThreadPoolExecutor(max_workers=8) as ex:
        while frontier:
            batch = [p for p in dict.fromkeys(frontier) if p not in seen]
            seen.update(batch)
            frontier = []
            for p, res in zip(batch, ex.map(fetch, batch)):
                for d in res["direct"]:
                    edges.add((p, d))
                    frontier.append(d)
            print(f"resolved {len(seen)} packages, {len(edges)} edges", file=sys.stderr)

    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)

    def closure(p):
        out, stack = set(), list(graph.get(p, ()))
        while stack:
            x = stack.pop()
            if x not in out and x != p:
                out.add(x)
                stack.extend(graph.get(x, ()))
        return out

    result = {}
    for p in targets:
        res = fetch(p)
        gh = GITHUB_FIXES.get(p, res["github"])
        trans = sorted(closure(p))
        result[p] = {"version": res["version"], "github_repo": gh, "found": res["found"],
                     "direct_deps": res["direct"], "transitive_deps": trans,
                     "n_direct": len(res["direct"]), "n_transitive": len(trans)}
    (DATA / "packages_deps.json").write_text(json.dumps(result, indent=1))
    pd.DataFrame(sorted(edges), columns=["package", "depends_on"]).to_csv(DATA / "dependency_edges.csv", index=False)
    print(f"saved {len(result)} packages, {len(edges)} edges")


if __name__ == "__main__":
    main()
