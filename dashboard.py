#!/usr/bin/env python3
"""Reasoning Theater Paper — Experiment Dashboard

Usage: python dashboard.py [--port 8765]
Opens a web dashboard showing experiment progress, metrics, and paper figures.
"""

import json
import os
import sys
import base64
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import statistics

PROJECT_DIR = Path(__file__).parent

# ── Data loading ──────────────────────────────────────────────

def load_results(jsonl_path: Path) -> list[dict]:
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)

def scan_experiments() -> list[dict]:
    """Find all experiment directories with results."""
    experiments = []
    for d in sorted(PROJECT_DIR.iterdir()):
        results_file = d / "results.jsonl"
        config_file = d / "config.json"
        if d.is_dir() and results_file.exists():
            config = load_config(config_file) if config_file.exists() else {}
            results = load_results(results_file)
            experiments.append({
                "name": d.name,
                "config": config,
                "results": results,
            })
    # Also check belief_runs/
    belief_dir = PROJECT_DIR / "belief_runs"
    if belief_dir.exists():
        for d in sorted(belief_dir.iterdir()):
            results_file = d / "results.jsonl"
            config_file = d / "config.json"
            if d.is_dir() and results_file.exists():
                config = load_config(config_file) if config_file.exists() else {}
                results = load_results(results_file)
                experiments.append({
                    "name": f"belief_runs/{d.name}",
                    "config": config,
                    "results": results,
                })
    return experiments

def compute_metrics(results: list[dict], config: dict) -> dict:
    n_target = config.get("n_problems", "?")
    n_done = len(results)

    correct = [r for r in results if r.get("selected_rollout_correct")]
    accuracy = len(correct) / n_done if n_done else 0

    # Theater & commitment fractions (only for correct problems with values)
    theater_fracs = [r["theater_fraction"] for r in results
                     if r.get("theater_fraction") is not None]
    commit_fracs = [r["commitment_fraction"] for r in results
                    if r.get("commitment_fraction") is not None]

    # CoT lengths
    cot_lens = [r["selected_rollout_len"] for r in results
                if r.get("selected_rollout_len")]

    # PSC agreement at different prefixes
    psc_by_frac = {}
    for r in results:
        for pr in r.get("prefix_results", []):
            f = pr["fraction"]
            if f not in psc_by_frac:
                psc_by_frac[f] = []
            psc_by_frac[f].append(pr.get("psc_agreement_rate", 0))

    # EFA accuracy at different prefixes
    efa_by_frac = {}
    for r in results:
        for pr in r.get("prefix_results", []):
            f = pr["fraction"]
            if f not in efa_by_frac:
                efa_by_frac[f] = []
            efa_by_frac[f].append(1 if pr.get("efa_correct") else 0)

    return {
        "n_target": n_target,
        "n_done": n_done,
        "progress_pct": round(n_done / n_target * 100, 1) if isinstance(n_target, int) and n_target > 0 else None,
        "accuracy": round(accuracy * 100, 1),
        "n_correct": len(correct),
        "theater_mean": round(statistics.mean(theater_fracs) * 100, 1) if theater_fracs else None,
        "theater_median": round(statistics.median(theater_fracs) * 100, 1) if theater_fracs else None,
        "commit_mean": round(statistics.mean(commit_fracs) * 100, 1) if commit_fracs else None,
        "commit_median": round(statistics.median(commit_fracs) * 100, 1) if commit_fracs else None,
        "n_theater": len(theater_fracs),
        "cot_mean": round(statistics.mean(cot_lens)) if cot_lens else None,
        "cot_median": round(statistics.median(cot_lens)) if cot_lens else None,
        "psc_curve": {str(k): round(statistics.mean(v), 3) for k, v in sorted(psc_by_frac.items())},
        "efa_curve": {str(k): round(statistics.mean(v) * 100, 1) for k, v in sorted(efa_by_frac.items())},
    }

def get_figures() -> list[dict]:
    """Return list of figure files with base64 data."""
    fig_dir = PROJECT_DIR / "figures"
    figs = []
    if fig_dir.exists():
        for f in sorted(fig_dir.iterdir()):
            if f.suffix in (".png", ".pdf"):
                figs.append({"name": f.name, "path": str(f), "ext": f.suffix})
    # Also the 6-panel overview
    for f in PROJECT_DIR.glob("reasoning_theater_6panel.*"):
        figs.append({"name": f.name, "path": str(f), "ext": f.suffix})
    return figs

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix
    mime = "image/png" if ext == ".png" else "application/pdf"
    return f"data:{mime};base64,{data}"

# ── HTML Generation ───────────────────────────────────────────

def build_html() -> str:
    experiments = scan_experiments()
    figures = get_figures()

    # Build experiment cards
    exp_cards = []
    all_metrics = []
    for exp in experiments:
        m = compute_metrics(exp["results"], exp["config"])
        all_metrics.append((exp["name"], exp["config"], m))

        model = exp["config"].get("model_name", "unknown")
        renderer = exp["config"].get("renderer_name", "unknown")
        progress_bar = ""
        if m["progress_pct"] is not None:
            color = "#4caf50" if m["progress_pct"] >= 100 else "#ff9800" if m["progress_pct"] >= 50 else "#f44336"
            progress_bar = f'''
            <div class="progress-bar">
                <div class="progress-fill" style="width:{min(m['progress_pct'],100)}%;background:{color}"></div>
                <span class="progress-text">{m['n_done']}/{m['n_target']} ({m['progress_pct']}%)</span>
            </div>'''

        theater_info = ""
        if m["theater_mean"] is not None:
            theater_info = f'''
            <div class="metric-row">
                <div class="metric"><span class="metric-val">{m['theater_mean']}%</span><span class="metric-lbl">Theater (mean)</span></div>
                <div class="metric"><span class="metric-val">{m['theater_median']}%</span><span class="metric-lbl">Theater (median)</span></div>
                <div class="metric"><span class="metric-val">{m['commit_mean']}%</span><span class="metric-lbl">Commit @ (mean)</span></div>
                <div class="metric"><span class="metric-val">{m['commit_median']}%</span><span class="metric-lbl">Commit @ (median)</span></div>
            </div>'''

        # PSC + EFA sparkline data
        psc_data = json.dumps(m["psc_curve"])
        efa_data = json.dumps(m["efa_curve"])

        exp_cards.append(f'''
        <div class="card experiment-card">
            <h3>{exp['name']}</h3>
            <div class="tag-row">
                <span class="tag">{model}</span>
                <span class="tag">{renderer}</span>
            </div>
            {progress_bar}
            <div class="metric-row">
                <div class="metric"><span class="metric-val">{m['accuracy']}%</span><span class="metric-lbl">Accuracy ({m['n_correct']}/{m['n_done']})</span></div>
                <div class="metric"><span class="metric-val">{m['cot_mean'] or '-'}</span><span class="metric-lbl">Avg CoT len</span></div>
                <div class="metric"><span class="metric-val">{m['n_theater']}</span><span class="metric-lbl">Has commitment</span></div>
            </div>
            {theater_info}
            <div class="chart-row">
                <div class="mini-chart">
                    <div class="chart-title">PSC Agreement by Prefix %</div>
                    <canvas id="psc_{exp['name'].replace('/','_')}" width="280" height="140" data-values='{psc_data}'></canvas>
                </div>
                <div class="mini-chart">
                    <div class="chart-title">EFA Accuracy by Prefix %</div>
                    <canvas id="efa_{exp['name'].replace('/','_')}" width="280" height="140" data-values='{efa_data}'></canvas>
                </div>
            </div>
        </div>''')

    # Figure gallery
    fig_html = []
    for fig in figures:
        if fig["ext"] == ".png":
            b64 = img_to_b64(fig["path"])
            fig_html.append(f'''
            <div class="fig-item">
                <img src="{b64}" alt="{fig['name']}" />
                <div class="fig-caption">{fig['name']}</div>
            </div>''')
        elif fig["ext"] == ".pdf":
            b64 = img_to_b64(fig["path"])
            fig_html.append(f'''
            <div class="fig-item">
                <object data="{b64}" type="application/pdf" width="400" height="320">
                    <p>{fig['name']} (PDF)</p>
                </object>
                <div class="fig-caption">{fig['name']}</div>
            </div>''')

    # Paper PDF embed
    paper_pdf = PROJECT_DIR / "paper.pdf"
    paper_embed = '<p class="muted">No paper.pdf found. Compile with: pdflatex paper.tex</p>'
    if paper_pdf.exists():
        b64 = img_to_b64(str(paper_pdf))
        paper_embed = f'<iframe src="{b64}" width="100%" height="800" style="border:1px solid #333;border-radius:8px;"></iframe>'

    # Summary table
    summary_rows = []
    for name, config, m in all_metrics:
        model = config.get("model_name", "?")
        renderer = config.get("renderer_name", "?")
        theater_str = f"{m['theater_mean']}%" if m['theater_mean'] is not None else "-"
        commit_str = f"{m['commit_mean']}%" if m['commit_mean'] is not None else "-"
        summary_rows.append(f'''
        <tr>
            <td>{name}</td>
            <td>{model}</td>
            <td>{renderer}</td>
            <td>{m['n_done']}/{m['n_target']}</td>
            <td><strong>{m['accuracy']}%</strong></td>
            <td>{commit_str}</td>
            <td>{theater_str}</td>
            <td>{m['cot_mean'] or '-'}</td>
        </tr>''')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reasoning Theater — Dashboard</title>
<style>
:root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --orange: #d29922; --red: #f85149;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.5; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
h2 {{ font-size: 1.3em; margin: 30px 0 15px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
h3 {{ font-size: 1.1em; margin-bottom: 8px; color: var(--accent); }}
.subtitle {{ color: var(--muted); margin-bottom: 20px; }}
.muted {{ color: var(--muted); }}

/* Nav tabs */
.tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 20px; }}
.tab {{ padding: 10px 20px; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; transition: all 0.2s; }}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* Cards */
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(620px, 1fr)); gap: 16px; }}

/* Tags */
.tag-row {{ display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }}
.tag {{ background: #1f2937; color: var(--muted); padding: 2px 10px; border-radius: 12px; font-size: 0.8em; }}

/* Progress */
.progress-bar {{ position: relative; height: 22px; background: #21262d; border-radius: 6px; overflow: hidden; margin-bottom: 12px; }}
.progress-fill {{ height: 100%; border-radius: 6px; transition: width 0.5s; }}
.progress-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 0.75em; font-weight: 600; }}

/* Metrics */
.metric-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; }}
.metric {{ text-align: center; min-width: 90px; }}
.metric-val {{ display: block; font-size: 1.4em; font-weight: 700; color: var(--accent); }}
.metric-lbl {{ font-size: 0.75em; color: var(--muted); }}

/* Charts */
.chart-row {{ display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }}
.mini-chart {{ background: #0d1117; border-radius: 8px; padding: 10px; }}
.chart-title {{ font-size: 0.75em; color: var(--muted); margin-bottom: 4px; text-align: center; }}

/* Summary table */
table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--muted); font-weight: 600; font-size: 0.8em; text-transform: uppercase; }}
td strong {{ color: var(--green); }}

/* Figures */
.fig-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
.fig-item {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }}
.fig-item img, .fig-item object {{ max-width: 100%; border-radius: 6px; }}
.fig-caption {{ font-size: 0.8em; color: var(--muted); margin-top: 6px; }}

/* TODO */
.todo-done {{ color: var(--green); }}
.todo-pending {{ color: var(--orange); }}
pre.todo {{ background: var(--card); padding: 16px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; font-size: 0.85em; line-height: 1.7; }}
</style>
</head>
<body>
<div class="container">
    <h1>Reasoning Theater — Experiment Dashboard</h1>
    <p class="subtitle">Paper: "Black-box Logprob Probing Reveals Early Behavioral Commitment in Chain-of-Thought Models"</p>

    <div class="tabs">
        <div class="tab active" onclick="switchTab('overview')">Overview</div>
        <div class="tab" onclick="switchTab('experiments')">Experiments</div>
        <div class="tab" onclick="switchTab('figures')">Figures</div>
        <div class="tab" onclick="switchTab('paper')">Paper PDF</div>
        <div class="tab" onclick="switchTab('todo')">TODO</div>
    </div>

    <!-- Overview -->
    <div id="overview" class="tab-content active">
        <h2>Summary Table</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th><th>Model</th><th>Renderer</th>
                        <th>Progress</th><th>Accuracy</th><th>Commit @</th>
                        <th>Theater %</th><th>Avg CoT</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(summary_rows)}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Experiments -->
    <div id="experiments" class="tab-content">
        <h2>Experiment Details</h2>
        <div class="grid">
            {''.join(exp_cards)}
        </div>
    </div>

    <!-- Figures -->
    <div id="figures" class="tab-content">
        <h2>Generated Figures</h2>
        <div class="fig-gallery">
            {''.join(fig_html)}
        </div>
    </div>

    <!-- Paper PDF -->
    <div id="paper" class="tab-content">
        <h2>Paper Preview</h2>
        {paper_embed}
    </div>

    <!-- TODO -->
    <div id="todo" class="tab-content">
        <h2>TODO Tracker</h2>
        <div class="card">
            <pre class="todo" id="todo-content"></pre>
        </div>
    </div>
</div>

<script>
// Tab switching
function switchTab(id) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');
}}

// Draw mini line charts
function drawChart(canvas, color) {{
    const data = JSON.parse(canvas.dataset.values);
    const ctx = canvas.getContext('2d');
    const keys = Object.keys(data).map(Number);
    const vals = Object.values(data).map(Number);
    if (keys.length === 0) return;

    const w = canvas.width, h = canvas.height;
    const pad = {{ t: 15, b: 25, l: 40, r: 10 }};
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

    const minV = Math.min(...vals) * 0.95;
    const maxV = Math.max(...vals) * 1.02;
    const range = maxV - minV || 1;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {{
        const y = pad.t + (ph / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
        ctx.fillText((maxV - (range / 4) * i).toFixed(1), pad.l - 4, y + 3);
    }}

    // X labels
    ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    keys.forEach((k, i) => {{
        const x = pad.l + (i / (keys.length - 1 || 1)) * pw;
        ctx.fillText((k * 100).toFixed(0) + '%', x, h - 5);
    }});

    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    keys.forEach((k, i) => {{
        const x = pad.l + (i / (keys.length - 1 || 1)) * pw;
        const y = pad.t + ph - ((vals[i] - minV) / range) * ph;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();

    // Dots
    ctx.fillStyle = color;
    keys.forEach((k, i) => {{
        const x = pad.l + (i / (keys.length - 1 || 1)) * pw;
        const y = pad.t + ph - ((vals[i] - minV) / range) * ph;
        ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    }});
}}

// Init charts
document.querySelectorAll('canvas[id^="psc_"]').forEach(c => drawChart(c, '#58a6ff'));
document.querySelectorAll('canvas[id^="efa_"]').forEach(c => drawChart(c, '#3fb950'));

// Load TODO
const todoContent = {json.dumps(open(PROJECT_DIR / "TODO.md").read())};
document.getElementById('todo-content').textContent = todoContent;
</script>
</body>
</html>'''


# ── Server ────────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            html = build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        elif parsed.path == "/refresh":
            # API endpoint to get fresh data
            html = build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            # Serve static files from project dir
            self.directory = str(PROJECT_DIR)
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress noisy logs


def main():
    parser = argparse.ArgumentParser(description="Reasoning Theater Dashboard")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"\n  Reasoning Theater Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.port}")
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
