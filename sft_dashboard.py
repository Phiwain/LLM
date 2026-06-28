#!/usr/bin/env python3
"""Mini dashboard pour le SFT — lit sft.log et affiche la progression."""
import re
import json
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
SFT_LOG = PROJECT_ROOT / "sft.log"

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFT Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }
  .header { text-align: center; margin-bottom: 20px; padding: 20px; background: #161b22; border-radius: 12px; border: 1px solid #30363d; }
  .header h1 { font-size: 1.8em; color: #58a6ff; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }
  .card h3 { color: #58a6ff; margin-bottom: 12px; }
  .stat-value { font-size: 2.2em; font-weight: bold; color: #e6edf3; }
  .stat-label { color: #8b949e; font-size: 0.9em; margin-top: 4px; }
  .progress-bar { width: 100%; height: 8px; background: #21262d; border-radius: 4px; margin-top: 8px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #1f6feb, #58a6ff); border-radius: 4px; transition: width 0.5s; }
  .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .chart-container h3 { color: #58a6ff; margin-bottom: 12px; }
  #logBox { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.85em; max-height: 300px; overflow-y: auto; white-space: pre-wrap; color: #8b949e; }
  .footer { text-align: center; color: #8b949e; margin-top: 20px; font-size: 0.85em; }
</style>
</head>
<body>
<div class="header">
  <h1>🎯 PhiwAIn 1B — SFT</h1>
  <div class="stat-label">Supervised Fine-Tuning sur Alpaca EN (52k) + FR (55k) = 107k exemples</div>
</div>

<div class="grid">
  <div class="card">
    <h3>📊 Step</h3>
    <div class="stat-value" id="curStep">—</div>
    <div class="stat-label">/ <span id="totalSteps">5000</span> steps</div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    <div class="stat-label" id="estTime" style="margin-top:8px"></div>
  </div>
  <div class="card">
    <h3>📉 Train Loss</h3>
    <div class="stat-value" id="curLoss">—</div>
  </div>
  <div class="card">
    <h3>⏱️ Running</h3>
    <div class="stat-value" id="elapsed">—</div>
    <div class="stat-label" id="statusLabel">Status: Charging...</div>
  </div>
</div>

<div class="chart-container">
  <h3>Training Loss</h3>
  <canvas id="lossChart" height="200"></canvas>
</div>

<div class="chart-container">
  <h3>📜 Log</h3>
  <div id="logBox"></div>
</div>

<div class="footer">SFT Dashboard · auto-refresh 5s · A100</div>

<script>
const ctx = document.getElementById('lossChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#58a6ff', backgroundColor: '#58a6ff20', fill: true, tension: 0.3, pointRadius: 0 }] },
  options: { responsive: true, animation: false, scales: { x: { display: true, grid: { color: '#21262d' } }, y: { grid: { color: '#21262d' } } }, plugins: { legend: { display: false } } }
});

async function update() {
  try {
    const res = await fetch('/api/sft');
    const d = await res.json();

    document.getElementById('curStep').textContent = d.step || 0;
    const total = d.total || 5000;
    document.getElementById('totalSteps').textContent = total;
    document.getElementById('progressFill').style.width = ((d.step || 0) / total * 100).toFixed(1) + '%';
    document.getElementById('curLoss').textContent = d.loss != null ? d.loss.toFixed(4) : '—';
    document.getElementById('elapsed').textContent = d.elapsed || '—';

    if (d.running) {
      document.getElementById('statusLabel').textContent = 'Status: 🟢 Running';
      if (d.eta) document.getElementById('estTime').textContent = 'ETA: ~' + d.eta;
    } else {
      document.getElementById('statusLabel').textContent = 'Status: ⏹️ Stopped';
    }

    if (d.steps && d.losses) {
      chart.data.labels = d.steps;
      chart.data.datasets[0].data = d.losses;
      chart.update('none');
    }

    if (d.log_tail) {
      document.getElementById('logBox').textContent = d.log_tail;
      const box = document.getElementById('logBox');
      box.scrollTop = box.scrollHeight;
    }
  } catch(e) { console.error(e); }
}
update();
setInterval(update, 5000);
</script>
</body>
</html>"""


def parse_sft_log():
    steps = []
    losses = []
    current_step = 0
    current_loss = None
    start_time = None
    running = False

    if not SFT_LOG.exists():
        return {"steps": steps, "losses": losses, "step": 0, "loss": None,
                "elapsed": "—", "running": False}

    with open(SFT_LOG, "r", errors="replace") as f:
        data = f.read()

    total_steps = 5000
    for line in data.split("\n"):
        if "|" in line and "loss" in line:
            m = re.search(r"(\d+)/(\d+).*loss ([\d.]+)", line)
            if m:
                current_step = int(m.group(1))
                total_steps = int(m.group(2))
                current_loss = float(m.group(3))
                steps.append(current_step)
                losses.append(current_loss)

        if "Entraînement SFT" in line:
            running = True
        if "Done!" in line:
            running = False

    elapsed = "—"
    eta = None
    if current_step > 10:
        for line in data.split("\n"):
            m = re.search(r"(\d+):(\d+).*?(\d+)/" + str(total_steps), line)
            if m:
                h, m_, _ = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elapsed_m = h * 60 + m_
                elapsed = f"{int(elapsed_m)}m"
                remaining = total_steps - current_step
                if remaining > 0 and current_step > 0:
                    eta_m = int(remaining * elapsed_m / current_step)
                    eta = f"{eta_m}m"
                break

    return {
        "step": current_step,
        "total": total_steps,
        "loss": current_loss,
        "steps": steps[-200:],
        "losses": losses[-200:],
        "running": running,
        "elapsed": elapsed,
        "eta": eta,
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/sft":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = parse_sft_log()
            # Add log tail
            if SFT_LOG.exists():
                with open(SFT_LOG, "r", errors="replace") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 3000))
                    data["log_tail"] = f.read()
            else:
                data["log_tail"] = ""
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    port = 8083
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"SFT Dashboard: http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
