"""Dashboard server for monitoring 11B MoE training in real-time."""
import os
import re
import time
import json
import subprocess
from pathlib import Path
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "checkpoints" / "train_log.txt"
TRAIN_LOG = Path("/root/pipeline.log")
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def parse_train_log():
    entries = []
    if not LOG_FILE.exists():
        return entries
    pattern = re.compile(r"step (\d+) \| train_loss ([\d.]+) \| eval_loss ([\d.]+) \| ppl ([\d.]+)")
    with open(LOG_FILE, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                entries.append({"step": int(m.group(1)), "train_loss": float(m.group(2)), "eval_loss": float(m.group(3)), "perplexity": float(m.group(4))})
    return entries


def parse_live_log():
    entries = []
    if not TRAIN_LOG.exists():
        return entries
    pattern = re.compile(r"(\d+)/100000.*?loss ([\d.]+).*?aux ([\d.]+).*?lr ([\d.e-]+).*?tok/s ([\d.]+)")
    with open(TRAIN_LOG, "r", errors="replace") as f:
        data = f.read()
    for m in pattern.finditer(data):
        entries.append({"step": int(m.group(1)), "loss": float(m.group(2)), "aux": float(m.group(3)), "lr": m.group(4), "tok_s": float(m.group(4))})
    return entries


def get_gpu_stats():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        parts = result.stdout.strip().split(", ")
        if len(parts) >= 5:
            return {"temp": float(parts[0]), "util": float(parts[1]), "mem_used": float(parts[2]), "mem_total": float(parts[3]), "power": float(parts[4])}
    except Exception:
        pass
    return None


def get_checkpoint_info():
    ckpts = []
    if CHECKPOINTS_DIR.exists():
        for f in sorted(CHECKPOINTS_DIR.iterdir()):
            if f.suffix == ".pt":
                stat = f.stat()
                ckpts.append({"name": f.name, "size_gb": round(stat.st_size / 1e9, 1), "mtime": time.strftime("%H:%M:%S", time.localtime(stat.st_mtime))})
    return ckpts


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>11B MoE Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh;padding:20px}
.header{text-align:center;margin-bottom:20px;padding:20px;background:#161b22;border-radius:12px;border:1px solid #30363d}
.header h1{font-size:1.8em;color:#58a6ff}
.header .subtitle{color:#8b949e;margin-top:5px;font-size:1.1em}
.status-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.9em;font-weight:bold;margin-top:8px}
.status-running{background:#1a7f37;color:#aff5b4}
.status-stopped{background:#da3633;color:#ffd7d5}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:20px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}
.card h3{color:#58a6ff;margin-bottom:12px;font-size:1.1em}
.stat-value{font-size:2.2em;font-weight:bold;color:#e6edf3}
.stat-label{color:#8b949e;font-size:0.9em;margin-top:4px}
.stat-row{display:flex;justify-content:space-between;margin:6px 0}
.stat-row .label{color:#8b949e}
.stat-row .value{color:#e6edf3;font-weight:bold}
.chart-container{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin-bottom:20px}
.chart-container h3{color:#58a6ff;margin-bottom:12px}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
@media(max-width:768px){.charts-grid{grid-template-columns:1fr}}
.progress-bar{width:100%;height:8px;background:#21262d;border-radius:4px;margin-top:8px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:4px;transition:width 0.5s}
.gpu-meter{display:flex;align-items:center;gap:10px;margin:8px 0}
.gpu-meter .bar{flex:1;height:12px;background:#21262d;border-radius:6px;overflow:hidden}
.gpu-meter .fill{height:100%;border-radius:6px}
.footer{text-align:center;color:#8b949e;margin-top:20px;font-size:0.85em}
#logBox{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px;font-family:'SF Mono','Fira Code',monospace;font-size:0.85em;max-height:300px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:#8b949e}
</style>
</head>
<body>
<div class="header">
<h1>MoE 11B Bilingual LLM — Training Dashboard</h1>
<div class="subtitle">9.03B params (1.33B active) | 16 experts | top-2 | EN/FR | A100 80GB | 8-bit AdamW</div>
<div id="statusBadge" class="status-badge status-running">Running</div>
</div>
<div class="grid">
<div class="card"><h3>Current Step</h3><div class="stat-value" id="curStep">—</div><div class="stat-label">/ 100,000 steps</div><div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div></div>
<div class="card"><h3>Train Loss</h3><div class="stat-value" id="curLoss">—</div><div class="stat-label" id="lossTrend"></div></div>
<div class="card"><h3>Eval Loss / PPL</h3><div class="stat-value" id="curEval">—</div><div class="stat-label" id="curPpl"></div></div>
<div class="card"><h3>Speed</h3><div class="stat-value" id="curTps">—</div><div class="stat-label">tokens/sec</div></div>
<div class="card"><h3>Elapsed / ETA</h3><div class="stat-row"><span class="label">Elapsed:</span><span class="value" id="elapsed">—</span></div><div class="stat-row"><span class="label">ETA:</span><span class="value" id="eta">—</span></div><div class="stat-row"><span class="label">Tokens seen:</span><span class="value" id="tokensSeen">—</span></div></div>
<div class="card"><h3>GPU (A100 80GB)</h3><div class="gpu-meter"><span style="width:60px">Util</span><div class="bar"><div class="fill" id="gpuUtil" style="width:0%;background:#3fb950"></div></div><span id="gpuUtilVal">—</span></div><div class="gpu-meter"><span style="width:60px">VRAM</span><div class="bar"><div class="fill" id="gpuMem" style="width:0%;background:#1f6feb"></div></div><span id="gpuMemVal">—</span></div><div class="stat-row"><span class="label">Temp:</span><span class="value" id="gpuTemp">—</span></div><div class="stat-row"><span class="label">Power:</span><span class="value" id="gpuPower">—</span></div></div>
</div>
<div class="charts-grid">
<div class="chart-container"><h3>Training Loss</h3><canvas id="lossChart" height="200"></canvas></div>
<div class="chart-container"><h3>Eval Loss & Perplexity</h3><canvas id="evalChart" height="200"></canvas></div>
</div>
<div class="charts-grid">
<div class="chart-container"><h3>Tokens/sec</h3><canvas id="tpsChart" height="200"></canvas></div>
<div class="chart-container"><h3>Aux Loss (Expert Balance)</h3><canvas id="auxChart" height="200"></canvas></div>
</div>
<div class="chart-container"><h3>Checkpoints</h3><div id="checkpointList" style="max-height:200px;overflow-y:auto"><table style="width:100%;border-collapse:collapse"><thead><tr><th style="text-align:left;padding:6px 12px;border-bottom:1px solid #30363d;color:#58a6ff">Checkpoint</th><th style="text-align:left;padding:6px 12px;border-bottom:1px solid #30363d;color:#58a6ff">Size</th><th style="text-align:left;padding:6px 12px;border-bottom:1px solid #30363d;color:#58a6ff">Saved at</th></tr></thead><tbody id="ckptBody"></tbody></table></div></div>
<div class="chart-container"><h3>Live Log</h3><div id="logBox"></div></div>
<div class="footer">11B MoE Dashboard | auto-refresh every 5s | RunPod A100 80GB</div>
<script>
const colors={blue:'#58a6ff',green:'#3fb950',orange:'#d29922',red:'#f85149',purple:'#bc8cff'};Chart.defaults.color='#8b949e';Chart.defaults.borderColor='#30363d';
function makeChart(ctx,label,color){return new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label,data:[],borderColor:color,backgroundColor:color+'20',fill:true,tension:0.3,pointRadius:0}]},options:{responsive:true,animation:false,scales:{x:{display:false},y:{grid:{color:'#21262d'}}},plugins:{legend:{display:false}}}})}
const lossChart=makeChart(document.getElementById('lossChart').getContext('2d'),'Train Loss',colors.blue);
const evalChart=new Chart(document.getElementById('evalChart').getContext('2d'),{type:'line',data:{labels:[],datasets:[{label:'Eval Loss',data:[],borderColor:colors.orange,backgroundColor:colors.orange+'20',fill:false,tension:0.3,yAxisID:'y'},{label:'Perplexity',data:[],borderColor:colors.purple,backgroundColor:colors.purple+'20',fill:false,tension:0.3,yAxisID:'y1'}]},options:{responsive:true,animation:false,scales:{x:{display:false},y:{position:'left',grid:{color:'#21262d'}},y1:{position:'right',grid:{drawOnChartArea:false}}},plugins:{legend:{labels:{color:'#c9d1d9'}}}}});
const tpsChart=makeChart(document.getElementById('tpsChart').getContext('2d'),'Tokens/sec',colors.green);
const auxChart=makeChart(document.getElementById('auxChart').getContext('2d'),'Aux Loss',colors.red);
function fmt(n){return n!=null?n.toLocaleString():'—'}
function fmtTime(s){if(!s)return'—';const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);return h>0?`${h}h ${m}m`:m>0?`${m}m ${sec}s`:`${sec}s`}
async function update(){try{const res=await fetch('/api/stats');const d=await res.json();
document.getElementById('statusBadge').className='status-badge '+(d.is_running?'status-running':'status-stopped');
document.getElementById('statusBadge').textContent=d.is_running?'Running':'Stopped';
document.getElementById('curStep').textContent=fmt(d.live_step||d.eval_step||0);
document.getElementById('progressFill').style.width=((d.live_step||0)/100000*100).toFixed(1)+'%';
document.getElementById('curLoss').textContent=d.live_loss!=null?d.live_loss.toFixed(4):'—';
if(d.eval_loss!=null){document.getElementById('curEval').textContent=d.eval_loss.toFixed(4);document.getElementById('curPpl').textContent='Perplexity: '+d.perplexity.toFixed(2)}
document.getElementById('curTps').textContent=d.tok_s!=null?Math.round(d.tok_s).toLocaleString():'—';
document.getElementById('elapsed').textContent=fmtTime(d.elapsed);document.getElementById('eta').textContent=fmtTime(d.eta);
document.getElementById('tokensSeen').textContent=fmt((d.live_step||0)*16*1023*4)+' tokens';
if(d.gpu){document.getElementById('gpuUtil').style.width=d.gpu.util+'%';document.getElementById('gpuUtilVal').textContent=d.gpu.util+'%';const memPct=(d.gpu.mem_used/d.gpu.mem_total*100).toFixed(1);document.getElementById('gpuMem').style.width=memPct+'%';document.getElementById('gpuMemVal').textContent=(d.gpu.mem_used/1024).toFixed(1)+' / '+(d.gpu.mem_total/1024).toFixed(0)+' GB';document.getElementById('gpuTemp').textContent=d.gpu.temp+'C';document.getElementById('gpuPower').textContent=d.gpu.power+'W'}
if(d.live_entries&&d.live_entries.length>0){const e=d.live_entries;const si=Math.max(1,Math.floor(e.length/500));const s=e.filter((_,i)=>i%si===0).slice(-500);lossChart.data.labels=s.map(e=>e.step);lossChart.data.datasets[0].data=s.map(e=>e.loss);lossChart.update('none');tpsChart.data.labels=s.map(e=>e.step);tpsChart.data.datasets[0].data=s.map(e=>e.tok_s);tpsChart.update('none');auxChart.data.labels=s.map(e=>e.step);auxChart.data.datasets[0].data=s.map(e=>e.aux);auxChart.update('none')}
if(d.eval_entries&&d.eval_entries.length>0){evalChart.data.labels=d.eval_entries.map(e=>e.step);evalChart.data.datasets[0].data=d.eval_entries.map(e=>e.eval_loss);evalChart.data.datasets[1].data=d.eval_entries.map(e=>e.perplexity);evalChart.update('none')}
if(d.checkpoints){document.getElementById('ckptBody').innerHTML=d.checkpoints.map(c=>`<tr><td style="padding:6px 12px;border-bottom:1px solid #30363d">${c.name}</td><td style="padding:6px 12px;border-bottom:1px solid #30363d">${c.size_gb} GB</td><td style="padding:6px 12px;border-bottom:1px solid #30363d">${c.mtime}</td></tr>`).join('')}
if(d.log_tail){document.getElementById('logBox').textContent=d.log_tail;const lb=document.getElementById('logBox');lb.scrollTop=lb.scrollHeight}
}catch(e){console.error(e)}}
update();setInterval(update,5000);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif parsed.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            eval_entries = parse_train_log()
            live_entries = parse_live_log()
            latest_live = live_entries[-1] if live_entries else {}
            latest_eval = eval_entries[-1] if eval_entries else {}
            is_running = False
            try:
                result = subprocess.run(["pgrep", "-f", "src.train"], capture_output=True, text=True, timeout=5)
                is_running = result.returncode == 0
            except Exception:
                pass
            elapsed = None
            eta = None
            if live_entries:
                step = latest_live.get("step", 0)
                if step > 10:
                    elapsed = step * 4.5
                    eta = (100000 - step) * 4.5
            data = {
                "is_running": is_running,
                "live_step": latest_live.get("step"),
                "live_loss": latest_live.get("loss"),
                "aux": latest_live.get("aux"),
                "lr": latest_live.get("lr"),
                "tok_s": latest_live.get("tok_s"),
                "eval_step": latest_eval.get("step"),
                "eval_loss": latest_eval.get("eval_loss"),
                "perplexity": latest_eval.get("perplexity"),
                "elapsed": elapsed,
                "eta": eta,
                "gpu": get_gpu_stats(),
                "checkpoints": get_checkpoint_info(),
                "eval_entries": eval_entries[-100:],
                "live_entries": live_entries[-2000:],
                "log_tail": self._get_log_tail(),
            }
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _get_log_tail(self):
        if not TRAIN_LOG.exists():
            return ""
        try:
            with open(TRAIN_LOG, "r", errors="replace") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 3000))
                return f.read()
        except Exception:
            return ""

    def log_message(self, format, *args):
        pass


def main():
    port = 8888
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
