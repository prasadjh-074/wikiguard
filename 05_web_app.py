import json
import requests
import joblib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sseclient import SSEClient
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI()

print("Loading trained model (wiki_revert_model.joblib)...")
model = joblib.load("wiki_revert_model.joblib")

# Extract pipeline pieces so we can show real-time feature attributions in the UI
preprocessor = model.named_steps["preprocessor"]
clf = model.named_steps["clf"]
tfidf = preprocessor.named_transformers_["text"]
dense_cols = list(preprocessor.transformers_[0][2])
feature_names = np.array(dense_cols + list(tfidf.get_feature_names_out()))
coefs = clf.coef_[0]
intercept = float(clf.intercept_[0])

session = requests.Session()
session.headers.update({"User-Agent": "WikiRevertWebUI/1.0 (Contact: local_dev@example.com)"})

def extract_added_text(old_revid, revid):
    if not old_revid or old_revid == 0:
        return ""
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "compare", "fromrev": old_revid, "torev": revid, "format": "json"}
    try:
        res = session.get(url, params=params, timeout=5).json()
        html = res.get("compare", {}).get("*", "")
        soup = BeautifulSoup(html, "html.parser")
        return " ".join([line.get_text(separator=" ", strip=True) for line in soup.find_all(class_="diff-addedline")])
    except Exception:
        return ""

def calc_caps_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    return sum(1 for c in letters if c.isupper()) / len(letters) if len(letters) > 5 else 0.0

def build_feature_row(user, comment, bytes_added, bytes_deleted, added_text):
    is_anon = 1 if (user.startswith("~") or user.replace(".", "").isdigit()) else 0
    row = {
        "is_anon": is_anon,
        "is_temp_or_anon": is_anon,
        "no_summary": 1 if comment.strip() == "" else 0,
        "log_bytes_added": float(np.log1p(bytes_added)),
        "log_bytes_deleted": float(np.log1p(bytes_deleted)),
        "caps_ratio": calc_caps_ratio(added_text),
        "exclamation_count": added_text.count("!"),
        "added_text": added_text
    }
    return pd.DataFrame([row]), is_anon

def explain_prediction(live_df):
    """Computes exact log-odds contributions (x_i * beta_i) for the live edit."""
    X_trans = preprocessor.transform(live_df)
    x_arr = X_trans.toarray()[0] if hasattr(X_trans, "toarray") else np.asarray(X_trans)[0]
    contribs = x_arr * coefs
    
    non_zero_idx = np.where(np.abs(contribs) > 0.02)[0]
    sorted_idx = non_zero_idx[np.argsort(np.abs(contribs[non_zero_idx]))[::-1]][:5]
    
    reasons = []
    for idx in sorted_idx:
        reasons.append({
            "feature": str(feature_names[idx]),
            "log_odds": round(float(contribs[idx]), 2)
        })
    return reasons, round(float(np.sum(contribs) + intercept), 2)

def event_generator():
    stream_url = "https://stream.wikimedia.org/v2/stream/recentchange"
    response = session.get(stream_url, stream=True, timeout=30)
    client = SSEClient(response)
    
    for event in client.events():
        if not event.data:
            continue
        try:
            change = json.loads(event.data)
        except json.JSONDecodeError:
            continue

        if (change.get("wiki") == "enwiki"
            and change.get("type") == "edit"
            and change.get("namespace") == 0
            and not change.get("bot")):
            
            title = change.get("title", "Unknown")
            user = change.get("user") or ""
            comment = change.get("comment") or ""
            
            old_len = change.get("length", {}).get("old", 0)
            new_len = change.get("length", {}).get("new", 0)
            bytes_added = max(0, new_len - old_len)
            bytes_deleted = max(0, old_len - new_len)
            
            rev_old = change.get("revision", {}).get("old", 0)
            rev_new = change.get("revision", {}).get("new", 0)
            added_text = extract_added_text(rev_old, rev_new)
            
            live_df, is_anon = build_feature_row(user, comment, bytes_added, bytes_deleted, added_text)
            prob = float(model.predict_proba(live_df)[0][1])
            reasons, total_log_odds = explain_prediction(live_df)
            
            payload = {
                "revid": rev_new,
                "title": title,
                "user": user,
                "is_anon": bool(is_anon),
                "comment": comment,
                "bytes_added": bytes_added,
                "bytes_deleted": bytes_deleted,
                "added_text": added_text[:280],
                "prob": round(prob, 4),
                "log_odds": total_log_odds,
                "reasons": reasons,
                "diff_url": f"https://en.wikipedia.org/wiki/?diff={rev_new}"
            }
            yield f"data: {json.dumps(payload)}\n\n"

@app.get("/stream")
def stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>WikiGuard Live — L1 Logistic Regression Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background: radial-gradient(circle at top left, #0f172a, #020617); color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }
    .glass { background: rgba(30, 41, 59, 0.65); backdrop-filter: blur(12px); border: 1px solid rgba(148, 163, 184, 0.15); }
    @keyframes slideIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    .card-enter { animation: slideIn 0.25s ease-out forwards; }
  </style>
</head>
<body class="min-h-screen p-6">
  <div class="max-w-7xl mx-auto space-y-6">
    <!-- Top Header -->
    <div class="glass rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-3">
          <span id="statusDot" class="h-3 w-3 rounded-full bg-emerald-400 animate-pulse"></span>
          <h1 class="text-2xl font-bold tracking-tight">WikiGuard Live <span class="text-indigo-400 font-normal text-lg">| L1 Logistic Regression Stream</span></h1>
        </div>
        <p class="text-slate-400 text-sm mt-1">Real-time revert probability & log-odds attribution on English Wikipedia</p>
      </div>
      <div class="flex flex-wrap items-center gap-4">
        <div class="glass px-4 py-2 rounded-xl flex items-center gap-3">
          <label class="text-xs text-slate-300 uppercase tracking-wider font-semibold">Alert Threshold</label>
          <input id="thresholdSlider" type="range" min="30" max="85" value="55" class="accent-indigo-500 cursor-pointer">
          <span id="thresholdVal" class="text-sm font-mono font-bold text-indigo-400">55%</span>
        </div>
        <button id="filterBtn" onclick="toggleFilter()" class="glass px-4 py-2 rounded-xl text-sm font-medium hover:bg-slate-800 transition">
          Show: All Edits
        </button>
        <button id="pauseBtn" onclick="togglePause()" class="bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-xl text-sm font-semibold transition">
          ⏸ Pause Stream
        </button>
      </div>
    </div>

    <!-- KPI Row -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="glass rounded-2xl p-5">
        <p class="text-xs uppercase tracking-wider text-slate-400">Edits Scored</p>
        <p id="kpiTotal" class="text-3xl font-bold mt-1 font-mono">0</p>
      </div>
      <div class="glass rounded-2xl p-5 border-red-500/30">
        <p class="text-xs uppercase tracking-wider text-red-400">High-Risk Flagged</p>
        <p id="kpiFlagged" class="text-3xl font-bold mt-1 font-mono text-red-400">0</p>
      </div>
      <div class="glass rounded-2xl p-5">
        <p class="text-xs uppercase tracking-wider text-slate-400">Mean Revert Prob</p>
        <p id="kpiAvgProb" class="text-3xl font-bold mt-1 font-mono text-indigo-400">0.0%</p>
      </div>
      <div class="glass rounded-2xl p-5">
        <p class="text-xs uppercase tracking-wider text-slate-400">Anon / Temp Editors</p>
        <p id="kpiAnon" class="text-3xl font-bold mt-1 font-mono text-amber-400">0</p>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Live Feed Column (2/3 width) -->
      <div class="lg:col-span-2 space-y-3">
        <h2 class="text-sm uppercase tracking-wider text-slate-400 font-semibold px-1">Live Wikipedia Edit Stream</h2>
        <div id="feed" class="space-y-3">
          <div id="waitingMsg" class="glass rounded-2xl p-10 text-center text-slate-400">
            Listening to <code class="text-indigo-400">stream.wikimedia.org</code>... First human edit will appear in a few seconds.
          </div>
        </div>
      </div>

      <!-- High-Risk Spotlight Column (1/3 width) -->
      <div class="space-y-3">
        <h2 class="text-sm uppercase tracking-wider text-red-400 font-semibold px-1">🚨 High-Risk Watchlist</h2>
        <div id="watchlist" class="space-y-3">
          <div id="emptyWatch" class="glass rounded-2xl p-6 text-center text-slate-500 text-sm">
            Edits exceeding your threshold will pin here for quick inspection.
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let paused = false;
    let onlyRisk = false;
    let threshold = 0.55;
    let total = 0, flagged = 0, anonCount = 0, probSum = 0;

    const slider = document.getElementById('thresholdSlider');
    const thresholdVal = document.getElementById('thresholdVal');
    slider.addEventListener('input', (e) => {
      threshold = parseInt(e.target.value) / 100;
      thresholdVal.textContent = e.target.value + '%';
    });

    function togglePause() {
      paused = !paused;
      const btn = document.getElementById('pauseBtn');
      const dot = document.getElementById('statusDot');
      btn.textContent = paused ? '▶ Resume Stream' : '⏸ Pause Stream';
      btn.className = paused ? 'bg-amber-600 hover:bg-amber-500 px-4 py-2 rounded-xl text-sm font-semibold transition' : 'bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-xl text-sm font-semibold transition';
      dot.className = paused ? 'h-3 w-3 rounded-full bg-amber-400' : 'h-3 w-3 rounded-full bg-emerald-400 animate-pulse';
    }

    function toggleFilter() {
      onlyRisk = !onlyRisk;
      document.getElementById('filterBtn').textContent = onlyRisk ? 'Show: High Risk Only 🚨' : 'Show: All Edits';
    }

    function renderCard(d) {
      const isRisk = d.prob >= threshold;
      const isWarn = !isRisk && d.prob >= (threshold - 0.12);
      const pct = (d.prob * 100).toFixed(1);

      const borderClass = isRisk ? 'border-red-500/60 bg-red-950/20' : (isWarn ? 'border-amber-500/40' : 'border-slate-700/50');
      const badgeClass = isRisk ? 'bg-red-500/20 text-red-300 border border-red-500/40' : (isWarn ? 'bg-amber-500/20 text-amber-300' : 'bg-emerald-500/20 text-emerald-300');
      const barColor = isRisk ? 'bg-red-500' : (isWarn ? 'bg-amber-400' : 'bg-emerald-500');

      const chips = (d.reasons || []).map(r => {
        const pos = r.log_odds > 0;
        const c = pos ? 'bg-red-500/15 text-red-300 border-red-500/30' : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30';
        const sign = pos ? '+' : '';
        return `<span class="text-xs font-mono px-2 py-0.5 rounded-md border ${c}">${r.feature}: ${sign}${r.log_odds}</span>`;
      }).join(' ');

      return `
        <div class="glass ${borderClass} rounded-2xl p-4 card-enter space-y-3">
          <div class="flex items-start justify-between gap-3">
            <div>
              <a href="${d.diff_url}" target="_blank" class="text-base font-bold text-white hover:text-indigo-400 transition underline decoration-slate-600">
                ${d.title} ↗
              </a>
              <div class="text-xs text-slate-400 mt-0.5 flex flex-wrap items-center gap-2">
                <span>👤 <strong class="text-slate-200">${d.user}</strong></span>
                ${d.is_anon ? '<span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[10px] font-semibold">TEMP / ANON</span>' : ''}
                <span>•</span>
                <span class="text-emerald-400 font-mono">+${d.bytes_added}B</span>
                <span class="text-red-400 font-mono">-${d.bytes_deleted}B</span>
              </div>
            </div>
            <div class="text-right shrink-0">
              <span class="px-2.5 py-1 rounded-lg text-xs font-mono font-bold ${badgeClass}">
                ${pct}% Revert Risk
              </span>
              <div class="text-[11px] text-slate-400 font-mono mt-1">z = ${d.log_odds}</div>
            </div>
          </div>

          <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div class="${barColor} h-full transition-all duration-300" style="width: ${Math.min(100, pct)}%"></div>
          </div>

          ${d.comment ? `<p class="text-xs text-slate-300"><span class="text-slate-500">Summary:</span> ${d.comment}</p>` : '<p class="text-xs text-amber-400/80 italic">No edit summary provided</p>'}
          ${d.added_text ? `<div class="bg-slate-950/60 p-2.5 rounded-xl text-xs text-slate-300 font-mono border border-slate-800 line-clamp-2">+ "${d.added_text}"</div>` : ''}
          ${chips ? `<div class="flex flex-wrap gap-1.5 pt-1">${chips}</div>` : ''}
        </div>
      `;
    }

    const evtSource = new EventSource('/stream');
    evtSource.onmessage = function(event) {
      if (paused) return;
      const d = JSON.parse(event.data);

      const waiting = document.getElementById('waitingMsg');
      if (waiting) waiting.remove();

      total++;
      probSum += d.prob;
      if (d.is_anon) anonCount++;
      const isRisk = d.prob >= threshold;
      if (isRisk) flagged++;

      document.getElementById('kpiTotal').textContent = total;
      document.getElementById('kpiFlagged').textContent = flagged;
      document.getElementById('kpiAvgProb').textContent = ((probSum / total) * 100).toFixed(1) + '%';
      document.getElementById('kpiAnon').textContent = anonCount;

      if (!onlyRisk || isRisk) {
        const feed = document.getElementById('feed');
        feed.insertAdjacentHTML('afterbegin', renderCard(d));
        while (feed.children.length > 25) feed.lastElementChild.remove();
      }

      if (isRisk) {
        const emptyWatch = document.getElementById('emptyWatch');
        if (emptyWatch) emptyWatch.remove();
        const watchlist = document.getElementById('watchlist');
        watchlist.insertAdjacentHTML('afterbegin', renderCard(d));
        while (watchlist.children.length > 10) watchlist.lastElementChild.remove();
      }
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

if __name__ == "__main__":
    print("🚀 WikiGuard Live Web UI starting at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")