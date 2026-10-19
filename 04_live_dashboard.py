import json
import requests
import joblib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sseclient import SSEClient
from rich.console import Console
from rich.table import Table

console = Console()
console.print("[bold cyan]Loading trained model (wiki_revert_model.joblib)...[/bold cyan]")
model = joblib.load('wiki_revert_model.joblib')

session = requests.Session()
session.headers.update({"User-Agent": "WikiRevertMiniProject/1.0 (Contact: local_dev@example.com)"})

def extract_added_text(old_revid, revid):
    if not old_revid or old_revid == 0:
        return ""
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "compare", "fromrev": old_revid, "torev": revid, "format": "json"}
    try:
        res = session.get(url, params=params, timeout=5).json()
        html = res.get('compare', {}).get('*', '')
        soup = BeautifulSoup(html, 'html.parser')
        return " ".join([line.get_text(separator=" ", strip=True) for line in soup.find_all(class_="diff-addedline")])
    except Exception:
        return ""

def monitor_live_stream():
    STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
    console.print("[bold magenta]🔌 Connecting to Wikipedia Live Stream...[/bold magenta]")
    
    response = session.get(STREAM_URL, stream=True, timeout=30)
    client = SSEClient(response)
    console.print("[bold green]✅ Connected! Scoring live human edits in real time (Press Ctrl+C to stop)...[/bold green]\n")
    
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
            user = str(change.get("user", ""))
            comment = str(change.get("comment", ""))
            
            old_len = change.get("length", {}).get("old", 0)
            new_len = change.get("length", {}).get("new", 0)
            bytes_added = max(0, new_len - old_len)
            bytes_deleted = max(0, old_len - new_len)
            
            rev_old = change.get("revision", {}).get("old", 0)
            rev_new = change.get("revision", {}).get("new", 0)
            added_text = extract_added_text(rev_old, rev_new)
            
            is_anon = int(user.startswith('~') or user.replace('.', '').isdigit())
            
            live_data = pd.DataFrame([{
                'is_anon': is_anon,
                'log_bytes_added': np.log1p(bytes_added),
                'log_bytes_deleted': np.log1p(bytes_deleted),
                'added_text': added_text
            }])
            
            prob = model.predict_proba(live_data)[0][1]
            
            if prob >= 0.55:
                badge = "[bold white on red] 🚨 HIGH REVERT RISK [/bold white on red]"
                border = "red"
            elif prob >= 0.45:
                badge = "[bold black on yellow] ⚠️  MODERATE RISK   [/bold black on yellow]"
                border = "yellow"
            else:
                badge = "[bold black on green] ✅ SAFE EDIT        [/bold black on green]"
                border = "green"
                
            table = Table(show_header=False, border_style=border)
            table.add_row("Article", f"[bold]{title}[/bold] (https://en.wikipedia.org/wiki/?diff={rev_new})")
            table.add_row("Editor", f"{user} {'[bold red](Temp/Anon)[/bold red]' if is_anon else '(Registered)'}")
            table.add_row("Delta", f"+{bytes_added}B / -{bytes_deleted}B | Summary: {comment[:50] or 'None'}")
            table.add_row("Revert Risk", f"[bold]{prob:.1%}[/bold]")
            if added_text:
                table.add_row("Added Text", f"[italic]{added_text[:90]}...[/italic]")
                
            console.print(badge)
            console.print(table)

if __name__ == "__main__":
    monitor_live_stream()