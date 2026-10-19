import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

def fetch_historical_edits(target_count=5000):
    """Fetches edits from 72 hours ago and labels them as reverted or survived."""
    
    # 1. Setup our "Time Machine" timestamp (72 hours ago)
    target_time = datetime.now(timezone.utc) - timedelta(hours=72)
    rcstart = target_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    session = requests.Session()
    session.headers.update({"User-Agent": "WikiRevertMiniProject/1.0 (Contact: local_dev@example.com)"})
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    dataset = []
    
    params = {
        "action": "query",
        "format": "json",
        "list": "recentchanges",
        "rcstart": rcstart,
        "rcdir": "older",           # Go backward in time from our start point
        "rclimit": 500,             # Max allowed per request
        "rcnamespace": 0,           # Only main encyclopedia articles
        "rctype": "edit",           # Only actual text edits (no page creations/logs)
        "rcprop": "ids|title|user|userid|parsedcomment|sizes|tags",
    }
    
    print(f"Fetching {target_count} edits starting from {rcstart}...")
    
    while len(dataset) < target_count:
        response = session.get(API_URL, params=params)
        data = response.json()
        
        if "query" not in data or "recentchanges" not in data["query"]:
            print("Error or end of data reached.")
            break
            
        changes = data["query"]["recentchanges"]
        
        for change in changes:
            # Skip bot edits (usually automated maintenance)
            if 'bot' in change.get('tags', []):
                continue
                
            # 2. Extract our metadata features
            old_len = change.get('oldlen', 0)
            new_len = change.get('newlen', 0)
            
            # 3. The Golden Label: Was this reverted?
            # MediaWiki automatically applies these tags if the edit was undone
            tags = change.get('tags', [])
            is_reverted = 1 if any(t in tags for t in ['mw-reverted', 'mw-undo', 'mw-rollback']) else 0
            
            dataset.append({
                "revid": change["revid"],
                "old_revid": change["old_revid"],
                "title": change["title"],
                "user": change["user"],
                "is_anon": 1 if change["userid"] == 0 else 0, # userid 0 means anonymous IP
                "comment": change.get("parsedcomment", ""),
                "bytes_added": max(0, new_len - old_len),
                "bytes_deleted": max(0, old_len - new_len),
                "is_reverted": is_reverted
            })
            
            if len(dataset) >= target_count:
                break
                
        print(f"Collected {len(dataset)} / {target_count} edits...")
        
        # Pagination: Set the next starting point if we need more
        if "continue" in data:
            params.update(data["continue"])
            time.sleep(1) # Be polite to the API
        else:
            break
            
    df = pd.DataFrame(dataset)
    return df

if __name__ == "__main__":
    df = fetch_historical_edits(target_count=5000)
    
    print("\n--- Dataset Overview ---")
    print(f"Total Edits Collected: {len(df)}")
    print(f"Revert Rate: {df['is_reverted'].mean():.2%}")
    print("\nSample Data:")
    print(df[['title', 'is_anon', 'bytes_added', 'bytes_deleted', 'is_reverted']].head())
    
    df.to_csv("historical_edits.csv", index=False)
    print("\nSaved to historical_edits.csv")