import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_added_text(old_revid, revid, session):
    """Hits the Compare API and extracts only the newly added text."""
    # If there is no previous revision (e.g., page creation), skip diffing
    if old_revid == 0:
        return ""
        
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "compare",
        "fromrev": old_revid,
        "torev": revid,
        "format": "json"
    }
    
    try:
        response = session.get(url, params=params, timeout=10).json()
        diff_html = response.get('compare', {}).get('*', '')
        
        if not diff_html:
            return ""
            
        # Parse the HTML diff provided by Wikipedia
        soup = BeautifulSoup(diff_html, 'html.parser')
        
        # Wikipedia diffs put added text inside elements with the class 'diff-addedline'
        added_lines = soup.find_all(class_="diff-addedline")
        added_text = " ".join([line.get_text(separator=" ", strip=True) for line in added_lines])
        
        return added_text
        
    except Exception as e:
        # If the API times out or fails on a weird edge case, just return empty text
        return ""

if __name__ == "__main__":
    print("Loading historical_edits.csv...")
    df = pd.read_csv("historical_edits.csv")
    
    # Take a 2000-row sample to keep the hackathon moving fast
    # We ensure we keep the same revert ratio by using stratify-like sampling
    df_sample = df.groupby('is_reverted', group_keys=False).apply(lambda x: x.sample(min(len(x), 1000)))
    # Shuffle the dataset
    df_sample = df_sample.sample(frac=1).reset_index(drop=True)
    
    print(f"Fetching diff text for {len(df_sample)} edits...")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "WikiRevertMiniProject/1.0 (Contact: local_dev@example.com)"})
    
    added_texts = []
    
    # tqdm gives us a nice progress bar in the terminal
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        text = extract_added_text(row['old_revid'], row['revid'], session)
        added_texts.append(text)
        time.sleep(0.1)  # Sleep briefly so Wikipedia doesn't block our IP
        
    df_sample['added_text'] = added_texts
    
    # Save the finalized dataset ready for Machine Learning
    df_sample.to_csv("training_data.csv", index=False)
    print("\nDone! Saved to training_data.csv")
    print(df_sample[['is_anon', 'is_reverted', 'added_text']].head(10))