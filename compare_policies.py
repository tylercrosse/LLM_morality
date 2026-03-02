
import pandas as pd
import re
import os
import glob

ACTION_C = "action1"
ACTION_D = "action2"
GAME_FILES = ["IPD", "ISH", "ICN", "ICD", "BOS"]

def parse_action(text):
    if not isinstance(text, str):
        return "illegal"
    s = text.lower()
    matches = [(m.start(), m.group(0)) for m in re.finditer(r"action1|action2", s)]
    if not matches:
        return "illegal"
    matches.sort(key=lambda x: x[0])
    token = matches[0][1]
    return "C" if token == ACTION_C else "D"

def load_responses(csv_path):
    # Use python engine to better handle multi-line quotes if needed, 
    # though C engine (default) usually fine
    df = pd.read_csv(csv_path)
    if 'response (after)' not in df.columns:
        print(f"Column 'response (after)' not found in {csv_path}")
        return [], df
        
    responses = df['response (after)'].apply(parse_action).tolist()
    return responses, df

def compare_models(model1_name, model2_name):
    base_path = "/root/LLM_morality/results"
    
    print(f"Comparing {model1_name} vs {model2_name} across games...")
    
    for game in GAME_FILES:
        print(f"\n--- Checking {game} ---")
        # Try to find file pattern
        pattern = f"*{game}*.csv"
        
        path1_dir = os.path.join(base_path, model1_name, "run1")
        path2_dir = os.path.join(base_path, model2_name, "run1")
        
        files1 = glob.glob(os.path.join(path1_dir, pattern))
        files2 = glob.glob(os.path.join(path2_dir, pattern))
        
        # Filter for "independent eval" as seen in previous ls
        files1 = [f for f in files1 if "independent eval" in f]
        files2 = [f for f in files2 if "independent eval" in f]
        
        if not files1 or not files2:
            print(f"Could not find matching files for {game}")
            if not files1: print(f"  Missing in {model1_name}")
            if not files2: print(f"  Missing in {model2_name}")
            continue
            
        # Take the first match
        f1 = files1[0]
        f2 = files2[0]
        
        res1, df1 = load_responses(f1)
        res2, df2 = load_responses(f2)
        
        if not res1:
            continue
            
        min_len = min(len(res1), len(res2))
        res1 = res1[:min_len]
        res2 = res2[:min_len]
        
        matches = 0
        total = 0
        for i in range(min_len):
            total += 1
            if res1[i] == res2[i]:
                matches += 1
                
        agreement = matches / total if total > 0 else 0
        print(f"Agreement on {game} ({total} samples): {agreement:.2%}")
        if matches < total:
            print(f"  Differences: {total - matches}")

if __name__ == "__main__":
    compare_models("PT3_COREDe", "PT3_COREUt")
