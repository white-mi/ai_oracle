"""Shared loaders for the published outputs of arXiv:2605.30802 (repo clone)."""
import glob, json, os, re
import numpy as np, pandas as pd
from scipy.stats import binomtest

REPO = os.environ.get("KOTA_REPO", "./repo")
M = ["gpt4o", "deepseek", "llama"]          # column prefixes; "gpt4o" is gpt-5-nano in the final runs
NAMES = {"gpt4o": "GPT-5 nano", "deepseek": "DeepSeek V3", "llama": "Llama 3.3 70B"}

def norm(s):
    return s.astype(str).str.strip().str.upper().values

_ws = lambda col: col.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def key(df, q="question_text", c="resolution_criteria", i="question_id"):
    """row identity = market id + question text + resolution criteria (question text alone is NOT unique)"""
    return (df[i].astype(str) + "|" + _ws(df[q]) + "|" + _ws(df[c])).values

def qkey(df, q="question_text", i="question_id"):
    """evidence identity = market id + question text (the search query uses the question text only)"""
    return (df[i].astype(str) + "|" + _ws(df[q])).values

def load():
    A = pd.read_csv(f"{REPO}/results/Final_Architecture_A_results.csv")
    B = pd.read_csv(f"{REPO}/results/Final_Architecture_B_results.csv")
    D = pd.read_csv(f"{REPO}/data/kalshibench_v2_evaluation_Filtered.csv")
    A["key"] = key(A); B["key"] = key(B); D["key"] = key(D, "question", "description", "id")
    A["qkey"] = qkey(A); B["qkey"] = qkey(B); D["qkey"] = qkey(D, "question", "id")
    for df in (A, B):
        df["truth"] = norm(df.ground_truth); df["y"] = (df["truth"] == "YES").astype(int)
    for m in M:
        A[f"{m}_d"] = norm(A[f"{m}_decision"]); A[f"{m}_ok"] = (A[f"{m}_d"] == A["truth"]).astype(int)
        for r in (1, 2):
            B[f"{m}_d{r}"] = norm(B[f"{m}_round{r}_decision"]); B[f"{m}_ok{r}"] = (B[f"{m}_d{r}"] == B["truth"]).astype(int)
    A["yes_votes_n"] = sum((A[f"{m}_d"] == "YES").astype(int) for m in M)
    A["maj"] = np.where(A.yes_votes_n >= 2, "YES", "NO"); A["maj_ok"] = (A.maj == A["truth"]).astype(int)
    w = sum(np.where(A[f"{m}_d"] == "YES", 1, -1) * A[f"{m}_confidence"].values for m in M)
    A["wtd"] = np.where(w > 0, "YES", "NO"); A["wtd_ok"] = (A.wtd == A["truth"]).astype(int)
    A["unan"] = A.yes_votes_n.isin([0, 3]).astype(int)
    A["mean_conf"] = np.mean([A[f"{m}_confidence"].values for m in M], axis=0)
    for r in (1, 2):
        B[f"yes{r}"] = sum((B[f"{m}_d{r}"] == "YES").astype(int) for m in M)
        B[f"maj{r}"] = np.where(B[f"yes{r}"] >= 2, "YES", "NO"); B[f"maj{r}_ok"] = (B[f"maj{r}"] == B["truth"]).astype(int)
    B["final"] = norm(B.final_decision); B["final_ok"] = (B.final == B["truth"]).astype(int)
    return A, B, D

def packets():
    pk = {}
    for f in glob.glob(f"{REPO}/cache/evidence/*.json"):
        e = json.load(open(f)); e["_file"] = os.path.basename(f)
        pk[e["question_id"] + "|" + re.sub(r"\s+", " ", e["question_text"]).strip()] = e
    return pk

def mcnemar(x, y):
    x = np.asarray(x).astype(bool); y = np.asarray(y).astype(bool)
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return n10, n01, p

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h

def cluster_boot(values, clusters, B=5000, seed=0, fn=np.mean):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float); clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(uniq))]
    out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx[j] for j in s]); out[b] = fn(values[ii])
    return np.percentile(out, [2.5, 97.5])
