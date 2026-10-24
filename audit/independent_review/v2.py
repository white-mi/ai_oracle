import sys, math
import pandas as pd, numpy as np
R=sys.argv[1]
A=pd.read_csv(f"{R}/results/Final_Architecture_A_results.csv"); B=pd.read_csv(f"{R}/results/Final_Architecture_B_results.csv")
up=lambda s: s.astype(str).str.strip().str.upper()
ta, tb = up(A.ground_truth), up(B.ground_truth)
M=["gpt4o","deepseek","llama"]
print("decision values A:", {m: up(A[m+"_decision"]).value_counts().to_dict() for m in M})
print("decision values B r1/r2:", {m: (up(B[m+"_round1_decision"]).value_counts().to_dict(), up(B[m+"_round2_decision"]).value_counts().to_dict()) for m in M})
n=len(A)
for m in M:
    a=(up(A[m+"_decision"])==ta).sum(); r1=(up(B[m+"_round1_decision"])==tb).sum(); r2=(up(B[m+"_round2_decision"])==tb).sum()
    print(m, a, f"{100*a/n:.2f}", r1, f"{100*r1/n:.2f}", r2, f"{100*r2/n:.2f}")
yesA=sum((up(A[m+"_decision"])=="YES").astype(int) for m in M); noA=sum((up(A[m+"_decision"])=="NO").astype(int) for m in M)
majA=np.where(yesA>noA,"YES","NO")
print("A majority (my rule yes>no else NO)", (majA==ta).sum(), " A final_decision col", (up(A.final_decision)==ta).sum(), " A is_correct", A.is_correct.astype(str).str.lower().eq("true").sum())
y1=sum((up(B[m+"_round1_decision"])=="YES").astype(int) for m in M); n1=sum((up(B[m+"_round1_decision"])=="NO").astype(int) for m in M)
maj1=np.where(y1>n1,"YES","NO"); maj1_ok=(maj1==tb)
fin_ok=(up(B.final_decision)==tb)
y2=sum((up(B[m+"_round2_decision"])=="YES").astype(int) for m in M); n2=sum((up(B[m+"_round2_decision"])=="NO").astype(int) for m in M)
maj2_ok=(np.where(y2>n2,"YES","NO")==tb)
print("B r1 majority", maj1_ok.sum(), f"{100*maj1_ok.sum()/n:.2f}", "B r2 majority", maj2_ok.sum(), "B final", fin_ok.sum(), f"{100*fin_ok.sum()/n:.2f}")
print("B tie_break", B.tie_break_method.value_counts().to_dict(), " round1 votes recorded vs mine equal:", ((B.round1_yes_votes==y1)&(B.round1_no_votes==n1)).all())
c2w=int((maj1_ok & ~fin_ok).sum()); w2c=int((~maj1_ok & fin_ok).sum())
def binom(k,nn):
    k=min(k,nn-k); return min(1,2*sum(math.comb(nn,i) for i in range(k+1))/2**nn)
print("r1maj->final c2w",c2w,"w2c",w2c,"p",round(binom(c2w,c2w+w2c),4))
# note: A vs B are different row orders -> gap decomposition is unpaired
print("A maj - B r1 maj =", (majA==ta).sum()-maj1_ok.sum(), "; B r1 maj - B final =", maj1_ok.sum()-fin_ok.sum())
# per model revision
for m in M:
    o1=up(B[m+"_round1_decision"])==tb; o2=up(B[m+"_round2_decision"])==tb
    print(m,"c2w",int((o1&~o2).sum()),"w2c",int((~o1&o2).sum()), "changed", int((up(B[m+"_round1_decision"])!=up(B[m+"_round2_decision"])).sum()))
# errors / missing decisions
for m in M:
    print(m, "A err non-null", A[m+"_error"].notna().sum(), "B r1 err", B[m+"_round1_error"].notna().sum(), "B r2 err", B[m+"_round2_error"].notna().sum())
# Table 9 / section 4.3
err={m:(up(A[m+"_decision"])!=ta).astype(int) for m in M}
print("P(L wrong|G wrong)", round(err['llama'][err['gpt4o']==1].mean(),4), "P(G wrong|L wrong)", round(err['gpt4o'][err['llama']==1].mean(),4))
