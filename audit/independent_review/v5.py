import sys, math, json
import pandas as pd, numpy as np
R=sys.argv[1]
A=pd.read_csv(f"{R}/results/Final_Architecture_A_results.csv"); B=pd.read_csv(f"{R}/results/Final_Architecture_B_results.csv")
up=lambda s: s.astype(str).str.strip().str.upper()
M=["gpt4o","deepseek","llama"]; A["t"]=up(A.ground_truth); B["t"]=up(B.ground_truth)
for m in M: A[m+"_ok"]=up(A[m+"_decision"])==A.t
w=sum(np.where(up(A[m+"_decision"])=="YES",1.0,-1.0)*A[m+"_confidence"].astype(float) for m in M)
A["wtd_ok"]=np.where(w>0,"YES","NO")==A.t
A["maj_ok"]=up(A.final_decision)==A.t
B["fin_ok"]=up(B.final_decision)==B.t
paper={"Education":100.0,"Health":50.0,"Financials":90.0,"Sports":92.2,"Social":76.9,"Elections":95.0,"Entertainment":87.8,"World":86.2,"Mentions":76.9,"Politics":76.2,"Climate and Weather":77.8,"Economics":57.1,"Crypto":67.8,"Science and Technology":40.0,"Companies":62.0,"Transportation":75.0}
print(sorted(A.category.unique()))
rows=[]
for c,pb in paper.items():
    s=A[A.category==c]
    v={m:round(100*s[m+"_ok"].mean(),1) for m in M}
    best=max(v.values())
    who=[m for m in M if abs(v[m]-pb)<0.06]
    rows.append((c,len(s),pb,v,best,who,round(100*s.wtd_ok.mean(),1),round(100*s.maj_ok.mean(),1),round(100*B[B.category==c].fin_ok.mean(),1)))
for r in rows: print(r)
print("categories where paper value==GPT:", sum('gpt4o' in r[5] for r in rows), "; == only GPT:", sum(r[5]==['gpt4o'] for r in rows), "; another model strictly better than paper:", [r[0] for r in rows if r[4]>r[2]+0.05])
# Politics
s=A[A.category=="Politics"]; print("Politics wtd",round(100*s.wtd_ok.mean(),2),"best", max(100*s[m+'_ok'].mean() for m in M))
# Table 5 on script pairing
a2,b2=A.copy(),B.copy()
for df in (a2,b2): df["mk"]=df.question_text.astype(str)+"||"+df.ground_truth.astype(str); df["occ"]=df.groupby("mk").cumcount()
P=a2.merge(b2,on=["mk","occ"],suffixes=("_a","_b"))
for col in ("maj_ok","wtd_ok"):
    x=P[col]; y=P.fin_ok
    print("Table5",col,int((x&y).sum()),int((x&~y).sum()),int((~x&y).sum()),int((~x&~y).sum()),"A acc",round(100*x.mean(),2),"B acc",round(100*y.mean(),2))
# does weighted_final column exist in analysis script? check the committed script's definition separately
# unique-market
def wil(k,n,z=1.96):
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return round(c-h,3),round(c+h,3)
for lab,df,col in (("A final",A,A.maj_ok),("A wtd",A,A.wtd_ok),("B final",B,B.fin_ok)):
    f=df.assign(ok=col).drop_duplicates("question_id",keep="first"); k=int(f.ok.sum()); n=len(f); print(lab,k,n,round(100*k/n,2),wil(k,n))
for m in M:
    for r in (1,2):
        f=B.assign(ok=up(B[f"{m}_round{r}_decision"])==B.t).drop_duplicates("question_id"); k=int(f.ok.sum()); print("B",m,r,k,len(f),round(100*k/len(f),2),wil(k,len(f)))
cs=json.load(open(f"{R}/results/escalation_analysis/comparison_summary.json"))
print({k:v for k,v in cs.items() if k!="coverage_accuracy_checkpoints"})
for p in cs["coverage_accuracy_checkpoints"]: print(p)
