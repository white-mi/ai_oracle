import sys, json
R=sys.argv[1]
for name, needle in (("highlights__row0004_KXAIRDROPMONAD_8ecbc7d6c53f.json","monad-airdrop-by"),("highlights__row0070_KXROMANIATOP_508b8b8b2424.json","nicolae-ciuc")):
    e=json.load(open(f"{R}/cache/evidence/{name}"))
    print("=====",name); print("Q:",e["question_text"]); print("crit:",e["resolution_criteria"][:400]); print("res date",e["resolution_date"],"retrieved",e["retrieval_timestamp"])
    for i,s in enumerate(e["sources"],1):
        flag = "  <<<" if needle in (s.get("url") or "") else ""
        print(f"  [{i}] {s.get('published_date')} {s['url']}{flag}")
    for s in e["sources"]:
        if needle in (s.get("url") or ""):
            print("---- HIGHLIGHTS of", s["url"], "title:", s["title"])
            for h in s.get("highlights") or []: print(repr(h)); print()
