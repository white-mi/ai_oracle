import ast, sys, json
from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class Source:
    title: str; url: str; published_date: Optional[str]=None; text: str=""; highlights: List[str]=field(default_factory=list)
    def content_for_prompt(self):
        return "\n".join(f"- {i}" for i in self.highlights) if self.highlights else self.text
def get(path):
    t=ast.parse(open(path).read()); fn=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=="_format_exa_sources")
    fn.decorator_list=[]; ns={"Source":Source,"List":List}; exec(compile(ast.Module(body=[fn],type_ignores=[]),path,"exec"),ns); return ns["_format_exa_sources"]
orig=get(sys.argv[1]+"/src/resolution/debate/runner.py"); new=get(sys.argv[2]+"/src/resolution/debate/runner.py")
e=json.load(open(sys.argv[1]+"/cache/evidence/highlights__row0089_KXSNLHOST_ddc2b51a4839.json"))
src=[Source(**s) for s in e["sources"]]
class S: evidence_render="titles"
a=orig(src); b=new(S(),src)
print("identical:", a==b, "len orig", len(a), "len titles", len(b), "diff chars", len(a)-len(b), "n sources", len(src))
