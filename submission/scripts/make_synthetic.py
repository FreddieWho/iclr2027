"""Deterministic controlled histories. Demo tokenization is only for smoke testing."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from core import Tokenizer, write_json, append_jsonl, digest

def make_history(index, seed, tokenizer, target_tokens, projects=40):
    rng=random.Random(seed*100003+index)
    hid=f'syn-{seed}-{index:05d}'
    rows=[]; facts={}
    def code(prefix): return prefix+''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789',k=7))
    def add(text, day):
        eid=code('E')
        item={'eid':eid,'day':day,'text':f'[{eid}] 2025-03-{day:02d}: {text}'}
        rows.append(item);return item
    for _ in range(projects):
        p=code('P'); ident=code('K'); owner=code('W'); relay=code('W'); badge=code('J')
        v1,v2,v3=[code('Z') for _ in range(3)]
        action0,action1=code('A'),code('A')
        f={}
        f['atomic']=add(f'Project {p} stores its archive under identifier {ident}.',rng.randint(1,5))
        f['u1']=add(f'Coordinator {owner} reports that project {p} uses site {v1}.',6)
        f['u2']=add(f'Coordinator {owner} moves project {p} from {v1} to {v2}; {v1} is superseded.',13)
        f['u3']=add(f'Coordinator {owner} confirms the latest site of project {p} is {v3}; {v2} is superseded.',23)
        f['rel1']=add(f'Project {p} is assigned to coordinator {owner}.',4)
        f['rel2']=add(f'Coordinator {owner} delegates handoff to operator {relay}.',9)
        f['rel3']=add(f'Operator {relay} uses badge {badge} for handoffs.',17)
        f['rule']=add(f'For project {p}, the routine review action is {action0}. For fragile shipments, use exception action {action1} instead.',7)
        f['revoke']=add(f'Project {p} revokes the fragile-shipment exception. Fragile shipments now use the routine review action {action0}. No other rule changes.',25)
        facts[p]=(ident,v3,badge,action0,f)
    rng.shuffle(rows);rows.sort(key=lambda x:x['day'])
    target_projects=rng.sample(list(facts),8)
    questions=[]
    for t,typ in enumerate(['atomic','atomic','update','update','relational','relational','constraint','constraint']):
        p=target_projects[t];ident,v3,badge,a0,f=facts[p]
        if typ=='atomic': question=f'What is the archive identifier for project {p}?';ans=ident;keys=['atomic']
        elif typ=='update': question=f'As of March 28, 2025, which site does project {p} currently use?';ans=v3;keys=['u1','u2','u3']
        elif typ=='relational': question=f'What badge is used by the operator who handles the handoff delegated by the coordinator assigned to project {p}?';ans=badge;keys=['rel1','rel2','rel3']
        else: question=f'As of March 28, 2025, what review action applies to a fragile shipment for project {p}?';ans=a0;keys=['rule','revoke']
        questions.append({'history_id':hid,'group_id':hid,'question_id':f'{hid}-q{t}','question_type':typ,'question':question,'answers':[ans],'scoring':'exact','gold_evidence':[f[k]['text'] for k in keys]})
    # Add unrelated but varied events; all events remain in chronological order.
    # Never cut a target event to obtain a desired input length.
    base_n=tokenizer.count('\n'.join(r['text'] for r in rows))
    if base_n>target_tokens:
        raise ValueError(f'Base events require {base_n} tokens; raise --tokens or lower --projects uniformly in calibration.')
    n_est=base_n
    while n_est<target_tokens:
        p=code('P');w=code('W');v=code('Z');day=rng.randint(1,28);eid=code('E')
        style=rng.randrange(4)
        if style==0:body=f'Project {p} stores an archive under identifier {code("K")}; coordinator {w} recorded the filing.'
        elif style==1:body=f'Coordinator {w} confirms project {p} now uses site {v}; its former site {code("Z")} is superseded.'
        elif style==2:body=f'Project {p} assigns coordinator {w}, who delegates the handoff to operator {code("W")} with badge {code("J")}.'
        else:body=f'For project {p}, the routine action is {code("A")}; fragile shipments use exception {code("A")} until revoked.'
        text=f'[{eid}] 2025-03-{day:02d}: {body}'
        rows.append({'eid':eid,'day':day,'text':text,'filler':True})
        n_est+=tokenizer.count(text+'\n')
    rng.shuffle(rows);rows.sort(key=lambda x:x['day'])
    text='\n'.join(r['text'] for r in rows)
    while tokenizer.count(text)>target_tokens:
        pos=next((j for j in range(len(rows)-1,-1,-1) if rows[j].get('filler')),None)
        if pos is None:raise ValueError('Base event set is too long under actual tokenization')
        rows.pop(pos);text='\n'.join(r['text'] for r in rows)
    public={'history_id':hid,'group_id':hid,'source':'controlled_synthetic','text':text,'generation_seed':seed,'n_tokens':tokenizer.count(text),'tokenizer':tokenizer.name,'mock_tokenizer':tokenizer.is_demo}
    return public,questions

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--n',type=int,default=32);a.add_argument('--tokens',type=int,default=32768);a.add_argument('--tokenizer',default='tiktoken:cl100k_base');a.add_argument('--seed',type=int,default=20260910);a.add_argument('--projects',type=int,default=40)
    args=a.parse_args();out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise SystemExit('Refusing to overwrite a nonempty dataset directory')
    out.mkdir(parents=True,exist_ok=True);tok=Tokenizer(args.tokenizer)
    # Smoke protocol uses fewer projects and explicitly different domain size.
    projects=8 if tok.is_demo and args.tokens<10000 else args.projects
    hashes=[]
    for i in range(args.n):
        h,qs=make_history(i,args.seed,tok,args.tokens,projects)
        append_jsonl(out/'histories.jsonl',h)
        for q in qs: append_jsonl(out/'queries.jsonl',q)
        hashes.append(digest(h))
    write_json(out/'manifest.json',{'n_histories':args.n,'history_hashes':hashes,'generation_seed':args.seed,'nominal_tokens':args.tokens,'tokenizer':tok.name,'is_smoke':tok.is_demo,'query_blind_contract':'Compressor may read histories.jsonl only. Queries/gold are withheld until memories are frozen.'})
    print(f'Created {args.n} histories in {out}; is_smoke={tok.is_demo}')
if __name__=='__main__':main()
