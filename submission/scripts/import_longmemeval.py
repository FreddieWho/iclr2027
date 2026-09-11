"""Preserve full natural histories; keep answer/evidence labels out of public text."""
from __future__ import annotations
import argparse
import hashlib
import random
from pathlib import Path
from core import load_json,append_jsonl,write_json,digest

def convert(item):
    sessions=item['haystack_sessions'];dates=item.get('haystack_dates',[]);sids=item.get('haystack_session_ids',[])
    if len(sessions)!=len(dates) or len(sessions)!=len(sids):raise ValueError('Session/date/id lengths do not align; inspect upstream schema')
    hid='lme-'+str(item['question_id']);parts=[];gold=[]
    answer_ids=set(str(x) for x in item.get('answer_session_ids',[]))
    for i,(sid,date,turns) in enumerate(zip(sids,dates,sessions)):
        for j,turn in enumerate(turns):
            if not isinstance(turn.get('content'),str):raise ValueError('Expected text turn')
            txt=f'[S{i:04d}T{j:04d}] date={date}; role={turn.get("role","unknown")}; {turn["content"]}'
            parts.append(txt)
            if turn.get('has_answer') or str(sid) in answer_ids:gold.append(txt)
    text='\n'.join(parts)
    group='lme-history-'+digest(text)[:20]
    public={'history_id':hid,'group_id':group,'source':'longmemeval_s_cleaned','text':text,'original_question_id':item['question_id']}
    q={'history_id':hid,'group_id':group,'question_id':hid+'-q','question_type':item['question_type'],'question':item['question'],'answers':[str(item['answer'])],'gold_evidence':gold,'question_date':item.get('question_date'),'scoring':'semantic_required'}
    return public,q

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=32);p.add_argument('--seed',type=int,default=20260910);a=p.parse_args()
    items=load_json(a.input)
    if not isinstance(items,list):raise SystemExit('Expected JSON list; check source schema')
    # Stratify on task metadata, not on score, gold location, or answerability.
    groups={}
    for x in items:groups.setdefault(x['question_type'],[]).append(x)
    rng=random.Random(a.seed)
    for xs in groups.values():rng.shuffle(xs)
    selected=[]
    while len(selected)<min(a.n,len(items)):
        for key in sorted(groups):
            if groups[key] and len(selected)<a.n:selected.append(groups[key].pop())
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise SystemExit('Refusing to overwrite a nonempty import directory')
    out.mkdir(parents=True,exist_ok=True)
    history_hashes=[]
    for item in selected:
        h,q=convert(item);append_jsonl(out/'histories.jsonl',h);append_jsonl(out/'queries.jsonl',q);history_hashes.append(digest(h))
    write_json(out/'manifest.json',{'source_file_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'n_histories':len(selected),'seed':a.seed,'history_hashes':history_hashes,'truncation':'NONE','scoring':'semantic_required','public_fields':'role/content/date/nonsemantic source identifiers only'})
    print(f'Imported {len(selected)} COMPLETE histories. Native context fit and semantic grading still need verification.')
if __name__=='__main__':main()
