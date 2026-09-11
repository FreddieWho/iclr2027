"""Binary promotion rule checker; never replaces substantive novelty/evidence audit."""
from __future__ import annotations
import argparse
from pathlib import Path
from core import load_json,write_json,digest

REQUIRED=['technical_validity','query_blind','budget_length','rewrite_controls','independent_confirmation','generalization','novelty','task_relevance']

def significant(c, threshold):
    try:
        e=float(c['estimate']);lo=float(c['ci_low']);hi=float(c['ci_high'])
        return lo<=e<=hi and abs(e)>=threshold and (lo>0 or hi<0)
    except (KeyError,ValueError,TypeError):return False

def evidence_exists(items,root):
    return bool(items) and all(isinstance(x,str) and (root/x).is_file() for x in items)

def decide(obj, root):
    root=Path(root)
    if obj.get('contains_mock',True):return {'decision':'NO_GO','reason':'TECHNICAL','detail':'Mock/unverified provenance cannot support scientific GO'}
    audits=obj.get('audits',{})
    failed=[k for k in REQUIRED if not audits.get(k,{}).get('pass') or not evidence_exists(audits.get(k,{}).get('evidence',[]),root)]
    if failed:
        reason='TECHNICAL' if any(k in failed for k in ['technical_validity','query_blind','budget_length']) else ('NOVELTY' if 'novelty' in failed else 'EVIDENCE')
        return {'decision':'NO_GO','reason':reason,'detail':'Required audit incomplete/failed','failed_audits':failed}
    passed=[]
    for c in obj.get('candidates',[]):
        if c.get('n_histories',0)<64 or not c.get('confirmatory',False) or not evidence_exists(c.get('evidence',[]),root):continue
        route=c.get('route_id')
        ok=False
        if route=='G1_ROBUST_PATH':ok=significant(c,.05)
        elif route=='G2_CONDITIONAL':
            ok=significant(c,.08) and c.get('subcondition_abs_effect',0)>=.08 and c.get('subcondition_n_histories',0)>=32
        elif route=='G3_REENCODING':
            ok=significant(c,.05) and c.get('mitigation_gain',0)>=.03 and c.get('mitigation_ci_low',-1)>0
        elif route=='G4_POLICY':
            ok=(c.get('estimate',0)>=.03 and c.get('ci_low',-1)>0) or (c.get('cost_saving_fraction',0)>=.30 and c.get('ci_low',-1)>=-.02)
        elif route=='G5_BOUNDARY':
            cs=c.get('equivalence_contrasts',[])
            ok=len(cs)>=2 and c.get('positive_control_valid',False) and c.get('secondary_setting_equivalent',False) and (c.get('cost_saving_fraction',0)>=.30 or c.get('documented_boundary',False)) and all(-.03<=q.get('ci_low',-1)<=q.get('ci_high',1)<=.03 for q in cs)
        if ok:passed.append(route)
    if passed:return {'decision':'GO','routes':passed,'detail':'Promote within tested scope; not a publication acceptance claim'}
    reason=obj.get('reason_if_none','EVIDENCE')
    if reason not in ('SMALL_EFFECT','NOVELTY','EVIDENCE','TECHNICAL'):reason='EVIDENCE'
    # SMALL_EFFECT requires actual interval evidence, not a free-text claim.
    if reason=='SMALL_EFFECT':
        cs=obj.get('small_effect_contrasts',[])
        if not cs or not all(q.get('n_histories',0)>=64 and -.03<=q.get('ci_low',-1)<=q.get('ci_high',1)<=.03 and evidence_exists(q.get('evidence',[]),root) for q in cs):reason='EVIDENCE'
    return {'decision':'NO_GO','reason':reason,'detail':'No candidate satisfies a documented GO route. EVIDENCE/TECHNICAL do not falsify the hypothesis.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--root',default='.');p.add_argument('--out',required=True);a=p.parse_args()
    obj=load_json(a.input);result=decide(obj,a.root);result['decision_input_hash']=digest(obj);write_json(a.out,result);print(result)
