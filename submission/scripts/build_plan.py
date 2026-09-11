import argparse
from core import load_json,write_json,path_targets,digest

def build(profile):
    cells=[]
    for b in profile['budgets']:
        for arm in profile['arms']:
            for rep in profile['replicates']:
                if arm in ('raw','no_memory','oracle') and (b != profile['budgets'][0] or rep!=profile['replicates'][0]):continue
                cells.append({'arm':arm,'budget':None if arm in ('raw','no_memory','oracle') else b,'replicate':rep,'targets':path_targets(arm,b,profile['history_tokens'])})
    # Upper bounds do not assume prefix sharing and exclude retries/evidence audits.
    n=profile.get('n_histories',1)
    comp=sum(len(c['targets']) for c in cells)*n
    reads=len(cells)*n*8
    return {'profile':profile,'profile_hash':digest(profile),'cells':cells,'estimated_calls_upper_before_retries':comp+reads,'estimated_compression_calls_before_prefix_cache':comp,'estimated_reader_calls_assuming_8_questions':reads,'note':'Natural datasets often have 1 question/history. Price and native-token estimates require actual model config.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    plan=build(load_json(a.profile));write_json(a.out,plan)
    print(f"Planned {len(plan['cells'])} cells/history; no network calls. Conservative calls: {plan['estimated_calls_upper_before_retries']}")
