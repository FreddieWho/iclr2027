"""O06 native VLM: independent one-image requests and append-only raw receipts."""
import argparse,hashlib,json,re,time,platform
from pathlib import Path
import numpy as np

def parse_response(text):
 s=text.strip()
 if re.fullmatch(r'YES[.!]?',s,re.I):return 1,'parsed'
 if re.fullmatch(r'NO[.!]?',s,re.I):return 0,'parsed'
 if re.search(r"cannot|can't|unable|sorry|refuse",s,re.I):return None,'refusal'
 return None,'parse_failure'

def summarize(rows):
 by={};sanity=[]
 for r in rows:
  if r['kind']=='sanity':sanity.append(r)
  else:by.setdefault(r['parent'],{})[r['state']]=r
 quads=[r for r in by.values() if set(r)=={'base','A','B','AB'}]
 correctness=np.array([[q[s]['pred']==q[s]['label'] for s in ['base','A','B','AB']] for q in quads],bool)
 parsed=np.array([[q[s]['parse_status']=='parsed' for s in ['base','A','B','AB']] for q in quads],bool)
 report={'completed_requests':len(rows),'complete_quartets':len(quads),'parse_failure':sum(r['parse_status']=='parse_failure' for r in rows),'refusal':sum(r['parse_status']=='refusal' for r in rows),
  'sanity_n':len(sanity),'sanity_accuracy_all_requests':sum(r['pred']==r['label'] for r in sanity)/len(sanity) if sanity else None,
  'all_requests_denominator_note':'parse failures/refusals count as incorrect; report separately, never drop silently'}
 if len(quads):
  atoms=correctness[:,1:3].all(1);joint=correctness[:,1:].all(1);ccm=(~correctness[:,3])&atoms
  report.update({'J_A_B_AB':float(joint.mean()),'J_base_A_B_AB':float(correctness.all(1).mean()),'atomic_A_accuracy':float(correctness[:,1].mean()),'atomic_B_accuracy':float(correctness[:,2].mean()),'atomic_both_correct_n':int(atoms.sum()),'conditional_AB_error_given_atoms_correct':float(ccm.sum()/atoms.sum()) if atoms.any() else None,'all_three_parsed_n':int(parsed[:,1:].all(1).sum()),'conditional_AB_model_error_n':int((ccm&parsed[:,3]).sum()),'conditional_AB_nonresponse_n':int((ccm&~parsed[:,3]).sum())})
  rng=np.random.default_rng(26092406);draw=rng.integers(0,len(quads),(2000,len(quads)));report['J_95CI_parent_bootstrap']=np.quantile(joint[draw].mean(1),[.025,.975]).tolist()
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--model-path',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 manifest=json.loads((a.data/'manifest.json').read_text());a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'summary.json').exists():raise FileExistsError('completed immutable run exists')
 response_path=a.out/'responses.jsonl'
 if response_path.exists() and not a.resume:raise FileExistsError('partial responses exist; explicit --resume required')
 import torch,transformers
 from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
 from PIL import Image
 torch.set_num_threads(4);torch.manual_seed(26092406)
 signature={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),'model':manifest['model'],'revision':manifest['revision'],'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),
  'model_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model_path.glob('*') if p.is_file() and p.suffix in ['.json','.safetensors']},'prompt':manifest['prompt'],'max_new_tokens':16,'image_size':448,'do_sample':False,'dtype':'bfloat16'}
 receipt=a.out/'run_manifest.json'
 if receipt.exists():assert json.loads(receipt.read_text())==signature,'resume signature mismatch'
 else:receipt.write_text(json.dumps(signature,indent=2))
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(a.model_path),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
 processor=AutoProcessor.from_pretrained(str(a.model_path),min_pixels=448*448,max_pixels=448*448,local_files_only=True)
 rows=[json.loads(s) for s in response_path.read_text().splitlines()] if response_path.exists() else [];done={r['id'] for r in rows};start=time.monotonic()
 for record in manifest['records']:
  if record['id'] in done:continue
  path=a.data/record['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
  image=Image.open(path).convert('RGB')
  # No labels, parent IDs, other states, or previous responses enter messages.
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':manifest['prompt']}]}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=processor(text=[text],images=[image],padding=True,return_tensors='pt').to('cuda')
  begin=time.monotonic()
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=16,do_sample=False)
  tokens=output[0,inputs.input_ids.shape[1]:].tolist();answer=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
  pred,status=parse_response(answer);row={**record,'response':answer,'generated_token_ids':tokens,'pred':pred,'parse_status':status,'seconds':time.monotonic()-begin}
  with response_path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  rows.append(row)
  if len(rows)==8:
   (a.out/'input_parse_gate.json').write_text(json.dumps({'first8_completed':8,'parsed':sum(r['parse_status']=='parsed' for r in rows),'accuracy_not_used_for_model_or_prompt_selection':True},indent=2))
  if len(rows)%16==0:print('completed',len(rows),'of',len(manifest['records']),flush=True)
 result=summarize(rows);result.update({'wall_seconds_this_invocation':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'status':'MATRIX_COMPLETE','completed':len(rows),'expected':len(manifest['records']),'scope':'one small native VLM; atomic adequacy reported, not threshold-gated; no MLP P1 confidence comparison'})
 (a.out/'summary.json').write_text(json.dumps(result,indent=2))
 (a.out/'receipt.json').write_text(json.dumps({'status':'MATRIX_COMPLETE','completed':[r['id'] for r in rows],'expected':272,'summary':'summary.json','scope':'exploratory reused visual test bank; not fresh confirmation'},indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
