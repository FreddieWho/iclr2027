"""Frozen, model-blind O06 sample and prompts; no inference selection."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
PROMPT='Does the red line segment cross the blue line segment strictly inside both segments? Ignore line thickness: judge the center lines. Reply with exactly YES or NO.'
REVISION='66285546d2b821cf421d4f5eb2576359d3770cd3'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'artifacts/e1a933_review/vision_canonical224_v2/data.npz';d=np.load(source);rng=np.random.default_rng(26092406)
 selected=[];seen=set()
 for i in rng.permutation(len(d['quartet_parents'])):
  p=int(d['quartet_parents'][i])
  if p not in seen:selected.append(int(i));seen.add(p)
  if len(selected)==64:break
 assert len(selected)==64
 records=[]
 def save(image,name,label,kind,parent,state):
  arr=np.uint8(np.clip(image.transpose(1,2,0)*255,0,255));im=Image.fromarray(arr).resize((448,448),Image.Resampling.BILINEAR)
  path=a.out/(name+'.png');im.save(path)
  records.append(dict(id=name,image=path.name,label=int(label),kind=kind,parent=parent,state=state,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prompt=PROMPT))
 for q,i in enumerate(selected):
  for state,img,y in zip(['base','A','B','AB'],d['quartet_images'][i],d['quartet_labels'][i]):save(img,f'q{q:03d}_{state}',y,'quartet',int(d['quartet_parents'][i]),state)
 from vision_protocol import render
 for i in range(16):
  positive=i%2==0
  x=np.array([[-.65,-.55],[.65,.55],[-.65,.55],[.65,-.55]]) if positive else np.array([[-.7,-.5],[.7,-.5],[-.7,.5],[.7,.5]])
  angle=(i//2)*np.pi/4;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);x=x@rot.T*.8
  save(render(x,np.random.default_rng(92400+i)),f'sanity{i:02d}',positive,'sanity',i,'single')
 manifest={'model':'Qwen/Qwen2.5-VL-3B-Instruct','revision':REVISION,'seed':26092406,'quartets':64,'unique_parents':64,'sanity':16,'requests':272,'max_new_tokens':16,'do_sample':False,'dtype':'bfloat16','image_pixels':448*448,'prompt':PROMPT,
  'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selected_source_quartet_indices':selected,
  'sample_rule':'first 64 unique physical parents in seeded permutation; no model output or labels used in selection',
  'render':'existing corrected canonical native64 RGB, quantize uint8 then bilinear448 identically for every state; no new information from upscale',
  'scope':'one 3B native VLM external bridge; no extrapolation to all VLMs',
  'gate':'first8 only check runtime and parse availability; no accuracy gate/model swapping; all raw responses retained',
  'official_sources':['https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct','https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE'],
  'records':records}
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('frozen',a.out,len(records))
if __name__=='__main__':main()
