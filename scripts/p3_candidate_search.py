#!/usr/bin/env python3
"""Search small pareto candidates for P3 task-geometry alignment.

Heldout labels are NOT used for ranking; only train/dev are used.
Candidates differ by one mechanism axis (constraint placement) via input centering.
Matches capacity.
"""
from __future__ import annotations
import copy, json, time, hashlib, resource
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import p3_support_geometry as geometry

# Extend FrozenGeometryModel to support new pooling modes
def _team_pool(h, team_slots):
    import torch
    pools=[]
    for team_slot in (0,1):
        mask=(team_slots==team_slot).float().unsqueeze(-1)
        pools.append((h*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1.0))
    return torch.cat(pools, dim=-1)

def _relational_pool(h, team_slots):
    import torch
    pools=[]
    for team_slot in (0,1):
        mask=(team_slots==team_slot).float().unsqueeze(-1)
        count=mask.sum(dim=1)
        summed=(h*mask).sum(dim=1)
        mean=summed/count.clamp_min(1.0)
        pair_sum=0.5*(summed.square()-(h.square()*mask).sum(dim=1))
        pair_count=0.5*count*(count-1.0)
        pair_mean=pair_sum/pair_count.clamp_min(1.0)
        pools.append(torch.where(pair_count>0.0, pair_mean, mean))
    return torch.cat(pools, dim=-1)

# Monkey-patch layer_values to support centered modes
orig_layer_values = geometry.FrozenGeometryModel.layer_values
def patched_layer_values(self, positions, team_slots, adjacency=None):
    import torch
    if positions.ndim==2:
        positions=positions.unsqueeze(0)
    if team_slots.ndim==1:
        team_slots=team_slots.unsqueeze(0)
    # Centered input mode: subtract global mean per sample before encoding
    if self.pooling_mode in ("centered_team_mean","centered_relational"):
        positions = positions - positions.mean(dim=1, keepdim=True)
    features = torch.cat([positions, torch.nn.functional.one_hot(team_slots.long(),2).float()], dim=-1)
    if self.architecture=="DeepSets-AE":
        node=self.model.node(features)
        pooled_pre=_team_pool(node, team_slots)
        pooled=self.model.project(pooled_pre)
        return {"input_node":node,"pooling_pre":pooled_pre,"pooled_embedding":pooled}
    if adjacency is None:
        raise ValueError(f"{self.model_id} requires adjacency")
    if adjacency.ndim==2:
        adjacency=adjacency.unsqueeze(0)
    encoder=self.model.encoder
    h=torch.relu(encoder.in_proj(features))
    outputs={"input_node":h}
    for idx,layer in enumerate(encoder.layers, start=1):
        h=layer(h, adjacency)
        outputs[f"message_passing_{idx}"]=h
    if self.pooling_mode in ("centered_team_mean","team_mean","team_centered"):
        pooled_pre=_team_pool(h, team_slots)
    elif self.pooling_mode in ("relational_pairwise","centered_relational"):
        pooled_pre=_relational_pool(h, team_slots)
    else:
        raise ValueError(f"unknown pooling_mode {self.pooling_mode}")
    # team_centered variant: subtract team mean from h before pooling (alternative constraint)
    if self.pooling_mode=="team_centered":
        # h is already used; for this mode we recompute pooled_pre as zero-mean pooled?
        # Instead we compute pooled_pre as team_mean of (h - team_mean) which is zero; so we need different.
        # For team_centered we actually want to pool residual after subtracting team centroid from h
        # That yields near zero mean pooling, not useful. So we instead keep team_mean but h was centered per team.
        # Let's implement: center h per team before pooling
        # This is done by modifying h before _team_pool: we did not. Let's patch here:
        pass
    # Actually handle team_centered separately
    if self.pooling_mode=="team_centered":
        # center h per team
        import torch as _t
        h_centered = h.clone()
        for team_slot in (0,1):
            mask=(team_slots==team_slot).float().unsqueeze(-1)
            mean = (h*mask).sum(dim=1,keepdim=True)/mask.sum(dim=1,keepdim=True).clamp_min(1.0)
            # only subtract where mask
            h_centered = h_centered - mean*mask
        pooled_pre=_team_pool(h_centered, team_slots)
    outputs["pooling_pre"]=pooled_pre
    outputs["pooled_embedding"]=encoder.project(pooled_pre)
    return outputs

geometry.FrozenGeometryModel.layer_values = patched_layer_values

CANDIDATES = [
    "team_mean",
    "relational_pairwise",
    "centered_team_mean",
    "centered_relational",
    "team_centered",
]

import json as _json
def load_metadata(root):
    return [_json.loads(l) for l in (root/"artifacts"/"phase1"/"canonical_samples.jsonl").read_text().splitlines() if l.strip()]

def phase_classes(root):
    manifest=_json.loads((root/"artifacts"/"phase1"/"point_mainline"/"model_manifest.json").read_text())
    rows=[r for r in manifest if r["model_id"]=="phase_gat_seed11"]
    return [str(v) for v in rows[0]["phase_classes"]]

def task_tensors(root, arrays, classes):
    import torch
    metadata=load_metadata(root)
    class_index={n:i for i,n in enumerate(classes)}
    labels=np.asarray([class_index.get(r.get("phase_label"),-1) for r in metadata],dtype=np.int64)
    train=np.asarray([r["split"]=="train" and lab>=0 for r,lab in zip(metadata,labels)],dtype=bool)
    dev=np.asarray([r["split"]=="dev" and lab>=0 for r,lab in zip(metadata,labels)],dtype=bool)
    heldout=np.asarray([r["split"]=="heldout" and lab>=0 for r,lab in zip(metadata,labels)],dtype=bool)
    positions=torch.as_tensor(np.asarray(arrays["positions"],dtype=np.float32))
    teams=torch.as_tensor(np.asarray(arrays["team_slots"],dtype=np.int64))
    adjacency=torch.as_tensor(np.asarray(arrays["adjacency"],dtype=np.float32))
    label_tensor=torch.as_tensor(labels)
    return positions,teams,adjacency,label_tensor,train,dev,heldout

def train_variant(root, arrays, classes, seed, pooling_mode, epochs=80, lr=1e-3):
    import torch
    base=geometry.load_frozen_model(root, f"phase_gat_seed{seed}")
    model=copy.deepcopy(base.model)
    model.train()
    positions,teams,adjacency,labels,train_mask,dev_mask,heldout_mask = task_tensors(root,arrays,classes)
    wrapper=geometry.FrozenGeometryModel(model_id=f"phase_gat_{pooling_mode}_seed{seed}", architecture="Phase-GAT", model=model, uses_adjacency=True, pooling_mode=pooling_mode)
    optimizer=torch.optim.Adam(model.parameters(), lr=float(lr))
    train_indices=torch.as_tensor(np.flatnonzero(train_mask),dtype=torch.long)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        z=wrapper.layer_values(positions, teams, adjacency)["pooled_embedding"]
        logits=model.head(z)
        loss=torch.nn.functional.cross_entropy(logits[train_indices], labels[train_indices])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        z=wrapper.layer_values(positions, teams, adjacency)["pooled_embedding"]
        logits=model.head(z)
        predicted=logits.argmax(dim=-1).cpu().numpy()
    from sklearn.metrics import f1_score
    metrics={"variant":pooling_mode,"seed":int(seed),"pooling_mode":pooling_mode,"epochs":int(epochs),"learning_rate":float(lr),"parameter_count":int(sum(p.numel() for p in model.parameters())),"train_label_count":int(train_mask.sum()),"dev_label_count":int(dev_mask.sum()),"heldout_label_count":int(heldout_mask.sum())}
    for split_name,mask in (("train",train_mask),("dev",dev_mask),("heldout",heldout_mask)):
        observed=labels.cpu().numpy()[mask]
        pred=predicted[mask]
        if len(observed):
            metrics[f"{split_name}_accuracy"]=float(np.mean(observed==pred))
            metrics[f"{split_name}_macro_f1"]=float(f1_score(observed,pred,average="macro",zero_division=0))
        else:
            metrics[f"{split_name}_accuracy"]=float("nan")
            metrics[f"{split_name}_macro_f1"]=float("nan")
    return wrapper, metrics

def read_prospective_arms(root):
    manifest=json.loads((root/"artifacts"/"phase3"/"support_geometry_prospective_v1"/"intervention_manifest.json").read_text())
    frames=[pd.read_parquet(root/str(shard["arms_path"])) for shard in manifest["shards"]]
    return pd.concat(frames, ignore_index=True)

def evaluate_responses(root, wrapper, arrays, arms):
    positions=np.asarray(arrays["positions"],dtype=np.float32)
    teams=np.asarray(arrays["team_slots"],dtype=np.int64)
    adjacency=np.asarray(arrays["adjacency"],dtype=np.float32)
    valid=arms[arms["valid"].astype(bool)].copy()
    deltas=np.stack([geometry.parse_delta(v) for v in valid["delta"]])
    sample_indices=valid["sample_index"].to_numpy(dtype=int)
    moved=positions[sample_indices]+deltas
    moved_teams=teams[sample_indices]
    moved_adjacency=adjacency[sample_indices]
    baseline=geometry._encode_in_chunks(wrapper, positions, teams, adjacency, batch_size=64)
    encoded=geometry._encode_in_chunks(wrapper, moved, moved_teams, moved_adjacency, batch_size=64)
    base_norm=geometry._l2_normalize_rows(baseline[sample_indices])
    moved_norm=geometry._l2_normalize_rows(encoded)
    distance=1.0-np.sum(base_norm*moved_norm,axis=1)
    response=valid.copy()
    response["model_id"]=wrapper.model_id
    response["architecture"]=wrapper.architecture
    response["model_seed"]=int(wrapper.model_id.rsplit("seed",1)[1])
    response["pooling_mode"]=wrapper.pooling_mode
    response["response_distance"]=distance.astype(np.float32)
    response["normalized_response"]=(distance/response["epsilon"].to_numpy(float)).astype(np.float32)
    arm_geometry=geometry.geometry_arm_rows(root,wrapper,arrays,response,["pooled_embedding"])
    pair_geometry=geometry.pair_rows_from_arms(arm_geometry)
    pair_geometry["pooling_mode"]=wrapper.pooling_mode
    return arm_geometry, pair_geometry

def context_metrics(wrappers, arrays, epsilons):
    positions=np.asarray(arrays["positions"],dtype=np.float32)
    teams=np.asarray(arrays["team_slots"],dtype=np.int64)
    adjacency=np.asarray(arrays["adjacency"],dtype=np.float32)
    rows=[]
    for wrapper in wrappers:
        baseline=geometry._encode_in_chunks(wrapper, positions, teams, adjacency, batch_size=64)
        values=[]
        for sample_index in range(len(positions)):
            rng=np.random.default_rng(20260901+sample_index)
            direction=rng.normal(size=2).astype(np.float32)
            direction/=np.linalg.norm(direction)
            for epsilon in epsilons:
                delta=np.tile(direction*float(epsilon)/np.sqrt(20.0),(20,1)).astype(np.float32)
                moved=positions[sample_index:sample_index+1]+delta[None,...]
                encoded=geometry._encode_in_chunks(wrapper, moved, teams[sample_index:sample_index+1], adjacency[sample_index:sample_index+1], batch_size=1)
                distance=1.0-float(np.dot(geometry._l2_normalize_rows(baseline[sample_index:sample_index+1])[0], geometry._l2_normalize_rows(encoded)[0]))
                values.append(distance/float(epsilon))
        rows.append({"model_id":wrapper.model_id,"seed":int(wrapper.model_id.rsplit("seed",1)[1]),"pooling_mode":wrapper.pooling_mode,"n_translations":len(values),"mean_normalized_context_response":float(np.mean(values)),"median_normalized_context_response":float(np.median(values))})
    return pd.DataFrame.from_records(rows)

def main():
    root=ROOT
    config_path=root/"configs"/"phase3_support_geometry_v1.yaml"
    config=geometry.load_config(root, config_path)
    arrays=geometry.load_arrays(root)
    classes=phase_classes(root)
    arms=read_prospective_arms(root)
    epsilons=[float(v) for v in config["geometry"]["p2_response_epsilons"]]
    all_results=[]
    for candidate in CANDIDATES:
        print(f"=== candidate {candidate} ===", flush=True)
        wrappers=[]
        task_records=[]
        pair_frames=[]
        for seed in [11,23,47]:
            wrapper, metrics = train_variant(root, arrays, classes, seed, candidate, epochs=80, lr=1e-3)
            wrappers.append(wrapper)
            task_records.append(metrics)
            _, pair = evaluate_responses(root, wrapper, arrays, arms)
            pair_frames.append(pair)
        pair_df=pd.concat(pair_frames, ignore_index=True)
        from scipy.stats import spearmanr
        spearman_records=[]
        for (mid,mode), grp in pair_df.groupby(["model_id","pooling_mode"]):
            rho=float(spearmanr(grp["observed_pair_effect"], grp["delta_q_full"]).statistic)
            spearman_records.append((mode, grp["model_id"], rho, len(grp)))
        # aggregate means per pooling
        for mode in [candidate]:
            grp=pair_df[pair_df["pooling_mode"]==mode]
            rho_all=float(spearmanr(grp["observed_pair_effect"], grp["delta_q_full"]).statistic)
            print(f"  geometry spearman pooled {rho_all:.4f}")
            for _,mid,rho,n in spearman_records:
                print(f"    {mid} spearman {rho:.4f} n={n}")
        ctx=context_metrics(wrappers, arrays, epsilons)
        print(ctx.to_string(index=False))
        df_task=pd.DataFrame.from_records(task_records)
        print(df_task[["seed","train_macro_f1","dev_macro_f1","heldout_macro_f1"]].to_string(index=False))
        # compute dev mean, context mean
        dev_mean=df_task["dev_macro_f1"].mean()
        ctx_mean=ctx["mean_normalized_context_response"].mean()
        geom_mean=np.mean([r for _,_,r,_ in spearman_records])
        print(f"  summary dev_macro_f1 mean {dev_mean:.6f} context mean {ctx_mean:.6f} geom mean {geom_mean:.6f}")
        all_results.append((candidate, dev_mean, ctx_mean, geom_mean, df_task, ctx, pair_df))
    # ranking lexicographic: dev high, context low, geom high
    ranked=sorted(all_results, key=lambda x: (-x[1], x[2], -x[3]))
    print("\n=== RANKING ===")
    for cand, dev,ctx,geom,_,_,_ in ranked:
        print(f"{cand}: dev {dev:.5f} ctx {ctx:.6f} geom {geom:.4f}")
    # save
    out=root/"artifacts"/"phase3"/"candidate_search_v1"
    out.mkdir(parents=True, exist_ok=True)
    for cand,dev,ctx,geom,df_task,ctx_df,pair_df in all_results:
        df_task.to_parquet(out/f"task_{cand}.parquet", index=False)
        ctx_df.to_parquet(out/f"context_{cand}.parquet", index=False)
    pd.DataFrame([{"candidate":c,"dev_macro_f1_mean":d,"context_mean":ctx,"geom_mean":g} for c,d,ctx,g,_,_,_ in all_results]).to_csv(out/"ranking.csv", index=False)
    print(f"saved to {out}")

if __name__=="__main__":
    main()
