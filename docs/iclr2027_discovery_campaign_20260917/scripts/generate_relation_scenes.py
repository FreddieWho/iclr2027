#!/usr/bin/env python3
"""Create procedural four-endpoint relation scenes; no neural evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.relations import make_relational_scenes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--min-margin', type=float, default=.02)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('--output must be a new directory')
    try:
        positions, labels, margins = make_relational_scenes(args.n, args.seed, args.min_margin)
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    ids = np.array([f'relation_{args.seed}_{i:06d}' for i in range(args.n)])
    np.savez_compressed(args.output/'scenes.npz', positions=positions, labels=labels,
                        oracle_margin=margins, parent_scene_id=ids)
    metadata = {
        'kind':'procedural_data_not_scientific_result', 'task':'proper_segment_crossing_AB_CD',
        'seed':args.seed, 'n_parent_scenes':args.n, 'class_counts':np.bincount(labels, minlength=2).tolist(),
        'min_margin':args.min_margin, 'roles':['A','B','C','D'],
        'all_variants_of_parent_must_share_split':True,
        'notes':['Label balancing is not a proof of all-cue balance.',
                 'No image renderer, pretrained model, accuracy, or metamer result is included.']}
    (args.output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Generated {args.n} labeled coordinate scenes at {args.output}; no model evaluated.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
