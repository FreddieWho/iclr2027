#!/usr/bin/env python3
"""Build a small, self-contained HTML overview from Phase 0 artifacts."""
from __future__ import annotations

import importlib.util
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "phase0"
OUT = ROOT / "reports" / "p0-overview.html"


def load_pipeline_module():
    path = ROOT / "scripts" / "run_phase0_pipeline.py"
    spec = importlib.util.spec_from_file_location("phase0_pipeline_for_overview", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def project_2d(values: np.ndarray, n_points: int = 320) -> list[list[float]]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T
    indices = np.linspace(0, len(points) - 1, min(n_points, len(points)), dtype=int)
    return np.round(points[indices], 4).tolist()


def image_data_uri(path: Path) -> str:
    """Embed a local raster artifact so the generated page has no file dependencies."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_data() -> dict[str, object]:
    required = [
        ARTIFACTS / "canonical_samples.npz",
        ARTIFACTS / "canonical_samples.jsonl",
        ARTIFACTS / "intervention_manifest.parquet",
        ARTIFACTS / "spectrum_sanity.csv",
        ARTIFACTS / "coordinate_embedding.npy",
        ARTIFACTS / "frozen_resnet18_embedding.npy",
        ARTIFACTS / "pipeline_summary.json",
        ARTIFACTS / "sample_grid.png",
        ARTIFACTS / "phase0_first_minimap.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 0 artifacts:\n" + "\n".join(missing))

    phase0 = load_pipeline_module()
    arrays = np.load(ARTIFACTS / "canonical_samples.npz")
    positions = arrays["positions"]
    adjacency = arrays["adjacency"]
    frames = arrays["frame"]
    teams = arrays["team_ids"]
    node_ids = arrays["node_ids"]
    interventions = pd.read_parquet(ARTIFACTS / "intervention_manifest.parquet")
    sanity = pd.read_csv(ARTIFACTS / "spectrum_sanity.csv")
    summary = json.loads((ARTIFACTS / "pipeline_summary.json").read_text(encoding="utf-8"))

    sample_indices = np.linspace(0, len(positions) - 1, 12, dtype=int)
    sample_view = [
        {"index": int(i), "frame": int(frames[i]), "positions": np.round(positions[i], 4).tolist()}
        for i in sample_indices
    ]

    vals, vecs = eigh(phase0.normalized_laplacian(adjacency[0]))
    ranges = phase0.band_ranges(len(vals))
    rng = np.random.default_rng(int(summary["seed"]))
    semantic_support = np.array([i for i, team in enumerate(teams) if team == teams[0]][:5], dtype=int)
    common = phase0.equal_energy(np.ones((len(teams), 1)) @ np.array([[1.0, 0.0]]))
    single = np.zeros((len(teams), 2), dtype=np.float64)
    single[0, 0] = 1.0
    semantic = phase0.support_intervention(len(teams), semantic_support, rng)
    random_support = rng.choice(len(teams), size=len(semantic_support), replace=False)
    random_delta = phase0.support_intervention(len(teams), random_support, rng)
    delta_by_kind: list[tuple[str, int | None, np.ndarray]] = [
        ("common", None, common),
        ("single", None, single),
        ("semantic_coalition", None, semantic),
        ("random_coalition", None, random_delta),
    ]
    for band_id, modes in enumerate(ranges):
        delta_by_kind.append((f"band_{band_id}", band_id, phase0.spectral_intervention(vecs, modes, rng)))

    intervention_view = []
    first_rows = interventions[interventions["sample_id"] == "sc_1886347_0000"].set_index("kind")
    for kind, band_id, delta in delta_by_kind:
        row = first_rows.loc[kind]
        intervention_view.append(
            {
                "kind": kind,
                "band_id": band_id,
                "delta": np.round(delta, 4).tolist(),
                "support": np.where(np.linalg.norm(delta, axis=1) > 1e-12)[0].tolist(),
                "energy": float(row["energy"]),
                "energy_error": float(row["energy_error"]),
                "rayleigh_quotient": float(row["rayleigh_quotient"]),
                "mode_purity": None if pd.isna(row["target_mode_purity"]) else float(row["target_mode_purity"]),
            }
        )

    energy_rows = []
    for kind, group in interventions.groupby("kind", sort=False):
        energy_rows.append(
            {
                "kind": str(kind),
                "count": int(len(group)),
                "mean_error": float(group["energy_error"].mean()),
                "max_error": float(group["energy_error"].max()),
            }
        )
    band_rows = []
    for band_id, group in sanity.groupby("band_id", sort=True):
        band_rows.append(
            {
                "band_id": int(band_id),
                "eigenvalue_min": float(group["eigenvalue_min"].mean()),
                "eigenvalue_max": float(group["eigenvalue_max"].mean()),
                "mean_purity": float(group["mode_purity"].mean()),
                "min_purity": float(group["mode_purity"].min()),
            }
        )

    edges = [[int(i), int(j)] for i in range(len(teams)) for j in range(i + 1, len(teams)) if adjacency[0, i, j] > 0]
    return {
        "summary": summary,
        "frames": {"count": int(len(positions)), "representative": sample_view},
        "nodes": {"count": int(len(node_ids)), "ids": node_ids.astype(int).tolist(), "teams": teams.astype(int).tolist()},
        "first_graph": {"edges": edges, "eigenvalues": np.round(vals, 4).tolist()},
        "interventions": intervention_view,
        "energy": energy_rows,
        "bands": band_rows,
        "embeddings": {
            "coordinate": project_2d(np.load(ARTIFACTS / "coordinate_embedding.npy")),
            "visual": project_2d(np.load(ARTIFACTS / "frozen_resnet18_embedding.npy")),
        },
        "embedded_images": {
            "sample_grid": image_data_uri(ARTIFACTS / "sample_grid.png"),
            "first_minimap": image_data_uri(ARTIFACTS / "phase0_first_minimap.png"),
        },
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>P0 Overview</title>
  <style>
    :root { color-scheme: light dark; --bg:#f7f7f4; --fg:#20211f; --muted:#686b64; --line:#d8d9d2; --surface:#ffffff; --green:#2d7a50; --blue:#3b6ea8; --orange:#c87932; --red:#bd5252; --purple:#8062a8; --yellow:#ad8a2f; }
    @media (prefers-color-scheme: dark) { :root { --bg:#171916; --fg:#eceee7; --muted:#a9ada3; --line:#3c4039; --surface:#222620; --green:#78c99a; --blue:#79a9e0; --orange:#e0a064; --red:#e08484; --purple:#b59cda; --yellow:#dfc56e; } }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { max-width:1100px; margin:0 auto; padding:28px 20px 44px; }
    h1,h2 { margin:0 0 8px; font-weight:500; letter-spacing:-.02em; }
    h1 { font-size:clamp(25px,4vw,38px); } h2 { font-size:20px; margin-top:28px; }
    p { margin:5px 0; color:var(--muted); }
    code { color:var(--fg); }
    .intro { display:flex; flex-wrap:wrap; justify-content:space-between; gap:10px 24px; border-bottom:1px solid var(--line); padding-bottom:18px; }
    .status { color:var(--green); font-weight:500; }
    .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:18px 0 8px; }
    .stat { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:13px 15px; }
    .stat-label { color:var(--muted); font-size:12px; } .stat-value { font-size:24px; margin-top:2px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; align-items:start; }
    .panel { min-width:0; } .panel h2 { margin-top:0; }
    .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:8px 0 10px; }
    label { color:var(--muted); } select,input { accent-color:var(--green); }
    select { color:var(--fg); background:var(--surface); border:1px solid var(--line); border-radius:7px; padding:5px 8px; }
    input[type=range] { min-width:180px; }
    svg { display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:10px; background:var(--surface); }
    .caption { font-size:12px; color:var(--muted); margin-top:6px; }
    .wide { grid-column:1 / -1; }
    .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .bar-list { display:grid; gap:7px; margin-top:8px; }
    .bar-line { display:grid; grid-template-columns:130px 1fr 110px; gap:8px; align-items:center; font-size:12px; }
    .bar-track { height:9px; background:var(--line); border-radius:5px; overflow:hidden; }
    .bar-fill { height:100%; background:var(--green); min-width:2px; }
    .embedding-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .image-strip { display:grid; grid-template-columns:1fr 220px; gap:18px; align-items:start; }
    .embedded-image { display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:10px; background:var(--surface); }
    .note { border-left:3px solid var(--orange); padding:7px 10px; color:var(--muted); margin-top:10px; }
    footer { margin-top:26px; padding-top:12px; border-top:1px solid var(--line); font-size:12px; color:var(--muted); }
    @media (max-width:700px) { main { padding:20px 12px 32px; } .stats,.grid,.chart-row,.embedding-grid,.image-strip { grid-template-columns:1fr; } .wide { grid-column:auto; } .bar-line { grid-template-columns:108px 1fr 92px; } }
  </style>
</head>
<body>
<main id="p0-overview">
  <section class="intro" aria-labelledby="title">
    <div><h1 id="title">P0：数据、干预与表示</h1><p>用一场真实足球比赛快速查看 Phase 0 做了什么。</p></div>
    <p class="status">基础 checkpoint：完成</p>
  </section>
  <section class="stats" aria-label="P0 summary">
    <div class="stat"><div class="stat-label">canonical samples</div><div class="stat-value" id="stat-samples">—</div><div class="stat-label">真实比赛帧</div></div>
    <div class="stat"><div class="stat-label">interventions</div><div class="stat-value" id="stat-interventions">—</div><div class="stat-label">每个样本 10 条</div></div>
    <div class="stat"><div class="stat-label">embedding</div><div class="stat-value">32D + 512D</div><div class="stat-label">坐标 autoencoder / 冻结 ResNet-18</div></div>
  </section>

  <section class="grid" aria-label="raw samples and interventions">
    <div class="panel">
      <h2>原始数据：阵型快照</h2>
      <div class="controls"><label for="sample-select">代表帧</label><select id="sample-select" aria-label="选择代表帧"></select><span id="sample-label" class="caption"></span></div>
      <svg id="raw-field" viewBox="0 0 620 360" role="img" aria-label="足球球员位置和关系图"></svg>
      <p class="caption">红/蓝代表两队；点是球员，线是第一帧的近邻关系图。坐标已按球场长宽归一化。</p>
    </div>
    <div class="panel">
      <h2>干预：谁和谁一起移动</h2>
      <div class="controls"><label for="intervention-select">类型</label><select id="intervention-select" aria-label="选择干预类型"></select><span id="intervention-label" class="caption"></span></div>
      <svg id="intervention-field" viewBox="0 0 620 360" role="img" aria-label="选定干预的球员位移箭头"></svg>
      <p class="caption">箭头是实际生成的坐标位移；所有类型的总 Frobenius 能量都归一到 1.0。</p>
    </div>
  </section>

  <section class="wide">
    <h2>已嵌入的阶段图像</h2>
    <div class="image-strip">
      <div><img id="sample-grid-image" class="embedded-image" alt="P0 样本网格图"><p class="caption">P0 生成的代表性原始场景网格。</p></div>
      <div><img id="first-minimap-image" class="embedded-image" alt="P0 第一帧 minimap"><p class="caption">第一帧 minimap。</p></div>
    </div>
  </section>

  <section class="wide">
    <h2>频段与能量检查</h2>
    <div class="chart-row">
      <div><p>6 个频段：20 个 Laplacian 模态按数量分桶。</p><div id="band-bars" class="bar-list" aria-label="频段纯度"></div></div>
      <div><p>干预能量误差：目标为 1.0。</p><div id="energy-bars" class="bar-list" aria-label="干预能量误差"></div></div>
    </div>
  </section>

  <section class="wide">
    <h2>两个 embedding 视角</h2>
    <div class="embedding-grid">
      <div><p>坐标 autoencoder：直接看 20×2 个坐标，输出 32D。</p><svg id="coord-embed" viewBox="0 0 620 280" role="img" aria-label="坐标 embedding 二维投影"></svg></div>
      <div><p>冻结 ResNet-18：看坐标渲染出的 minimap，输出 512D。</p><svg id="visual-embed" viewBox="0 0 620 280" role="img" aria-label="视觉 embedding 二维投影"></svg></div>
    </div>
    <div class="note">这两个 embedding 是同一批样本的并行观察，不是串联或联合训练。P0 还没有测量干预前后表示变化，因此这里不宣称已经发现科学现象。</div>
  </section>
  <footer>数据：SkillCorner match 1886347 · P0 artifacts/phase0 · 离线页面，无外部请求</footer>
</main>
<script id="p0-data" type="application/json">__P0_DATA__</script>
<script>
(function () {
  const root = document.getElementById('p0-overview');
  const data = JSON.parse(document.getElementById('p0-data').textContent);
  const $ = (id) => root.querySelector('#' + id);
  const colors = ['var(--red)', 'var(--blue)', 'var(--orange)', 'var(--purple)'];
  const kindLabels = {common:'全体共同移动', single:'单节点移动', semantic_coalition:'结构支持 5 节点', random_coalition:'随机支持 5 节点'};
  data.interventions.forEach((x) => { if (!kindLabels[x.kind]) kindLabels[x.kind] = '频段 ' + x.band_id; });
  $('#stat-samples').textContent = data.frames.count.toLocaleString();
  $('#stat-interventions').textContent = data.summary.n_interventions.toLocaleString();
  $('#sample-grid-image').src = data.embedded_images.sample_grid;
  $('#first-minimap-image').src = data.embedded_images.first_minimap;
  const sampleSelect = $('#sample-select');
  data.frames.representative.forEach((s, i) => { const option = document.createElement('option'); option.value = i; option.textContent = 'sample ' + s.index + ' · frame ' + s.frame; sampleSelect.appendChild(option); });
  const interventionSelect = $('#intervention-select');
  data.interventions.forEach((x, i) => { const option = document.createElement('option'); option.value = i; option.textContent = kindLabels[x.kind]; interventionSelect.appendChild(option); });
  function xy(p, width, height, pad) { return [pad + (p[0] + 1) * .5 * (width - 2 * pad), height - pad - (p[1] + 1) * .5 * (height - 2 * pad)]; }
  function fieldBase(svg, positions, withGraph) {
    const width=620, height=360, pad=22, parts=[];
    parts.push('<rect x="' + pad + '" y="' + pad + '" width="' + (width-2*pad) + '" height="' + (height-2*pad) + '" rx="7" fill="var(--green)" fill-opacity=".16" stroke="var(--line)"/>');
    parts.push('<line x1="310" y1="22" x2="310" y2="338" stroke="var(--line)"/><circle cx="310" cy="180" r="39" fill="none" stroke="var(--line)"/>');
    if (withGraph) data.first_graph.edges.forEach((e) => { const a=xy(positions[e[0]],width,height,pad), b=xy(positions[e[1]],width,height,pad); parts.push('<line x1="'+a[0].toFixed(1)+'" y1="'+a[1].toFixed(1)+'" x2="'+b[0].toFixed(1)+'" y2="'+b[1].toFixed(1)+'" stroke="var(--line)" stroke-opacity=".65"/>'); });
    return parts;
  }
  function renderRaw() {
    const s=data.frames.representative[Number(sampleSelect.value)], parts=fieldBase($('raw-field'),s.positions,s.index===0);
    s.positions.forEach((p,i)=>{const q=xy(p,620,360,22); parts.push('<circle cx="'+q[0].toFixed(1)+'" cy="'+q[1].toFixed(1)+'" r="7" fill="'+colors[i<10?0:1]+'" stroke="var(--surface)" stroke-width="2"><title>node '+i+' · team '+data.nodes.teams[i]+'</title></circle>');});
    $('raw-field').innerHTML=parts.join(''); $('#sample-label').textContent='第 '+(Number(sampleSelect.value)+1)+' / '+data.frames.representative.length+' 个代表样本';
  }
  function renderIntervention() {
    const s=data.frames.representative[0], x=data.interventions[Number(interventionSelect.value)], parts=fieldBase($('intervention-field'),s.positions,true);
    parts.push('<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--orange)"/></marker></defs>');
    s.positions.forEach((p,i)=>{const q=xy(p,620,360,22); parts.push('<circle cx="'+q[0].toFixed(1)+'" cy="'+q[1].toFixed(1)+'" r="7" fill="'+colors[i<10?0:1]+'" fill-opacity=".32" stroke="var(--surface)" stroke-width="2"/>');});
    x.delta.forEach((d,i)=>{const p=s.positions[i], a=xy(p,620,360,22), end=[p[0]+d[0]*.8,p[1]+d[1]*.8], b=xy(end,620,360,22); if (Math.hypot(d[0],d[1])>.00001) parts.push('<line x1="'+a[0].toFixed(1)+'" y1="'+a[1].toFixed(1)+'" x2="'+b[0].toFixed(1)+'" y2="'+b[1].toFixed(1)+'" stroke="var(--orange)" stroke-width="2" marker-end="url(#arrow)"/>');});
    $('intervention-field').innerHTML=parts.join(''); $('intervention-label').textContent='energy='+x.energy.toExponential(2)+' · error='+x.energy_error.toExponential(2)+(x.mode_purity==null?'':' · purity='+x.mode_purity.toFixed(4));
  }
  function renderBars() {
    const band=$('band-bars'), maxPurity=1; band.innerHTML=data.bands.map((x)=>'<div class="bar-line"><span>band '+x.band_id+'</span><span class="bar-track"><span class="bar-fill" style="width:'+(x.mean_purity/maxPurity*100).toFixed(3)+'%"></span></span><span>'+x.mean_purity.toFixed(6)+'</span></div>').join('');
    const energy=$('energy-bars'), maxError=Math.max(...data.energy.map((x)=>x.max_error)); energy.innerHTML=data.energy.map((x)=>'<div class="bar-line"><span>'+(kindLabels[x.kind])+'</span><span class="bar-track"><span class="bar-fill" style="width:'+Math.max(2,x.max_error/Math.max(maxError,1e-30)*100).toFixed(2)+'%"></span></span><span>'+x.max_error.toExponential(2)+'</span></div>').join('');
  }
  function renderScatter(id, points, color) { const parts=['<line x1="45" y1="245" x2="600" y2="245" stroke="var(--line)"/><line x1="45" y1="20" x2="45" y2="245" stroke="var(--line)"/>']; const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]), xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys); points.forEach((p)=>{const x=52+(p[0]-xmin)/Math.max(xmax-xmin,1e-9)*540, y=238-(p[1]-ymin)/Math.max(ymax-ymin,1e-9)*210; parts.push('<circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="3" fill="'+color+'" fill-opacity=".58"/>');}); $(id).innerHTML=parts.join(''); }
  sampleSelect.addEventListener('input',renderRaw); interventionSelect.addEventListener('change',renderIntervention); renderRaw(); renderIntervention(); renderBars(); renderScatter('coord-embed',data.embeddings.coordinate,'var(--blue)'); renderScatter('visual-embed',data.embeddings.visual,'var(--purple)');
}());
</script>
</body>
</html>
'''


def main() -> int:
    payload = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.replace("__P0_DATA__", payload), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
