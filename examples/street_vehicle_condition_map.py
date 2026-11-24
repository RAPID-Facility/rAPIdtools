"""
Classify street-level vehicle detections as burned or intact and map them.

A condition-aware version of ``street_detections_map.py``. It reads the
GeoJSON written by ``MapillaryFeatureExtractor`` (one point per vehicle) and
the crops written by ``MapillaryObjectImageExtractor``, shows a vision-language
model the crops of every vehicle and asks whether it is burned or intact, then
writes two files next to the crops:

* a GeoJSON with the verdict (``vehicle_condition``) and the model's one-line
  reason (``vehicle_evidence``) added to every vehicle;
* a self-contained HTML map with burned vehicles in red, intact vehicles in
  green and the few the model could not judge in gray. Clicking a marker
  shows the verdict, the evidence and the crops it was judged from.

Run with::

    python examples/street_vehicle_condition_map.py

The classifier defaults to Gemini and needs a Google API key, read from
``--api-key``, ``GOOGLE_API_KEY`` or ``api_key.txt``. Any provider known to
``rapidtools.models.load`` works, for instance a local model::

    python examples/street_vehicle_condition_map.py --model gemma4

To redraw the map from a run that was already classified, without calling the
model again::

    python examples/street_vehicle_condition_map.py --no-classify

Open the HTML file in a browser. The map tiles and the Leaflet library load
from the internet; the crops are referenced relative to the HTML file, so
keep it next to the ``crops`` folder.
"""

# ruff: noqa: E501  (the embedded HTML template has long lines)
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from street_detections_map import (  # noqa: E402
    DEFAULT_RUN,
    MAPILLARY_IMAGE_URL,
    crops_by_asset,
    routes_from_listing,
    routes_from_observations,
)

CONDITIONS = ('burned', 'intact', 'unclear')
ATTRIBUTE_PREFIX = 'vehicle'  # attributes become vehicle_condition, vehicle_evidence

CONDITION_PROMPT = (
    'These crops come from street-view panoramas taken after a wildfire. They '
    'all show the same vehicle, outlined in red. Decide whether that vehicle is '
    'burned or intact. Return a single JSON object (not a list) for the vehicle '
    'as a whole, with two keys: "condition", exactly one of burned, '
    'intact or unclear (burned: any fire or heat damage, from scorched or '
    'blistered paint and melted parts to a burned-out shell or bare frame; '
    'intact: no fire damage, even if dirty, dented or old; unclear: the '
    'vehicle is too small, too hidden or too blurred to judge); and '
    '"evidence", one short sentence naming what you saw.'
)

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root {
    --bg: #ffffff; --panel: #f6f7f9; --text: #1b1f24; --muted: #5c6670;
    --line: #d9dee4; --accent: #0b6e99; --burned: #d62828; --intact: #1ea672;
    --unclear: #8a949e; --sel: #ffd400; --route: #2b7bff;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14171b; --panel: #1d2127; --text: #e6e9ed; --muted: #9aa4af;
      --line: #313843; --accent: #5cb3dc; --burned: #ff5c5c; --intact: #3ccf8e;
      --unclear: #a3adb8; --sel: #ffe14d; --route: #6ea8ff;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14171b; --panel: #1d2127; --text: #e6e9ed; --muted: #9aa4af;
    --line: #313843; --accent: #5cb3dc; --burned: #ff5c5c; --intact: #3ccf8e;
    --unclear: #a3adb8; --sel: #ffe14d; --route: #6ea8ff;
  }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  #app { display: grid; grid-template-columns: 1fr minmax(320px, 440px); height: 100%; }
  #map { height: 100%; }
  #side { background: var(--panel); border-left: 1px solid var(--line); overflow: auto;
    padding: 14px 16px; box-sizing: border-box; }
  h1 { font-size: 16px; margin: 0 0 6px; }
  .muted { color: var(--muted); }
  .counts { display: flex; gap: 14px; margin: 8px 0 2px; font-size: 13px; }
  .counts b { font-size: 18px; font-weight: 600; margin-right: 4px; }
  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; margin: 10px 0 12px;
    padding-bottom: 12px; border-bottom: 1px solid var(--line); }
  .controls label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .controls input[type=number] { width: 64px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
  .burned { background: var(--burned); } .intact { background: var(--intact); }
  .unclear { background: var(--unclear); }
  .route { background: var(--route); } .frame { background: #fff; border: 1px solid var(--route); }
  .verdict { display: inline-block; padding: 2px 8px; border-radius: 10px; color: #fff;
    font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
  .verdict.burned { background: var(--burned); } .verdict.intact { background: var(--intact); }
  .verdict.unclear { background: var(--unclear); }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  td { padding: 2px 4px; vertical-align: top; border-bottom: 1px solid var(--line); }
  td:first-child { color: var(--muted); white-space: nowrap; width: 38%; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px;
    margin-top: 10px; }
  .grid figure { margin: 0; }
  .grid img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 4px;
    border: 1px solid var(--line); display: block; background: #000; }
  .grid figcaption { font-size: 11px; color: var(--muted); margin-top: 3px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  a { color: var(--accent); }
  .nav { display: flex; gap: 8px; margin: 8px 0; }
  button { background: var(--bg); color: var(--text); border: 1px solid var(--line);
    border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  .leaflet-container { background: #222; }
  @media (max-width: 760px) {
    #app { grid-template-columns: 1fr; grid-template-rows: 55% 45%; }
    #side { border-left: 0; border-top: 1px solid var(--line); padding: 12px 16px; }
  }
</style>
</head>
<body>
<div id="app">
  <div id="map"></div>
  <aside id="side">
    <h1>__TITLE__</h1>
    <div class="counts" id="counts"></div>
    <div class="muted" id="summary"></div>
    <div class="controls">
      <label><input type="checkbox" id="f-burned" checked> <span class="swatch burned"></span> burned</label>
      <label><input type="checkbox" id="f-intact" checked> <span class="swatch intact"></span> intact</label>
      <label><input type="checkbox" id="f-unclear" checked> <span class="swatch unclear"></span> unclear</label>
      <label>min images <input type="number" id="f-images" value="1" min="1"></label>
      <label>max sigma (m) <input type="number" id="f-sigma" value="" min="0" step="0.1" placeholder="any"></label>
      <label><input type="checkbox" id="f-route" checked> <span class="swatch route"></span> collection route</label>
      <label><input type="checkbox" id="f-frames"> <span class="swatch frame"></span> camera positions</label>
    </div>
    <div id="detail" class="muted">Click a marker to see its condition, the evidence and the crops it was judged from.
      Use the left and right arrow keys to step through the vehicles in view.</div>
  </aside>
</div>
<script>
const DATA = __DATA__;
const CROPS_DIR = __CROPS_DIR__;
const ROUTES = __ROUTES__;   // [{seq, date, frames: [[lon, lat], ...]}] in capture order
const MAPILLARY_URL = __MAPILLARY_URL__;
const MODEL = __MODEL__;

// Bing aerial tiles, addressed by quadkey (the same keyless endpoint rapidtools uses).
const BingAerial = L.TileLayer.extend({
  getTileUrl(coords) {
    let q = '';
    for (let i = coords.z; i > 0; i--) {
      let d = 0; const m = 1 << (i - 1);
      if ((coords.x & m) !== 0) d += 1;
      if ((coords.y & m) !== 0) d += 2;
      q += d;
    }
    return `https://ecn.t3.tiles.virtualearth.net/tiles/a${q}.jpeg?g=1`;
  },
});
const bing = new BingAerial('', {maxZoom: 20, attribution: 'Imagery &copy; Microsoft Bing Maps'});
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'});
const map = L.map('map', {layers: [bing], preferCanvas: true});
L.control.layers({'Bing aerial': bing, 'OpenStreetMap': osm}).addTo(map);
L.control.scale({imperial: true}).addTo(map);

function getCSS(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// The collection route: one polyline per sequence (white casing under a
// colored line), and the individual camera positions on demand.
const routeLayer = L.layerGroup().addTo(map);
const frameLayer = L.layerGroup();
const routeColor = getCSS('--route');
for (const r of ROUTES) {
  const latlngs = r.frames.map(f => [f[1], f[0]]);
  if (latlngs.length < 2) continue;
  L.polyline(latlngs, {color: '#fff', weight: 6, opacity: 0.6, interactive: false}).addTo(routeLayer);
  L.polyline(latlngs, {color: routeColor, weight: 2.5, opacity: 0.95})
    .bindTooltip(`sequence ${r.seq} · ${r.frames.length} frames · ${r.date || ''}`, {sticky: true})
    .addTo(routeLayer);
}
const frameRenderer = L.canvas({padding: 0.5});
for (const r of ROUTES)
  for (const f of r.frames)
    L.circleMarker([f[1], f[0]], {renderer: frameRenderer, radius: 2.5, weight: 1, color: routeColor,
      fillColor: '#fff', fillOpacity: 1, interactive: false}).addTo(frameLayer);
document.getElementById('f-route').addEventListener('change', e =>
  e.target.checked ? routeLayer.addTo(map) : map.removeLayer(routeLayer));
document.getElementById('f-frames').addEventListener('change', e =>
  e.target.checked ? frameLayer.addTo(map) : map.removeLayer(frameLayer));

// Burned vehicles are red, intact ones green, and the few the model could
// not judge gray. Burned markers are drawn last and slightly larger so they
// stay visible among the intact majority.
const renderer = L.canvas({padding: 0.5});
const markers = new Map();
let selected = null;
let lines = L.layerGroup().addTo(map);
const color = v => getCSS('--' + v.condition);
const radius = v => v.condition === 'burned' ? 6 : 5;
const order = v => v.condition === 'burned' ? 2 : v.condition === 'unclear' ? 0 : 1;

for (const v of [...DATA].sort((a, b) => order(a) - order(b))) {
  const m = L.circleMarker([v.lat, v.lon], {renderer, radius: radius(v), weight: 1, color: '#111',
    fillColor: color(v), fillOpacity: 0.9});
  m.bindTooltip(`${v.id} · ${v.condition} (${v.n_images} img)`, {direction: 'top'});
  m.on('click', () => select(v.id));
  markers.set(v.id, m);
}
const group = L.featureGroup([...markers.values()]).addTo(map);
map.fitBounds(group.getBounds().pad(0.05));

function visibleIds() {
  const show = {
    burned: document.getElementById('f-burned').checked,
    intact: document.getElementById('f-intact').checked,
    unclear: document.getElementById('f-unclear').checked,
  };
  const minImages = Number(document.getElementById('f-images').value || 1);
  const maxSigmaRaw = document.getElementById('f-sigma').value;
  const maxSigma = maxSigmaRaw === '' ? Infinity : Number(maxSigmaRaw);
  return DATA.filter(v =>
    show[v.condition] && v.n_images >= minImages &&
    (v.sigma == null || v.sigma <= maxSigma)).map(v => v.id);
}
function applyFilters() {
  const keep = new Set(visibleIds());
  for (const [id, m] of markers) {
    if (keep.has(id)) { if (!group.hasLayer(m)) group.addLayer(m); }
    else if (group.hasLayer(m)) group.removeLayer(m);
  }
  const counts = {burned: 0, intact: 0, unclear: 0};
  for (const v of DATA) if (keep.has(v.id)) counts[v.condition] += 1;
  document.getElementById('counts').innerHTML = ['burned', 'intact', 'unclear']
    .map(c => `<span><b style="color: var(--${c})">${counts[c]}</b>${c}</span>`).join('');
  const nFrames = ROUTES.reduce((n, r) => n + r.frames.length, 0);
  const nSeq = new Set(ROUTES.map(r => r.seq)).size;
  document.getElementById('summary').textContent =
    `${keep.size} of ${DATA.length} vehicles shown · judged by ${MODEL || 'a vision-language model'} · ` +
    `route: ${nSeq} sequences, ${nFrames} frames`;
}
for (const id of ['f-burned', 'f-intact', 'f-unclear', 'f-images', 'f-sigma'])
  document.getElementById(id).addEventListener('input', applyFilters);
applyFilters();

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function select(id) {
  const v = DATA.find(d => d.id === id);
  if (!v) return;
  if (selected && markers.has(selected)) {
    const prev = DATA.find(d => d.id === selected);
    markers.get(selected).setStyle({color: '#111', weight: 1, radius: radius(prev)});
  }
  selected = id;
  const m = markers.get(id);
  m.setStyle({color: getCSS('--sel'), weight: 3, radius: 9}).bringToFront();
  lines.clearLayers();
  for (const c of v.cams) {
    L.polyline([[c[1], c[0]], [v.lat, v.lon]], {color: color(v), weight: 2, opacity: 0.9, dashArray: '4 4'}).addTo(lines);
    L.circleMarker([c[1], c[0]], {radius: 4, color: '#fff', weight: 1, fillColor: color(v), fillOpacity: 1}).addTo(lines);
  }
  const rows = [
    ['condition', `<span class="verdict ${esc(v.condition)}">${esc(v.condition)}</span>`],
    ['evidence', esc(v.evidence || '')],
    ['id', esc(v.id)], ['label', esc(v.label)], ['localization', esc(v.loc)],
    ['images / sightings', `${v.n_images} / ${v.n_obs}`],
    ['position sigma', v.sigma == null ? '' : `${v.sigma} m`],
    ['parallax', v.parallax == null ? '' : `${v.parallax}°`],
    ['closest range', v.min_range == null ? '' : `${v.min_range} m`],
    ['seen', esc([v.first_seen, v.last_seen].filter(Boolean).map(s => s.slice(0, 19).replace('T', ' ')).join(' to '))],
    ['position', `${v.lat.toFixed(6)}, ${v.lon.toFixed(6)}`],
  ];
  const table = rows.map(([k, val]) => `<tr><td>${esc(k)}</td><td>${val}</td></tr>`).join('');
  const figs = v.crops.map((c, i) => {
    const imageId = c.image_id;
    const link = imageId ? `<a href="${MAPILLARY_URL.replace('{image_id}', imageId)}" target="_blank" rel="noopener">image ${esc(imageId)}</a>` : esc(c.file);
    const cam = v.cams[i] ? ` · ${v.cams[i][2].toFixed(1)} m` : '';
    return `<figure><a href="${esc(CROPS_DIR + '/' + c.file)}" target="_blank" rel="noopener">` +
      `<img loading="lazy" src="${esc(CROPS_DIR + '/' + c.file)}" alt="${esc(v.id)} crop ${i + 1}"></a>` +
      `<figcaption title="${esc(c.file)}">${link}${cam}</figcaption></figure>`;
  }).join('');
  document.getElementById('detail').className = '';
  document.getElementById('detail').innerHTML =
    `<div class="nav"><button id="prev">&larr; previous</button><button id="next">next &rarr;</button>` +
    `<button id="zoom">zoom to</button></div><table>${table}</table>` +
    `<div class="grid">${figs || '<span class="muted">No crops for this object.</span>'}</div>`;
  document.getElementById('prev').onclick = () => step(-1);
  document.getElementById('next').onclick = () => step(1);
  document.getElementById('zoom').onclick = () => map.setView([v.lat, v.lon], Math.max(map.getZoom(), 19));
  if (!map.getBounds().contains([v.lat, v.lon])) map.panTo([v.lat, v.lon]);
  history.replaceState(null, '', `#id=${encodeURIComponent(id)}`);
}
function step(delta) {
  const ids = visibleIds();
  if (!ids.length) return;
  let i = ids.indexOf(selected);
  i = i < 0 ? 0 : (i + delta + ids.length) % ids.length;
  select(ids[i]);
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') step(1);
  if (e.key === 'ArrowLeft') step(-1);
});
const wanted = new URLSearchParams(location.hash.slice(1)).get('id');
if (wanted && markers.has(wanted)) {
  const v = DATA.find(d => d.id === wanted);
  map.setView([v.lat, v.lon], 19);
  select(wanted);
}
</script>
</body>
</html>
"""


def _seen_in(features: list[dict]) -> dict[str, set[str]]:
    """Per asset, the IDs of the images it was sighted in (to skip stale crops)."""
    return {
        str(f.get('id') or f['properties'].get('id')): {
            str(o.get('image_id')) for o in f['properties'].get('observations') or []
        }
        for f in features
    }


def verdict_from_properties(p: dict) -> tuple[str, str]:
    """
    The condition and evidence recorded on an asset, whatever shape it took.

    The analyzer writes ``vehicle_condition`` and ``vehicle_evidence`` when
    the model returns one JSON object. Now and then a model answers with a
    list instead, one verdict per crop, which the analyzer keeps verbatim in
    ``vehicle_analysis_raw``; the per-crop verdicts are combined here, and a
    vehicle burned in any view counts as burned.
    """
    condition = p.get(f'{ATTRIBUTE_PREFIX}_condition')
    evidence = p.get(f'{ATTRIBUTE_PREFIX}_evidence') or ''
    if condition and normalize_condition(condition) != 'unclear' or evidence:
        return normalize_condition(condition), str(evidence)
    raw = p.get(f'{ATTRIBUTE_PREFIX}_analysis_raw')
    if not raw:
        return normalize_condition(condition), str(evidence)
    text = str(raw).strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return 'unclear', str(evidence)
    items = parsed if isinstance(parsed, list) else [parsed]
    items = [i for i in items if isinstance(i, dict)]
    if not items:
        return 'unclear', str(evidence)
    verdicts = [normalize_condition(i.get('condition')) for i in items]
    reasons = [str(i.get('evidence') or '').strip() for i in items]
    if 'burned' in verdicts:
        combined = 'burned'
    elif 'intact' in verdicts:
        combined = 'intact'
    else:
        combined = 'unclear'
    return combined, '; '.join(r for r in reasons if r)


def normalize_condition(value: object) -> str:
    """Map whatever the model wrote to one of ``CONDITIONS``."""
    text = str(value or '').strip().lower()
    for condition in CONDITIONS:
        if text.startswith(condition):
            return condition
    if any(w in text for w in ('burn', 'char', 'scorch', 'melt', 'fire', 'debris')):
        return 'burned'
    if any(w in text for w in ('undamaged', 'fine', 'normal', 'no damage')):
        return 'intact'
    return 'unclear'


def classify(
    geojson_path: Path,
    crops_dir: Path,
    output: Path,
    provider: str,
    model_id: str | None,
    api_key: str,
    max_workers: int,
    max_images: int,
    prompt: str = CONDITION_PROMPT,
) -> Counter:
    """Judge every vehicle from its crops and write the classified GeoJSON."""
    from rapidtools import PhysicalAssetCollection, configure_logging
    from rapidtools.core.image_asset import ImageAsset
    from rapidtools.models import GenerationConfig, load
    from rapidtools.processing.image_analyzers import AssetAnalyzer

    configure_logging()
    data = json.loads(geojson_path.read_text())
    features = data['features']
    seen_in = _seen_in(features)
    crops = crops_by_asset(crops_dir, list(seen_in), seen_in)

    collection = PhysicalAssetCollection.from_geojson(data)
    for asset in collection:
        asset.add_image_assets(
            [
                ImageAsset(path=crops_dir / c['file'], id=c['image_id'] or c['file'])
                for c in crops.get(asset.id, [])
            ]
        )

    kwargs = {'model_id': model_id} if model_id else {}
    if api_key:
        kwargs['api_key'] = api_key
    model = load(provider, **kwargs)
    analyzer = AssetAnalyzer(
        model,
        prompt=prompt,
        attribute_prefix=ATTRIBUTE_PREFIX,
        max_images_per_asset=max_images,
        max_workers=max_workers,
        generation=GenerationConfig(json_mode=True, temperature=0.0),
    )
    analyzer(collection)

    counts: Counter = Counter()
    for asset in collection:
        if not asset.image_assets:
            condition, evidence = 'unclear', 'no crops to judge from'
        else:
            condition, evidence = verdict_from_properties(asset.attributes)
        asset.attributes[f'{ATTRIBUTE_PREFIX}_condition'] = condition
        asset.attributes[f'{ATTRIBUTE_PREFIX}_evidence'] = evidence
        counts[condition] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    collection.to_geojson(output, ignore_properties=['image_assets'])
    return counts


def build(
    geojson_path: Path,
    crops_dir: Path,
    output: Path,
    title: str,
    token: str | None = None,
) -> Counter:
    """Write the HTML map from a classified GeoJSON; returns the condition counts."""
    data = json.loads(geojson_path.read_text())
    features = data['features']
    routes = (
        routes_from_listing(features, token)
        if token
        else routes_from_observations(features)
    )
    seen_in = _seen_in(features)
    crops = crops_by_asset(crops_dir, list(seen_in), seen_in)
    rows = []
    counts: Counter = Counter()
    models: Counter = Counter()
    for f in features:
        p = f['properties']
        asset_id = str(f.get('id') or p.get('id'))
        lon, lat = f['geometry']['coordinates'][:2]
        by_image = {}
        for o in p.get('observations') or []:
            by_image.setdefault(str(o.get('image_id')), o)
        cams = []
        for c in crops.get(asset_id, []):
            o = by_image.get(c['image_id'] or '')
            if o and o.get('camera_lon') is not None:
                cams.append(
                    [o['camera_lon'], o['camera_lat'], float(o.get('range_m') or 0.0)]
                )
        condition, evidence = verdict_from_properties(p)
        counts[condition] += 1
        if p.get('ai_model_used'):
            models[str(p['ai_model_used'])] += 1
        rows.append(
            {
                'id': asset_id,
                'lon': lon,
                'lat': lat,
                'condition': condition,
                'evidence': evidence,
                'loc': p.get('localization', ''),
                'label': p.get('label', ''),
                'n_images': p.get('n_images', 0),
                'n_obs': p.get('n_observations', 0),
                'sigma': p.get('position_sigma_m'),
                'parallax': p.get('parallax_deg'),
                'min_range': p.get('min_range_m'),
                'first_seen': p.get('first_seen'),
                'last_seen': p.get('last_seen'),
                'crops': crops.get(asset_id, []),
                'cams': cams,
            }
        )
    try:
        rel_crops = crops_dir.resolve().relative_to(output.resolve().parent).as_posix()
    except ValueError:
        rel_crops = crops_dir.resolve().as_uri()
    model_name = models.most_common(1)[0][0] if models else ''
    html = (
        TEMPLATE.replace('__TITLE__', title)
        .replace('__DATA__', json.dumps(rows, separators=(',', ':')))
        .replace('__CROPS_DIR__', json.dumps(rel_crops))
        .replace('__ROUTES__', json.dumps(routes, separators=(',', ':')))
        .replace('__MAPILLARY_URL__', json.dumps(MAPILLARY_IMAGE_URL))
        .replace('__MODEL__', json.dumps(model_name))
    )
    output.write_text(html, encoding='utf-8')
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--geojson',
        type=Path,
        default=DEFAULT_RUN / 'spokane_vehicles.geojson',
        help='Detections to classify (output of vehicle_detection_from_street.py).',
    )
    parser.add_argument('--crops', type=Path, default=DEFAULT_RUN / 'crops')
    parser.add_argument(
        '--classified',
        type=Path,
        default=DEFAULT_RUN / 'spokane_vehicles_condition.geojson',
        help='Where the GeoJSON with the burned / intact verdicts is written '
        '(or read from, with --no-classify).',
    )
    parser.add_argument(
        '--output', type=Path, default=DEFAULT_RUN / 'vehicle_condition_map.html'
    )
    parser.add_argument('--title', default='Street-level vehicle condition')
    parser.add_argument(
        '--no-classify',
        action='store_true',
        help='Skip the model and draw the map from the --classified file as it is.',
    )
    parser.add_argument(
        '--model',
        default='gemini',
        help='Provider for rapidtools.models.load, e.g. gemini, claude, gemma4, '
        'qwen_vl (default: gemini).',
    )
    parser.add_argument(
        '--model-id', default=None, help="The provider's model, e.g. gemini-3.8-flash."
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='API key for hosted providers (or a file holding it). Falls back to '
        'GOOGLE_API_KEY and api_key.txt.',
    )
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument(
        '--max-images',
        type=int,
        default=2,
        help='Crops shown to the model per vehicle (default: 2).',
    )
    parser.add_argument(
        '--token',
        default=None,
        help='Mapillary access token (or a file holding it). With it the route is the '
        'complete drive listed by Mapillary; without it, the frames that saw a vehicle.',
    )
    args = parser.parse_args(argv)

    if not args.no_classify:
        api_key = args.api_key or ''
        if api_key and Path(api_key).is_file():
            api_key = Path(api_key).read_text().strip()
        if not api_key:
            api_key = os.environ.get('GOOGLE_API_KEY', '')
        if not api_key and Path('api_key.txt').is_file():
            api_key = Path('api_key.txt').read_text().strip()
        model_id = args.model_id
        if args.model == 'gemini' and model_id is None:
            model_id = 'gemini-3.8-flash'  # the lite tier under-calls fire damage
        counts = classify(
            args.geojson,
            args.crops,
            args.classified,
            args.model,
            model_id,
            api_key,
            args.max_workers,
            args.max_images,
        )
        print(f'Classified {sum(counts.values())} vehicles: {dict(counts)}')
        print(f'Written to {args.classified}')
    elif not args.classified.is_file():
        parser.error(
            f'--no-classify needs an existing classified file: {args.classified}'
        )

    token = args.token
    if token and Path(token).is_file():
        token = Path(token).read_text().strip()
    counts = build(args.classified, args.crops, args.output, args.title, token=token)
    print(f'{sum(counts.values())} vehicles mapped ({dict(counts)}) to {args.output}')


if __name__ == '__main__':
    main()
