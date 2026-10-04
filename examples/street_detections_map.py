"""
Build a self-contained HTML map of street-level detections and their crops.

Reads the GeoJSON written by ``MapillaryFeatureExtractor`` (one point per
object) and the folder of crops written by ``MapillaryObjectImageExtractor``,
and writes one HTML file next to the crops: every object is a marker on a
Bing aerial basemap (the same keyless tiles the GUI uses); clicking a marker
shows its attributes and all of its crops tiled, with a link to each source
image on Mapillary and a line from each camera to the object.

Run with::

    python examples/street_detections_map.py

or point it at another run::

    python examples/street_detections_map.py \\
        --geojson output/spokane_vehicles/spokane_vehicles.geojson \\
        --crops output/spokane_vehicles/crops \\
        --output output/spokane_vehicles/vehicles_map.html

Open the HTML file in a browser. The map tiles and the Leaflet library load
from the internet; the crops are referenced relative to the HTML file, so
keep it next to the ``crops`` folder (or pass ``--output`` inside the run's
folder, which is the default).
"""

# ruff: noqa: E501  (the embedded HTML template has long lines)
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_RUN = Path('output/spokane_vehicles')
MAPILLARY_IMAGE_URL = 'https://www.mapillary.com/app/?pKey={image_id}&focus=photo'

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
    --line: #d9dee4; --accent: #0b6e99; --tri: #1ea672; --single: #e8912d;
    --sel: #d62828; --route: #2b7bff;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14171b; --panel: #1d2127; --text: #e6e9ed; --muted: #9aa4af;
      --line: #313843; --accent: #5cb3dc; --tri: #3ccf8e; --single: #f2a94a;
      --sel: #ff5c5c; --route: #6ea8ff;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14171b; --panel: #1d2127; --text: #e6e9ed; --muted: #9aa4af;
    --line: #313843; --accent: #5cb3dc; --tri: #3ccf8e; --single: #f2a94a;
    --sel: #ff5c5c; --route: #6ea8ff;
  }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  #app { display: grid; grid-template-columns: 1fr minmax(320px, 440px); height: 100%; }
  #map { height: 100%; }
  #side { background: var(--panel); border-left: 1px solid var(--line); overflow: auto;
    padding: 14px 16px; box-sizing: border-box; }
  h1 { font-size: 16px; margin: 0 0 6px; }
  .muted { color: var(--muted); }
  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; margin: 10px 0 12px;
    padding-bottom: 12px; border-bottom: 1px solid var(--line); }
  .controls label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .controls input[type=number] { width: 64px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
  .tri { background: var(--tri); } .single { background: var(--single); }
  .route { background: var(--route); } .frame { background: #fff; border: 1px solid var(--route); }
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
    <div class="muted" id="summary"></div>
    <div class="controls">
      <label><input type="checkbox" id="f-tri" checked> <span class="swatch tri"></span> triangulated</label>
      <label><input type="checkbox" id="f-single" checked> <span class="swatch single"></span> single view</label>
      <label>min images <input type="number" id="f-images" value="1" min="1"></label>
      <label>max sigma (m) <input type="number" id="f-sigma" value="" min="0" step="0.1" placeholder="any"></label>
      <label><input type="checkbox" id="f-route" checked> <span class="swatch route"></span> collection route</label>
      <label><input type="checkbox" id="f-frames"> <span class="swatch frame"></span> camera positions</label>
    </div>
    <div id="detail" class="muted">Click a marker to see its attributes and crops.
      Use the left and right arrow keys to step through the vehicles in view.</div>
  </aside>
</div>
<script>
const DATA = __DATA__;
const CROPS_DIR = __CROPS_DIR__;
const ROUTES = __ROUTES__;   // [{seq, date, frames: [[lon, lat], ...]}] in capture order

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

// The collection route: one polyline per sequence (white casing under a
// coloured line), and the individual camera positions on demand.
const routeLayer = L.layerGroup().addTo(map);
const frameLayer = L.layerGroup();
const routeColour = getComputedStyle(document.documentElement).getPropertyValue('--route').trim();
for (const r of ROUTES) {
  const latlngs = r.frames.map(f => [f[1], f[0]]);
  if (latlngs.length < 2) continue;
  L.polyline(latlngs, {color: '#fff', weight: 6, opacity: 0.6, interactive: false}).addTo(routeLayer);
  L.polyline(latlngs, {color: routeColour, weight: 2.5, opacity: 0.95})
    .bindTooltip(`sequence ${r.seq} · ${r.frames.length} frames · ${r.date || ''}`, {sticky: true})
    .addTo(routeLayer);
}
const frameRenderer = L.canvas({padding: 0.5});
for (const r of ROUTES)
  for (const f of r.frames)
    L.circleMarker([f[1], f[0]], {renderer: frameRenderer, radius: 2.5, weight: 1, color: routeColour,
      fillColor: '#fff', fillOpacity: 1, interactive: false}).addTo(frameLayer);
document.getElementById('f-route').addEventListener('change', e =>
  e.target.checked ? routeLayer.addTo(map) : map.removeLayer(routeLayer));
document.getElementById('f-frames').addEventListener('change', e =>
  e.target.checked ? frameLayer.addTo(map) : map.removeLayer(frameLayer));

const renderer = L.canvas({padding: 0.5});
const markers = new Map();
let selected = null;
let lines = L.layerGroup().addTo(map);
const colour = v => v.loc === 'triangulated' ? getCSS('--tri') : getCSS('--single');
function getCSS(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

for (const v of DATA) {
  const m = L.circleMarker([v.lat, v.lon], {renderer, radius: 5, weight: 1, color: '#111',
    fillColor: colour(v), fillOpacity: 0.9});
  m.bindTooltip(`${v.id} (${v.loc}, ${v.n_images} img)`, {direction: 'top'});
  m.on('click', () => select(v.id));
  markers.set(v.id, m);
}
const group = L.featureGroup([...markers.values()]).addTo(map);
map.fitBounds(group.getBounds().pad(0.05));

function visibleIds() {
  const tri = document.getElementById('f-tri').checked;
  const single = document.getElementById('f-single').checked;
  const minImages = Number(document.getElementById('f-images').value || 1);
  const maxSigmaRaw = document.getElementById('f-sigma').value;
  const maxSigma = maxSigmaRaw === '' ? Infinity : Number(maxSigmaRaw);
  return DATA.filter(v =>
    (v.loc === 'triangulated' ? tri : single) && v.n_images >= minImages &&
    (v.sigma == null || v.sigma <= maxSigma)).map(v => v.id);
}
function applyFilters() {
  const keep = new Set(visibleIds());
  for (const [id, m] of markers) {
    if (keep.has(id)) { if (!group.hasLayer(m)) group.addLayer(m); }
    else if (group.hasLayer(m)) group.removeLayer(m);
  }
  const nFrames = ROUTES.reduce((n, r) => n + r.frames.length, 0);
  const nSeq = new Set(ROUTES.map(r => r.seq)).size;
  document.getElementById('summary').textContent =
    `${keep.size} of ${DATA.length} vehicles shown · route: ${nSeq} sequences, ${nFrames} frames`;
}
for (const id of ['f-tri', 'f-single', 'f-images', 'f-sigma'])
  document.getElementById(id).addEventListener('input', applyFilters);
applyFilters();

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function select(id) {
  const v = DATA.find(d => d.id === id);
  if (!v) return;
  if (selected && markers.has(selected)) markers.get(selected).setStyle({color: '#111', weight: 1, radius: 5});
  selected = id;
  const m = markers.get(id);
  m.setStyle({color: getCSS('--sel'), weight: 3, radius: 8}).bringToFront();
  lines.clearLayers();
  for (const c of v.cams) {
    L.polyline([[c[1], c[0]], [v.lat, v.lon]], {color: getCSS('--sel'), weight: 2, opacity: 0.8, dashArray: '4 4'}).addTo(lines);
    L.circleMarker([c[1], c[0]], {radius: 4, color: '#fff', weight: 1, fillColor: getCSS('--sel'), fillOpacity: 1}).addTo(lines);
  }
  const rows = [
    ['id', v.id], ['label', v.label], ['localization', v.loc],
    ['images / sightings', `${v.n_images} / ${v.n_obs}`],
    ['position sigma', v.sigma == null ? '' : `${v.sigma} m`],
    ['parallax', v.parallax == null ? '' : `${v.parallax}°`],
    ['closest range', v.min_range == null ? '' : `${v.min_range} m`],
    ['seen', [v.first_seen, v.last_seen].filter(Boolean).map(s => s.slice(0, 19).replace('T', ' ')).join(' to ')],
    ['position', `${v.lat.toFixed(6)}, ${v.lon.toFixed(6)}`],
  ];
  const table = rows.map(([k, val]) => `<tr><td>${esc(k)}</td><td>${esc(val)}</td></tr>`).join('');
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
const MAPILLARY_URL = __MAPILLARY_URL__;
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


def crops_by_asset(crops_dir: Path, asset_ids: list[str]) -> dict[str, list[dict]]:
    """Map asset IDs to their crop files; file names end in ``_<image_id>.<ext>``."""
    found: dict[str, list[dict]] = defaultdict(list)
    ids = sorted(asset_ids, key=len, reverse=True)  # longest first for prefix matching
    for path in sorted(crops_dir.iterdir()):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            continue
        stem = path.stem
        for asset_id in ids:
            marker = f'{asset_id}_'
            pos = stem.find(marker)
            if pos < 0:
                continue
            rest = stem[pos + len(marker) :]
            image_id = rest if re.fullmatch(r'\d+', rest) else None
            found[asset_id].append({'file': path.name, 'image_id': image_id})
            break
    return found


GAP_M = 60.0  # a jump longer than this between consecutive frames starts a new segment


def _segments(frames: list[tuple[str, str, float, float]]) -> list[list[list[float]]]:
    """Split one sequence's frames (sorted by time) at gaps of ``GAP_M``."""
    from rapidtools.processing.street_localization import haversine_m

    segments: list[list[list[float]]] = []
    for _, _, lon, lat in frames:
        if (
            segments
            and haversine_m(segments[-1][-1][0], segments[-1][-1][1], lon, lat) <= GAP_M
        ):
            segments[-1].append([round(lon, 6), round(lat, 6)])
        else:
            segments.append([[round(lon, 6), round(lat, 6)]])
    return segments


def routes_from_observations(features: list[dict]) -> list[dict]:
    """Camera positions of every sighting, grouped by sequence in capture order."""
    frames: dict[str, dict[str, tuple[str, str, float, float]]] = defaultdict(dict)
    for f in features:
        for o in f['properties'].get('observations') or []:
            if o.get('camera_lon') is None:
                continue
            seq = str(o.get('sequence_id') or 'unknown')
            frames[seq].setdefault(
                str(o['image_id']),
                (
                    str(o.get('captured_at') or ''),
                    str(o['image_id']),
                    o['camera_lon'],
                    o['camera_lat'],
                ),
            )
    return _routes(frames)


def routes_from_listing(features: list[dict], token: str) -> list[dict]:
    """Every frame Mapillary lists in the detections' extent (complete drive)."""
    from rapidtools import BoundingBox, MapillaryClient

    lons = [f['geometry']['coordinates'][0] for f in features]
    lats = [f['geometry']['coordinates'][1] for f in features]
    pad = 0.0008  # about 60-90 m
    bbox = BoundingBox(
        min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad
    )
    client = MapillaryClient(token, save_dir='.')
    frames: dict[str, dict[str, tuple[str, str, float, float]]] = defaultdict(dict)
    min_lon, min_lat, max_lon, max_lat = bbox.bounds
    for image in client.fetch_images_in_bbox(bbox, filter_rapid_only=True):
        p = image.properties
        # The listing covers whole coverage tiles (about 2 km); keep the box.
        if not (
            min_lon <= p['longitude'] <= max_lon and min_lat <= p['latitude'] <= max_lat
        ):
            continue
        seq = str(p.get('sequence') or 'unknown')
        when = str(p.get('captured_at') or p.get('capture_date') or '')
        frames[seq][str(image.id)] = (
            when,
            str(image.id),
            p['longitude'],
            p['latitude'],
        )
    return _routes(frames)


def _chain(
    frames: list[tuple[str, str, float, float]],
) -> list[tuple[str, str, float, float]]:
    """
    Put frames in driving order when they carry no time of capture.

    Starts at the frame farthest from the sequence's centre (an end of the
    drive) and repeatedly steps to the nearest unvisited frame; a survey
    sampled every few metres chains almost perfectly this way.
    """
    import math

    if len(frames) < 3:
        return frames
    lat0 = frames[0][3]
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    pts = [((f[2] - frames[0][2]) * kx, (f[3] - frames[0][3]) * ky) for f in frames]
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    current = max(
        range(len(pts)), key=lambda i: (pts[i][0] - cx) ** 2 + (pts[i][1] - cy) ** 2
    )
    remaining = set(range(len(pts))) - {current}
    order = [current]
    while remaining:
        x, y = pts[current]
        current = min(
            remaining, key=lambda i: (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2
        )
        remaining.remove(current)
        order.append(current)
    return [frames[i] for i in order]


def _routes(frames: dict[str, dict[str, tuple[str, str, float, float]]]) -> list[dict]:
    routes = []
    for seq, by_id in frames.items():
        values = list(by_id.values())
        # Full timestamps order the drive exactly; a bare date does not.
        if all(len(v[0]) > 10 for v in values):
            ordered = sorted(values)
        else:
            ordered = _chain(values)
        date = min((v[0][:10] for v in values if v[0]), default='')
        for segment in _segments(ordered):
            routes.append({'seq': seq, 'date': date, 'frames': segment})
    return routes


def build(
    geojson_path: Path,
    crops_dir: Path,
    output: Path,
    title: str,
    token: str | None = None,
) -> int:
    data = json.loads(geojson_path.read_text())
    features = data['features']
    routes = (
        routes_from_listing(features, token)
        if token
        else routes_from_observations(features)
    )
    ids = [str(f.get('id') or f['properties'].get('id')) for f in features]
    crops = crops_by_asset(crops_dir, ids)
    rows = []
    for f, asset_id in zip(features, ids, strict=True):
        p = f['properties']
        lon, lat = f['geometry']['coordinates'][:2]
        # Camera position per crop, from the sightings that produced it:
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
            else:
                cams.append(None)
        rows.append(
            {
                'id': asset_id,
                'lon': lon,
                'lat': lat,
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
                'cams': [c for c in cams if c],
            }
        )
    try:
        rel_crops = crops_dir.resolve().relative_to(output.resolve().parent).as_posix()
    except ValueError:
        rel_crops = crops_dir.resolve().as_uri()
    html = (
        TEMPLATE.replace('__TITLE__', title)
        .replace('__DATA__', json.dumps(rows, separators=(',', ':')))
        .replace('__CROPS_DIR__', json.dumps(rel_crops))
        .replace('__ROUTES__', json.dumps(routes, separators=(',', ':')))
        .replace('__MAPILLARY_URL__', json.dumps(MAPILLARY_IMAGE_URL))
    )
    output.write_text(html, encoding='utf-8')
    return len(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--geojson', type=Path, default=DEFAULT_RUN / 'spokane_vehicles.geojson'
    )
    parser.add_argument('--crops', type=Path, default=DEFAULT_RUN / 'crops')
    parser.add_argument(
        '--output', type=Path, default=DEFAULT_RUN / 'vehicles_map.html'
    )
    parser.add_argument('--title', default='Street-level vehicle detections')
    parser.add_argument(
        '--token',
        default=None,
        help='Mapillary access token (or a file holding it). With it the route is the '
        'complete drive listed by Mapillary; without it, the frames that saw a vehicle.',
    )
    args = parser.parse_args(argv)
    token = args.token
    if token and Path(token).is_file():
        token = Path(token).read_text().strip()
    n = build(args.geojson, args.crops, args.output, args.title, token=token)
    print(f'{n} objects written to {args.output}')


if __name__ == '__main__':
    main()
