import base64
import copy
import io
import json
import math
import zipfile

import numpy as np
import streamlit as st
from opensimplex import OpenSimplex
from PIL import Image

# streamlit-drawable-canvas 0.9.3 calls streamlit.elements.image.image_to_url,
# which was removed in Streamlit 1.37+. Patch it back before importing the canvas.
import streamlit.elements.image as _ste
if not hasattr(_ste, "image_to_url"):
    def _image_to_url(img, width, clamp, channels, output_format, image_id):
        buf = io.BytesIO()
        img.save(buf, format=output_format)
        mime = "image/png" if output_format.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
    _ste.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas

ZOOM = 6  # display scale for the tile canvas

# ── .piskel parsing ──────────────────────────────────────────────────────────

def load_piskel(file_bytes: bytes) -> dict:
    return json.loads(file_bytes.decode("utf-8"))

def decode_layer_image(layer_json_str: str) -> np.ndarray:
    layer = json.loads(layer_json_str)
    b64 = layer["chunks"][0]["base64PNG"]
    raw = base64.b64decode(b64.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.array(img, dtype=np.uint8)

def encode_layer_image(layer_json_str: str, arr: np.ndarray) -> str:
    layer = json.loads(layer_json_str)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    layer["chunks"][0]["base64PNG"] = b64
    return json.dumps(layer)

def build_piskel_bytes(original: dict, new_layer0_arr: np.ndarray) -> bytes:
    out = copy.deepcopy(original)
    layers = out["piskel"]["layers"]
    layers[0] = encode_layer_image(layers[0], new_layer0_arr)
    return json.dumps(out).encode("utf-8")

# ── palette extraction ───────────────────────────────────────────────────────

def extract_palette(arr: np.ndarray):
    pixels = arr.reshape(-1, 4)
    opaque = pixels[pixels[:, 3] > 0]
    if len(opaque) == 0:
        return np.empty((0, 4), dtype=np.uint8), np.empty(0)
    palette, counts = np.unique(opaque, axis=0, return_counts=True)
    proportions = counts / counts.sum()
    sort_idx = np.argsort(-counts)
    return palette[sort_idx].astype(np.uint8), proportions[sort_idx]

# ── lock mask ────────────────────────────────────────────────────────────────

def build_lock_mask(H: int, W: int, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    if top > 0:
        mask[:top, :] = True
    if bottom > 0:
        mask[H - bottom:, :] = True
    if left > 0:
        mask[:, :left] = True
    if right > 0:
        mask[:, W - right:] = True
    return mask

def apply_canvas_locks(
    mask: np.ndarray,
    canvas_objects: list,
    zoom_factor: int,
    H: int,
    W: int,
) -> np.ndarray:
    for obj in canvas_objects:
        if obj.get("type") != "rect":
            continue
        sx = obj.get("scaleX", 1.0)
        sy = obj.get("scaleY", 1.0)
        x0 = max(0, int(obj["left"] / zoom_factor))
        y0 = max(0, int(obj["top"] / zoom_factor))
        x1 = min(W, math.ceil((obj["left"] + obj["width"] * sx) / zoom_factor))
        y1 = min(H, math.ceil((obj["top"] + obj["height"] * sy) / zoom_factor))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask

def render_lock_preview(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[mask, 0] = np.clip(out[mask, 0].astype(np.int32) // 2 + 127, 0, 255).astype(np.uint8)
    out[mask, 1] = (out[mask, 1] // 2).astype(np.uint8)
    out[mask, 2] = (out[mask, 2] // 2).astype(np.uint8)
    return out

# ── variation generation ─────────────────────────────────────────────────────

def generate_variation(
    original: np.ndarray,
    palette: np.ndarray,
    proportions: np.ndarray,
    lock_mask: np.ndarray,
    noise_scale: float,
    drift: float,
    seed: int,
) -> np.ndarray:
    H, W = original.shape[:2]
    gen = OpenSimplex(seed=seed)

    uniform = np.ones(len(palette)) / len(palette)
    weights = proportions * (1.0 - drift) + uniform * drift
    thresholds = np.cumsum(weights)
    thresholds[-1] = 1.0

    freq = noise_scale / max(H, W)
    noise = np.array(
        [[gen.noise2(x * freq, y * freq) for x in range(W)] for y in range(H)],
        dtype=np.float32,
    )
    lo, hi = noise.min(), noise.max()
    if hi > lo:
        noise = (noise - lo) / (hi - lo)
    else:
        noise[:] = 0.5

    indices = np.searchsorted(thresholds, noise)
    indices = np.clip(indices, 0, len(palette) - 1)
    out = palette[indices]
    out[lock_mask] = original[lock_mask]
    return out.astype(np.uint8)

# ── sprite sheet ─────────────────────────────────────────────────────────────

def build_sprite_sheet(images: list[np.ndarray]) -> Image.Image:
    if not images:
        return Image.new("RGBA", (1, 1))
    H, W = images[0].shape[:2]
    cols = min(len(images), 8)
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGBA", (cols * W, rows * H))
    for i, arr in enumerate(images):
        sheet.paste(Image.fromarray(arr, "RGBA"), ((i % cols) * W, (i // cols) * H))
    return sheet

# ── helpers ───────────────────────────────────────────────────────────────────

def zoom(arr: np.ndarray, factor: int = ZOOM) -> Image.Image:
    H, W = arr.shape[:2]
    return Image.fromarray(arr, "RGBA").resize((W * factor, H * factor), Image.NEAREST)

def palette_html(palette: np.ndarray, proportions: np.ndarray) -> str:
    swatches = []
    for color, pct in zip(palette, proportions):
        r, g, b, a = color
        hex_c = f"#{r:02x}{g:02x}{b:02x}"
        swatches.append(
            f'<span title="{hex_c} ({pct*100:.1f}%)" style="'
            f"display:inline-block;width:20px;height:20px;"
            f"background:{hex_c};margin:2px;border:1px solid #333;"
            f'opacity:{a/255:.2f}"></span>'
        )
    return "".join(swatches)

# ── main UI ───────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Piskel Tile Variator", layout="wide")
st.title("Piskel Tile Variator")
st.caption("Upload a .piskel file → extract palette → generate noise-based variations with locked edges.")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

uploaded = st.file_uploader("Drop a .piskel file here", type=["piskel"])

if uploaded is None:
    st.info("Upload a `.piskel` file to get started.")
    st.stop()

file_bytes = uploaded.read()
try:
    piskel_data = load_piskel(file_bytes)
    layer0_arr = decode_layer_image(piskel_data["piskel"]["layers"][0])
except Exception as e:
    st.error(f"Failed to parse .piskel: {e}")
    st.stop()

H = piskel_data["piskel"]["height"]
W = piskel_data["piskel"]["width"]
num_layers = len(piskel_data["piskel"]["layers"])
palette, proportions = extract_palette(layer0_arr)

# ── tile preview + interactive lock drawing ───────────────────────────────────

st.subheader("Lock Region")
st.caption("Drag rectangles on the tile to lock additional pixel areas. Red tint on the right shows the full protected region.")

draw_col, preview_col, pal_col = st.columns([2, 2, 3])

with draw_col:
    st.markdown("**Draw to lock** (drag rectangles)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=1,
        stroke_color="rgba(220, 30, 30, 0.9)",
        background_image=zoom(layer0_arr, ZOOM),
        height=H * ZOOM,
        width=W * ZOOM,
        drawing_mode="rect",
        key=f"lock_canvas_{st.session_state.canvas_key}_{uploaded.name}",
    )
    if st.button("Clear drawn locks"):
        st.session_state.canvas_key += 1
        st.rerun()

with preview_col:
    st.markdown("**Lock preview** (red = protected)")
    # build a preliminary mask from edge sliders for live preview
    # sliders are defined below — read from session state with defaults
    _top    = st.session_state.get("lock_top", 0)
    _bottom = st.session_state.get("lock_bottom", 0)
    _left   = st.session_state.get("lock_left", 0)
    _right  = st.session_state.get("lock_right", 0)
    preview_mask = build_lock_mask(H, W, _top, _bottom, _left, _right)
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        preview_mask = apply_canvas_locks(
            preview_mask, canvas_result.json_data["objects"], ZOOM, H, W
        )
    st.image(zoom(render_lock_preview(layer0_arr, preview_mask), ZOOM), use_container_width=False)
    locked_count = int(preview_mask.sum())
    st.caption(f"{locked_count} / {H*W} pixels locked ({locked_count*100//(H*W)}%)")

with pal_col:
    st.markdown(f"**Palette — {len(palette)} colors** (sorted by frequency)")
    st.markdown(palette_html(palette, proportions), unsafe_allow_html=True)
    st.caption(f"{W}×{H} px · {num_layers} layer(s) · hover swatches for hex + %")

if len(palette) == 0:
    st.warning("No opaque pixels found in layer 0.")
    st.stop()

# ── slider guide ──────────────────────────────────────────────────────────────

with st.expander("How do the controls work?"):
    st.markdown("""
**Noise scale (1–10)** — spatial frequency of the simplex noise field that drives color placement.
- **1** → large smooth blobs; only a few broad color regions per tile. Good for coarse ground types.
- **5** → medium patches; balanced organic look (recommended starting point).
- **10** → fine, grainy texture; many tiny color flecks across the tile.
*Analogy: low = zoomed-in landscape photo, high = zoomed-out satellite view.*

---

**Color drift % (0–100)** — how far the color distribution drifts from the original tile's proportions.
- **0%** → strict match: if the original is ~60% dark-brown the variation will be ~60% dark-brown too. Keeps the overall tone faithful.
- **100%** → uniform: all palette colors appear in equal amounts regardless of original frequency. Pushes toward higher contrast.
- **20–40%** → sweet spot for natural-looking variety without losing the original feel.

---

**Base seed** — starting random seed; variation *N* uses `seed + N`.
- Same seed + same settings = identical output every time (reproducible).
- Jump the seed by 100+ to get completely unrelated patterns at the same settings.
- Increment by 1 to step through the variation sequence.

---

**Variations (4 / 8 / 16 / 32)** — how many tiles to generate per batch. All appear in the preview grid and both download formats.

---

**Lock top / bottom / left / right (0–3)** — rows or columns to copy verbatim from the original along each edge. Use these for tiles whose edges must match a neighboring tile exactly. Combined with drawn rectangles, the total protected region is shown in the red overlay above.
""")

st.divider()

# ── settings ──────────────────────────────────────────────────────────────────

st.subheader("Settings")

ec1, ec2, ec3, ec4 = st.columns(4)
lock_top    = ec1.slider("Lock top rows",    0, 3, 0, key="lock_top")
lock_bottom = ec2.slider("Lock bottom rows", 0, 3, 0, key="lock_bottom")
lock_left   = ec3.slider("Lock left cols",   0, 3, 0, key="lock_left")
lock_right  = ec4.slider("Lock right cols",  0, 3, 0, key="lock_right")

sc1, sc2, sc3, sc4 = st.columns(4)
noise_scale = sc1.slider("Noise scale", 1, 10, 4, help="1 = large blobs, 10 = fine grain")
drift_pct   = sc2.slider("Color drift %", 0, 100, 20, help="0 = match original proportions, 100 = uniform spread")
num_vars    = sc3.selectbox("Variations", [4, 8, 16, 32], index=1)
base_seed   = sc4.number_input("Base seed", min_value=0, max_value=99999, value=42, step=1)

st.divider()

# ── generate ──────────────────────────────────────────────────────────────────

if st.button("Generate variations", type="primary"):
    lock_mask = build_lock_mask(H, W, lock_top, lock_bottom, lock_left, lock_right)
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        lock_mask = apply_canvas_locks(
            lock_mask, canvas_result.json_data["objects"], ZOOM, H, W
        )
    drift = drift_pct / 100.0

    variation_arrays: list[np.ndarray] = []
    progress = st.progress(0, text="Generating…")

    for i in range(num_vars):
        arr = generate_variation(
            original=layer0_arr,
            palette=palette,
            proportions=proportions,
            lock_mask=lock_mask,
            noise_scale=noise_scale,
            drift=drift,
            seed=int(base_seed) + i,
        )
        variation_arrays.append(arr)
        progress.progress((i + 1) / num_vars, text=f"Generating {i+1}/{num_vars}…")

    progress.empty()
    st.success(f"Generated {num_vars} variations.")

    # preview grid
    st.subheader("Preview")
    preview_cols = min(num_vars, 8)
    grid_rows = math.ceil(num_vars / preview_cols)
    for row in range(grid_rows):
        cols = st.columns(preview_cols)
        for col_i in range(preview_cols):
            idx = row * preview_cols + col_i
            if idx < num_vars:
                cols[col_i].image(
                    zoom(variation_arrays[idx], 4),
                    caption=f"#{idx+1}",
                    use_container_width=False,
                )

    st.divider()
    st.subheader("Download")

    dl1, dl2 = st.columns(2)

    zip_buf = io.BytesIO()
    base_name = uploaded.name.replace(".piskel", "")
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, arr in enumerate(variation_arrays):
            piskel_bytes = build_piskel_bytes(piskel_data, arr)
            zf.writestr(f"{base_name}_var{i+1:02d}.piskel", piskel_bytes)
    zip_buf.seek(0)

    dl1.download_button(
        label=f"Download {num_vars} .piskel files (zip)",
        data=zip_buf,
        file_name=f"{base_name}_variations.zip",
        mime="application/zip",
    )

    sheet = build_sprite_sheet(variation_arrays)
    sheet_buf = io.BytesIO()
    sheet.save(sheet_buf, format="PNG")
    sheet_buf.seek(0)

    dl2.download_button(
        label="Download sprite sheet PNG",
        data=sheet_buf,
        file_name=f"{base_name}_sheet.png",
        mime="image/png",
    )

    st.image(sheet.resize((sheet.width * 3, sheet.height * 3), Image.NEAREST), caption="Sprite sheet (3× zoom)")
