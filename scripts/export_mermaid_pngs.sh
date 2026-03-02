#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_MD="${1:-$ROOT_DIR/WRITE_UP.md}"
OUT_DIR="${2:-$ROOT_DIR/mermaid_pngs}"
TMP_DIR="$OUT_DIR/.tmp"

CHROME_BIN_DEFAULT="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MERMAID_JS_CANDIDATES=(
  "$HOME/.vscode/extensions/shd101wyy.markdown-preview-enhanced-0.8.20/crossnote/dependencies/mermaid/mermaid.min.js"
  "$HOME/.vscode/extensions/ms-vscode.copilot-mermaid-diagram-0.0.3/dist/media/mermaid/mermaid.esm.min.mjs"
)

CHROME_BIN="${CHROME_BIN:-$CHROME_BIN_DEFAULT}"
WINDOW_SIZE="${WINDOW_SIZE:-2400,3000}"
VIRTUAL_TIME_BUDGET="${VIRTUAL_TIME_BUDGET:-15000}"
CROP_WHITESPACE="${CROP_WHITESPACE:-1}"
DEVICE_SCALE_FACTOR="${DEVICE_SCALE_FACTOR:-2}"

if [[ ! -f "$SRC_MD" ]]; then
  echo "Source markdown not found: $SRC_MD" >&2
  exit 1
fi

if [[ ! -x "$CHROME_BIN" ]]; then
  echo "Chrome binary not executable: $CHROME_BIN" >&2
  exit 1
fi

MERMAID_JS=""
for candidate in "${MERMAID_JS_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    MERMAID_JS="$candidate"
    break
  fi
done

if [[ -z "$MERMAID_JS" ]]; then
  echo "Could not find a local Mermaid JS bundle." >&2
  echo "Checked paths:" >&2
  printf '  - %s\n' "${MERMAID_JS_CANDIDATES[@]}" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$TMP_DIR"
rm -f "$OUT_DIR"/diagram_*.mmd "$OUT_DIR"/diagram_*.png "$TMP_DIR"/*.html

awk -v outdir="$OUT_DIR" '
  BEGIN { in_block=0; idx=0 }
  /^```mermaid[[:space:]]*$/ {
    in_block=1
    idx++
    start=NR
    file=sprintf("%s/diagram_%02d_line_%d.mmd", outdir, idx, start)
    next
  }
  in_block && /^```[[:space:]]*$/ {
    in_block=0
    close(file)
    next
  }
  in_block { print $0 >> file }
' "$SRC_MD"

MMD_FILES=("$OUT_DIR"/diagram_*.mmd)

if [[ ! -e "${MMD_FILES[0]}" ]]; then
  echo "No Mermaid blocks found in: $SRC_MD" >&2
  exit 1
fi

MERMAID_URI="file://$MERMAID_JS"

for mmd in "${MMD_FILES[@]}"; do
  base="$(basename "$mmd" .mmd)"
  png="$OUT_DIR/${base}.png"
  html="$TMP_DIR/${base}.html"

  diagram_json="$(node -e 'const fs=require("fs"); process.stdout.write(JSON.stringify(fs.readFileSync(process.argv[1], "utf8")));' "$mmd")"

  cat > "$html" <<EOF
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {
      margin: 0;
      background: #ffffff;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: "Helvetica Neue", Arial, sans-serif;
    }
    #wrap {
      padding: 24px;
      box-sizing: border-box;
      width: 100%;
      height: 100%;
    }
    #diagram {
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>
  <div id="wrap">
    <div id="diagram" class="mermaid"></div>
  </div>
  <script src="$MERMAID_URI"></script>
  <script>
    const source = $diagram_json;
    const el = document.getElementById("diagram");
    el.textContent = source;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "default"
    });
    mermaid.run({ nodes: [el] }).catch((err) => {
      document.body.innerHTML = "<pre>" + String(err) + "</pre>";
    });
  </script>
</body>
</html>
EOF

  "$CHROME_BIN" \
    --headless=new \
    --disable-gpu \
    --allow-file-access-from-files \
    --hide-scrollbars \
    --force-device-scale-factor="$DEVICE_SCALE_FACTOR" \
    --window-size="$WINDOW_SIZE" \
    --virtual-time-budget="$VIRTUAL_TIME_BUDGET" \
    --screenshot="$png" \
    "$html" >/dev/null 2>&1

  if [[ "$CROP_WHITESPACE" == "1" ]]; then
    python3 - "$png" <<'PY'
from PIL import Image, ImageChops
import sys

path = sys.argv[1]
img = Image.open(path).convert("RGB")
bg = Image.new("RGB", img.size, (255, 255, 255))
diff = ImageChops.difference(img, bg)
bbox = diff.getbbox()

if bbox:
    pad = 24
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.size[0], bbox[2] + pad)
    bottom = min(img.size[1], bbox[3] + pad)
    img.crop((left, top, right, bottom)).save(path)
PY
  fi

  if [[ ! -f "$png" ]]; then
    echo "Failed to render: $mmd" >&2
    exit 1
  fi

  echo "Rendered: $png"
done

echo "Done. PNG files are in: $OUT_DIR"
