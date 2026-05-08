# Reed-Solomon and QR Toolkit

A small Python toolkit for experimenting with Reed-Solomon coding, QR decoding, QR generation, and visual error correction. The project now follows a clearer package layout so the math core, QR utilities, GUI, tests, and benchmarks are separated into focused folders.

## What is inside

- Reed-Solomon field arithmetic and polynomial encoding/decoding.
- QR code extraction, unmasking, block healing, and text recovery.
- A CustomTkinter GUI for interactive encoding and QR workflows.
- Benchmark scripts comparing the custom implementation with third-party libraries.
- Scratch files and examples kept out of the main code path.

## Project layout

| Path | Purpose |
| --- | --- |
| `core/` | Core Reed-Solomon math, Galois field helpers, primitive tables, and the decoder. |
| `qr/` | QR-specific helpers for detection, unmasking, block healing, and QR generation. |
| `utils/` | Shared data transformation and error simulation helpers. |
| `gui/` | Desktop UI for encoding, QR generation, and QR recovery. |
| `benchmarks/` | Performance comparison scripts. |
| `tests/` | Pytest-based regression tests. |
| `scratch/` | Experimental and example scripts that are not part of the main workflow. |
| `images/` | Generated benchmark graphs and QR previews, except `tab1_qr.png` which stays at the repo root for the GUI. |
| `models/` | OpenCV WeChat QR detector and super-resolution model files. |
| `test_data/` | Sample images used by the QR pipeline. |

## Requirements

The project targets Python 3.11+ and uses the packages listed in `requirements.txt`.

Some features are optional:

- `opencv-contrib-python` is needed for `cv2.wechat_qrcode_WeChatQRCode`.
- `customtkinter` is only needed for the GUI.
- `reedsolo`, `unireedsolomon`, `bchlib`, and `galois` are used by the benchmark script.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you only want the algorithm core and tests, you can install a smaller subset of the dependencies, but the full requirements file is the easiest path for the complete project.

## Usage

### Start the GUI

```bash
python -m gui.rs_interface
```

### Run the test suite

```bash
pytest
```

### Run the benchmark script

```bash
python -m benchmarks.compare
```

### Use the core modules from Python

```python
from core.reed_solomon import ReedSolomon
from core.decode import DecodeReedSolomon

rs = ReedSolomon(8, 223)
for index, value in enumerate(b"Hello"):
    rs[index] = value
rs.encode()

fixed = DecodeReedSolomon(rs).decode()
fixed.get_original()
```

## Notes

- The codebase was reorganized into packages, so imports now use `core`, `qr`, `utils`, `gui`, `tests`, and `benchmarks`.
- The QR detector expects the model files in `models/`.
- Scratch scripts are intentionally kept separate from the reusable modules.

## Development hints

- `core.reed_solomon.ReedSolomon` owns the polynomial arithmetic and encoding flow.
- `core.decode.DecodeReedSolomon` owns the generic Reed-Solomon correction pipeline.
- `qr.qr_decode` contains the QR-specific decoding pipeline and image preprocessing.
- `gui.rs_interface.RSApp` wires the pieces together into a desktop workflow.
