"""
MaeBounme Lao-OCR — Minimal Standalone Python ONNX Demo (Apache-2.0)
Usage:
    pip install onnxruntime opencv-python numpy
    python examples/ocr_demo.py path/to/text_line.png
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def recognize_line(img_bgr: np.ndarray, sess: ort.InferenceSession, charset: list[str]) -> str:
    h, w = img_bgr.shape[:2]
    target_h = 48
    target_w = max(8, int(round(w * (target_h / max(1, h)))))
    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    inp = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, 3, 48, W)

    in_name = sess.get_inputs()[0].name
    logits = sess.run(None, {in_name: inp})[0]  # (1, T, C)
    ids = np.argmax(logits[0], axis=-1)

    chars = []
    prev = -1
    for idx in ids:
        idx = int(idx)
        if idx != 0 and idx != prev and idx < len(charset):
            chars.append(charset[idx])
        prev = idx
    return "".join(chars)


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/ocr_demo.py <line_image_path>")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    rec_path = root / "models" / "rec.onnx"
    dict_path = root / "models" / "dict.txt"

    charset = ["<blank>"] + [line.rstrip("\r\n") for line in dict_path.read_text(encoding="utf-8").splitlines()]
    sess = ort.InferenceSession(str(rec_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    img_bytes = np.fromfile(sys.argv[1], dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {sys.argv[1]}")

    text = recognize_line(img, sess, charset)
    print(text)


if __name__ == "__main__":
    main()
