import argparse
import base64
import imghdr
import os
import sys
from pathlib import Path


def load_b64_data(b64: str) -> bytes:
    # Strip data URI prefix if present
    if b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except Exception:
            pass
    # Remove whitespace/newlines
    cleaned = "".join(b64.split())
    # Decode
    try:
        return base64.b64decode(cleaned, validate=False)
    except Exception as e:
        raise ValueError(f"Failed to decode base64: {e}")


def detect_ext(data: bytes) -> str:
    kind = imghdr.what(None, h=data)
    if not kind:
        return "bin"
    return "jpg" if kind == "jpeg" else kind


def main():
    parser = argparse.ArgumentParser(description="Decode a Base64 image and save to file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--b64", help="Base64 string (use @file to read from file)")
    group.add_argument("--from-file", dest="from_file", help="Path to a file containing the Base64 string; use - for stdin")
    parser.add_argument("--output", "-o", help="Output filepath (if omitted, auto-generate name + extension)")
    args = parser.parse_args()

    # Read base64 source
    if args.from_file:
        if args.from_file == "-":
            b64_text = sys.stdin.read()
        else:
            b64_text = Path(args.from_file).read_text(encoding="utf-8")
    else:
        if args.b64.startswith("@") and len(args.b64) > 1:
            b64_text = Path(args.b64[1:]).read_text(encoding="utf-8")
        else:
            b64_text = args.b64

    data = load_b64_data(b64_text)

    # Decide output path
    if args.output:
        out_path = Path(args.output)
    else:
        ext = detect_ext(data)
        base = "decoded_image"
        # avoid overwriting
        out_path = Path(f"{base}.{ext}")
        i = 1
        while out_path.exists():
            out_path = Path(f"{base}_{i}.{ext}")
            i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)

    abs_path = out_path.resolve()
    print(f"Saved: {abs_path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()

