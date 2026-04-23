#!/usr/bin/env python3
"""Extract FLIR ADAS v2 split zip files and convert to YOLO format."""

import struct, zlib, os, json, shutil
from pathlib import Path
from tqdm import tqdm

SPLIT_SIZE = 1073741824  # 1GB (.zip.00 size)
OUT_DIR = Path('/home/bisavunma/yolo_project/flir_dataset')
FILE00 = 'FLIR_ADAS_v2.zip.00'
FILE01 = 'FLIR_ADAS_v2.zip.01'

class SplitZipReader:
    """Reads across two split zip files."""
    def __init__(self, f0, f1, split_at):
        self.f0 = f0
        self.f1 = f1
        self.split_at = split_at
        self._pos = 0

    def seek(self, pos):
        self._pos = pos

    def tell(self):
        return self._pos

    def read(self, n):
        result = b''
        remaining = n
        pos = self._pos

        while remaining > 0:
            if pos < self.split_at:
                to_read = min(remaining, self.split_at - pos)
                self.f0.seek(pos)
                chunk = self.f0.read(to_read)
                result += chunk
                pos += len(chunk)
                remaining -= len(chunk)
                if len(chunk) < to_read:
                    break
            else:
                offset_in_f1 = pos - self.split_at
                self.f1.seek(offset_in_f1)
                chunk = self.f1.read(remaining)
                result += chunk
                pos += len(chunk)
                remaining -= len(chunk)
                break

        self.seek(pos)
        return result


def parse_local_header(reader, pos):
    """Parse a zip local file header at pos. Returns (filename, data_start, comp_size, uncomp_size, method, flags)."""
    reader.seek(pos)
    header = reader.read(30)
    if len(header) < 30 or header[:4] != b'PK\x03\x04':
        return None
    sig, ver, flags, method, mod_time, mod_date, crc, comp_size, uncomp_size, fn_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
    filename = reader.read(fn_len).decode('utf-8', errors='replace')
    # skip extra
    data_start = pos + 30 + fn_len + extra_len
    return filename, data_start, comp_size, uncomp_size, method, flags, crc


def extract_file(reader, data_start, comp_size, method, flags, uncomp_size):
    """Extract and decompress file data."""
    reader.seek(data_start)
    data = reader.read(comp_size)
    if len(data) < comp_size:
        return None  # truncated
    if method == 0:
        return data
    elif method == 8:
        try:
            return zlib.decompress(data, -15)
        except zlib.error:
            return None
    return None


def scan_all_entries(reader, total_size):
    """Scan all local file headers across the split archive."""
    entries = []
    pos = 0
    print("Scanning archive entries...")

    while pos < total_size - 30:
        reader.seek(pos)
        sig = reader.read(4)
        if sig != b'PK\x03\x04':
            break

        reader.seek(pos)
        header = reader.read(30)
        sig, ver, flags, method, mod_time, mod_date, crc, comp_size, uncomp_size, fn_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
        filename = reader.read(fn_len).decode('utf-8', errors='replace')
        data_start = pos + 30 + fn_len + extra_len

        entries.append({
            'filename': filename,
            'data_start': data_start,
            'comp_size': comp_size,
            'uncomp_size': uncomp_size,
            'method': method,
            'flags': flags,
            'crc': crc,
        })

        pos = data_start + comp_size

        if len(entries) % 500 == 0:
            print(f"  Found {len(entries)} entries so far (pos={pos})...")

    return entries


def main():
    f0 = open(FILE00, 'rb')
    f1 = open(FILE01, 'rb')

    f0.seek(0, 2); size0 = f0.tell()
    f1.seek(0, 2); size1 = f1.tell()
    total = size0 + size1
    print(f"Total archive size: {total/1e9:.2f} GB")

    reader = SplitZipReader(f0, f1, size0)

    entries = scan_all_entries(reader, total)
    print(f"\nTotal entries found: {len(entries)}")

    # Show categories
    categories = {}
    for e in entries:
        fn = e['filename']
        if '/' in fn:
            cat = fn.split('/')[0]
        else:
            cat = 'root'
        categories[cat] = categories.get(cat, 0) + 1

    print("Categories:")
    for k, v in sorted(categories.items()):
        print(f"  {k}: {v}")

    # Extract annotation JSONs first
    print("\n--- Extracting annotation files ---")
    os.makedirs(OUT_DIR / 'annotations', exist_ok=True)

    json_files = {}
    for e in entries:
        fn = e['filename']
        if fn.endswith('.json') and 'coco' in fn.lower():
            print(f"Extracting: {fn}")
            # Check if we have the full data
            end_pos = e['data_start'] + e['comp_size']
            if end_pos <= total:
                data = extract_file(reader, e['data_start'], e['comp_size'], e['method'], e['flags'], e['uncomp_size'])
                if data:
                    out_path = OUT_DIR / 'annotations' / fn.replace('/', '_')
                    out_path.write_bytes(data)
                    print(f"  -> {out_path} ({len(data)/1e6:.1f} MB)")
                    json_files[fn] = out_path
                else:
                    print(f"  -> FAILED to decompress")
            else:
                print(f"  -> TRUNCATED (need {end_pos}, have {total})")

    # Extract images
    print("\n--- Extracting images ---")

    image_entries = [e for e in entries if e['filename'].endswith('.jpg') and
                     e['data_start'] + e['comp_size'] <= total]

    print(f"Extractable image files: {len(image_entries)}")

    for e in tqdm(image_entries, desc="Extracting images"):
        fn = e['filename']
        # fn like: images_rgb_train/data/filename.jpg
        parts = fn.split('/')
        if len(parts) >= 3:
            subset = parts[0]  # e.g. images_rgb_train
            img_name = parts[-1]
            out_dir = OUT_DIR / 'images' / subset
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / img_name

            if not out_path.exists():
                data = extract_file(reader, e['data_start'], e['comp_size'], e['method'], e['flags'], e['uncomp_size'])
                if data:
                    out_path.write_bytes(data)

    f0.close()
    f1.close()

    print("\nDone! Extracted files summary:")
    for subset_dir in sorted((OUT_DIR / 'images').iterdir()):
        count = len(list(subset_dir.glob('*.jpg')))
        print(f"  {subset_dir.name}: {count} images")

    return json_files


if __name__ == '__main__':
    main()
