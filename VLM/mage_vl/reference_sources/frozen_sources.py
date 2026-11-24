"""Read the archived source audit and preserve signatures for exact path-only migrations."""
import hashlib
import json
from pathlib import Path

SOURCES = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def protocol_script_hash(path):
    actual = digest(path)
    entry = json.loads((SOURCES / 'relocations.json').read_text()).get(path.name)
    return entry[1] if entry and actual == entry[0] else actual

def verified_audit():
    rows = json.loads((SOURCES / 'audit.json').read_text())
    for row in rows:
        files = dict(row['file_sha256'])
        if row['license_sha256']:
            files['LICENSE'] = row['license_sha256']
        for name, expected in files.items():
            if digest(SOURCES / row['name'] / name) != expected:
                raise ValueError(f"Reference source mismatch: {name}")
    return rows
