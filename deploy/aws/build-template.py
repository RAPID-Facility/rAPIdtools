#!/usr/bin/env python3
"""
Embed bootstrap.sh into rapidtools-gui.yaml.

The CloudFormation template carries the instance bootstrap script inside the
EC2 user data so that launching the stack needs nothing but the one YAML file.
User data is limited to 16 KB, so the script is stored gzip-compressed and
base64-encoded between the ``RTBLOB`` markers. Run this after every edit of
``bootstrap.sh``::

    python deploy/aws/build-template.py          # rewrite the template in place
    python deploy/aws/build-template.py --check  # exit 1 if the template is stale

To read what a template will run::

    python deploy/aws/build-template.py --extract > bootstrap-from-template.sh

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / 'rapidtools-gui.yaml'
SCRIPT = HERE / 'bootstrap.sh'
BLOCK = re.compile(
    r'^(?P<indent> *)'
    r"(?P<head>base64 -d <<'RTBLOB' \| gunzip >/opt/rapidtools/bootstrap\.sh\n)"
    r'(?P<blob>(?: *[A-Za-z0-9+/=]+\n)*)'
    r'(?P=indent)RTBLOB\n',
    re.MULTILINE,
)
USER_DATA_LIMIT = 16384  # bytes, before the base64 encoding EC2 applies itself
TEMPLATE_LIMIT = 51200  # bytes, for `aws cloudformation deploy` without a bucket
HEADER_ALLOWANCE = 2600  # the settings header the template puts before the script


def encode(script: bytes) -> list[str]:
    """Return ``script`` gzip-compressed as base64 lines of 76 characters."""
    packed = gzip.compress(script, compresslevel=9, mtime=0)
    text = base64.b64encode(packed).decode()
    return [text[i : i + 76] for i in range(0, len(text), 76)]


def decode(lines: list[str]) -> bytes:
    """Inverse of :func:`encode`; an empty block decodes to ``b''``."""
    text = ''.join(line.strip() for line in lines)
    return gzip.decompress(base64.b64decode(text)) if text else b''


def main() -> int:
    """Rewrite, check or extract the embedded script; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='verify, change nothing')
    group.add_argument('--extract', action='store_true', help='print the script')
    args = parser.parse_args()

    template = TEMPLATE.read_text()
    match = BLOCK.search(template)
    if match is None:
        print(f'RTBLOB block not found in {TEMPLATE.name}', file=sys.stderr)
        return 2
    embedded = decode(match['blob'].splitlines())
    if args.extract:
        sys.stdout.write(embedded.decode())
        return 0

    # Compare the decoded script, not the compressed bytes: gzip output differs
    # between Python and zlib versions.
    script = SCRIPT.read_bytes()
    current = embedded == script
    if args.check:
        if not current:
            print(f'{TEMPLATE.name} is stale: run build-template.py', file=sys.stderr)
            return 1
        print(f'{TEMPLATE.name} embeds the current {SCRIPT.name}')
        return 0

    indent = match['indent']
    lines = match['blob'].splitlines() if current else encode(script)
    blob = ''.join(f'{indent}{line.strip()}\n' for line in lines)
    updated = (
        template[: match.start()]
        + f'{indent}{match["head"]}{blob}{indent}RTBLOB\n'
        + template[match.end() :]
    )
    user_data = sum(len(line.strip()) + 1 for line in lines) + HEADER_ALLOWANCE
    size = len(updated.encode())
    if user_data > USER_DATA_LIMIT:
        print(
            f'user data: about {user_data} bytes, limit {USER_DATA_LIMIT}',
            file=sys.stderr,
        )
        return 1
    if size > TEMPLATE_LIMIT:
        print(f'template: {size} bytes, limit {TEMPLATE_LIMIT}', file=sys.stderr)
        return 1
    if updated != template:
        TEMPLATE.write_text(updated)
    print(f'{TEMPLATE.name}: {size} bytes, user data about {user_data} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
