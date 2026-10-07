#!/usr/bin/env python3
"""Prepare analysis workflows with user-specified data and output paths."""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil

TOKEN = re.compile(r'__HISEMBLE_([A-Z_]+)__')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for configured analysis scripts and result tables')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / 'workflows'
    output = args.output.resolve()
    if output.exists():
        parser.error('Output directory must not exist; choose a new directory.')
    if source == output or source in output.parents:
        parser.error('Output must be outside workflows.')
    config = json.loads(args.config.read_text())
    # Values are inserted into Python/R/shell literals: restrict them to portable,
    # absolute POSIX paths to avoid introducing quoting or interpolation changes.
    for key, value in config.items():
        if not isinstance(value, str) or not re.fullmatch(r'/[A-Za-z0-9_./-]+', value):
            parser.error(f'{key}: use an absolute POSIX path without spaces or shell metacharacters')
    pending = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name == '.DS_Store' or '__pycache__' in path.parts:
            continue
        text = path.read_text(encoding='utf-8-sig')
        keys = set(TOKEN.findall(text))
        missing = keys - config.keys()
        if missing:
            parser.error(f'{path.relative_to(source)}: missing keys {sorted(missing)}')
        text = TOKEN.sub(lambda m: config[m[1]].rstrip('/'), text)
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
        elif path.suffix == '.ipynb':
            json.loads(text)
        pending.append((path.relative_to(source), text))
    # All configuration and syntax checks finish before writing any files.
    output.mkdir(parents=True)
    for rel, text in pending:
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    # Several analysis scripts import the shared `utils` module from the workflow root.
    # PYTHONPATH can point here without copying any third-party packages.
    print(f'Configured {len(pending)} files in {output}. No analysis was run.')
    print('Shared helper module:', output / 'utils.py')
    print('For workflow scripts, add this directory to PYTHONPATH:', output)
    print('Inspect ANALYSIS_MAP.md and INPUTS_AND_ENVIRONMENTS.md in the archive root before selecting an entry point.')

if __name__ == '__main__':
    main()
