"""Copy verified output JSONs with a different team_id, leaving the source intact."""
import argparse
import json
from pathlib import Path

from check_output import read_json
from submission import check_submission

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('team_id')
    args = parser.parse_args()
    files = sorted(args.source.glob('*.json'))
    if not files:
        parser.error('No JSON output files to copy')
    if args.destination.exists():
        parser.error('Destination must be a new directory')
    documents = []
    for path in files:
        doc = read_json(path)
        if check_submission(doc) or doc['video_id'] != path.stem:
            parser.error(f'Unverified source output: {path.name}')
        doc['team_id'] = args.team_id
        check_submission(doc)
        documents.append((path.name, doc))
    args.destination.mkdir(parents=True)
    for name, doc in documents:
        (args.destination / name).write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Copied {len(documents)} locally checked outputs.')

if __name__ == '__main__':
    main()
