"""Provisional local checker. The challenge validator is supplied separately."""
import argparse
import json
import math
from pathlib import Path

import csv
import sys

from submission import check_submission


def check(document, video_id, duration):
    if "team_id" in document:
        warnings = check_submission(document)
        if document["video_id"] != video_id or abs(document["duration_sec"] - duration) > 0.5:
            raise ValueError("Video identity or duration mismatch")
        for warning in warnings:
            print(warning, file=sys.stderr)
        print("Local kit-guide checks completed with unresolved warnings." if warnings else "Local kit-guide checks passed; organizer validator unavailable.")
        return
    if set(document) != {"video_id", "label", "actions", "highlights"}:
        raise ValueError("Unexpected provisional top-level keys.")
    if document["video_id"] != video_id or not isinstance(document["label"], str):
        raise ValueError("Unexpected video ID or label.")
    total = 0
    previous_end = 0
    for name, extra in (("actions", ("verb", "noun")), ("highlights", ("importance",))):
        if not isinstance(document[name], list):
            raise ValueError(f"{name} must be an array.")
        for row in document[name]:
            if set(row) != {"start", "end", *extra}:
                raise ValueError(f"Unexpected keys in {name} interval.")
            start, end = row["start"], row["end"]
            if not all(type(n) in (int, float) and math.isfinite(n) for n in (start, end)):
                raise ValueError(f"Non-finite or non-numeric time in {name}.")
            if not 0 <= start < end <= duration:
                raise ValueError(f"Out of range interval in {name}.")
            if name == "actions":
                if not all(isinstance(row[key], str) and row[key] for key in extra):
                    raise ValueError("Missing verb or noun.")
            else:
                if start < previous_end:
                    raise ValueError("Highlights must be sorted and nonoverlapping.")
                importance = row["importance"]
                if type(importance) not in (int, float) or not 0 <= importance <= 1:
                    raise ValueError("Importance must be between zero and one.")
                previous_end = end
                total += end - start
    if total > duration * 0.15 + 1e-8:
        raise ValueError("Highlight duration exceeds 15%.")
    print(f"Provisional check passed ({total:.3f}/{duration:.3f}s).")


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_pairs)


def check_directory(directory, manifest, team_id=None):
    """Check every output and account for every official manifest row."""
    if not directory.is_dir():
        raise ValueError(f'Output directory does not exist: {directory}')
    expected = {}
    folded = set()
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'video_id', 'duration_sec', 'dataset'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest needs video_id, duration_sec, dataset columns')
        for line, row in enumerate(reader, 2):
            video_id = row.get('video_id')
            if not video_id or video_id != video_id.strip() or video_id in ('.', '..') or any(c in video_id for c in '/\\:*?"<>|'):
                raise ValueError(f'Manifest line {line}: unsafe or empty video_id')
            if video_id.casefold() in folded:
                raise ValueError(f'Manifest line {line}: duplicate video_id {video_id}')
            folded.add(video_id.casefold())
            duration = float(row['duration_sec'])
            if not math.isfinite(duration) or duration <= 0 or not (row.get('dataset') or '').strip():
                raise ValueError(f'Manifest line {line}: invalid duration or empty dataset')
            expected[video_id] = duration
    if not expected:
        raise ValueError('Manifest is empty')
    files = {p.name: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.json'}
    names = {video_id + '.json' for video_id in expected}
    report = dict(expected=len(expected), found=len(files), checked=0, valid=0,
                  missing=sorted(names - files.keys()), unexpected=sorted(files.keys() - names),
                  errors=[], warnings=[])
    for name in sorted(names & files.keys()):
        report['checked'] += 1
        try:
            document = read_json(files[name])
            warnings = check_submission(document)
            if document['video_id'] != Path(name).stem:
                raise ValueError('video_id does not match filename and manifest')
            if abs(document['duration_sec'] - expected[document['video_id']]) > 0.5 + 1e-8:
                raise ValueError('duration_sec differs from manifest by more than 0.5 seconds')
            if team_id is None:
                team_id = document['team_id']
            if document['team_id'] != team_id:
                raise ValueError(f'team_id differs from expected {team_id!r}')
            report['warnings'].extend(f'{name}: {warning}' for warning in warnings)
            if not warnings:
                report['valid'] += 1
        except (ValueError, TypeError, KeyError, OSError) as exc:
            report['errors'].append(f'{name}: {exc}')
    report['status'] = ('failed' if report['missing'] or report['unexpected'] or report['errors']
                        else 'unresolved' if report['warnings'] else 'passed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--dir', type=Path, dest='directory')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--team-id')
    parser.add_argument('--report', type=Path, help='Write the complete local-check report outside outputs')
    args = parser.parse_args()
    try:
        if args.directory:
            if not args.manifest or args.video or args.output:
                parser.error('--dir requires --manifest and no positional arguments')
            if args.report and args.report.resolve().is_relative_to(args.directory.resolve()):
                parser.error('Keep --report outside the output directory')
            report = check_directory(args.directory, args.manifest, args.team_id)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(report, indent=2))
            return 0 if report['status'] == 'passed' else 2 if report['status'] == 'unresolved' else 1
        if not args.video or not args.output or args.manifest or args.report or args.team_id:
            parser.error('Supply VIDEO OUTPUT, or --dir OUTPUTS --manifest MANIFEST')
        from media import probe
        duration, _ = probe(args.video)
        document = read_json(args.output)
        # CLI submission checks always require the current six-key format.
        warnings = check_submission(document)
        check(document, args.video.stem, duration)
        return 2 if warnings else 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f'Local check failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
