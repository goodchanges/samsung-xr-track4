import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from check_output import check_directory, read_json
from submission import check_submission


def example():
    return dict(team_id='team', video_id='v', duration_sec=100.0,
                label=dict(name='cooking', score=0.5),
                actions=[dict(segment=[1.0, 2.0], verb='take', noun='cup', score=0.5)],
                highlights=[dict(segment=[0.0, 4.0]), dict(segment=[4.0, 8.0])])


class SubmissionChecks(unittest.TestCase):
    def test_organizer_examples_when_kit_is_present(self):
        kit = Path(__file__).resolve().parents[1] / 'kit'
        if not (kit / 'P02_109.json').is_file() or not (kit / 'skiing_03.json').is_file():
            self.skipTest('Organizer example files are supplied separately')
        self.assertEqual(check_submission(read_json(kit / 'P02_109.json')), [])
        skiing = read_json(kit / 'skiing_03.json')
        with self.assertRaisesRegex(ValueError, 'unsupported class'):
            check_submission(skiing)
        skiing['label']['name'] = 'ski'
        self.assertEqual(check_submission(skiing), [])

    def test_valid_and_invalid_fields(self):
        self.assertEqual(check_submission(example()), [])
        variants = [
            (('team_id',), 12), (('video_id',), ' '),
            (('duration_sec',), True), (('duration_sec',), float('nan')),
            (('duration_sec',), 0), (('duration_sec',), '100'),
            (('label',), 'cooking'), (('label', 'name'), 'skiing'),
            (('label', 'score'), True), (('label', 'score'), float('inf')),
            (('label', 'score'), -0.01), (('actions',), {}),
            (('actions', 0, 'verb'), 'invented'), (('actions', 0, 'noun'), 42),
            (('actions', 0, 'score'), 1.1), (('actions', 0, 'segment'), [0, 1, 2]),
            (('actions', 0, 'segment'), [1, 1]),
            (('actions', 0, 'segment'), [0, 100.51]),
            (('actions', 0, 'segment'), [0, 1.001]),
            (('actions', 0, 'segment'), [False, 1]),
            (('highlights',), []), (('highlights', 0, 'segment'), [0, 3.99]),
            (('highlights', 1, 'segment'), [3, 7]),
            (('highlights', 1, 'segment'), [4, 14.51]),
        ]
        for keys, value in variants:
            with self.subTest(keys=keys, value=value):
                document = copy.deepcopy(example())
                parent = document
                for key in keys[:-1]:
                    parent = parent[key]
                parent[keys[-1]] = value
                with self.assertRaises(ValueError):
                    check_submission(document)
        for document in ([], dict(example(), extra=1), {k: v for k, v in example().items() if k != 'team_id'}):
            with self.assertRaises(ValueError):
                check_submission(document)
        document = example()
        document['actions'] *= 2001
        with self.assertRaises(ValueError):
            check_submission(document)

    def test_short_video_conflict_is_unresolved(self):
        document = example()
        document.update(duration_sec=11.13, highlights=[dict(segment=[0., 4.])])
        self.assertIn('UNRESOLVED', check_submission(document)[0])

    def test_directory_accounts_for_every_manifest_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outputs = root / 'outputs'
            outputs.mkdir()
            manifest = root / 'manifest.csv'
            manifest.write_text('video_id,duration_sec,dataset\nv,100,epic_kitchens_100\nw,100,fpvsum\n')
            (outputs / 'v.json').write_text(json.dumps(example()))
            report = check_directory(outputs, manifest)
            self.assertEqual(report['missing'], ['w.json'])
            self.assertEqual(report['status'], 'failed')
            other = dict(example(), video_id='w')
            (outputs / 'w.json').write_text(json.dumps(other))
            self.assertEqual(check_directory(outputs, manifest)['status'], 'passed')
            for key, value in [('team_id', 'different'), ('video_id', 'wrong'), ('duration_sec', 101)]:
                (outputs / 'w.json').write_text(json.dumps(dict(other, **{key: value})))
                self.assertEqual(len(check_directory(outputs, manifest)['errors']), 1)
            (outputs / 'w.json').write_text('{invalid')
            self.assertEqual(len(check_directory(outputs, manifest)['errors']), 1)
            (outputs / 'extra.json').write_text('{}')
            self.assertEqual(check_directory(outputs, manifest)['unexpected'], ['extra.json'])
            manifest.write_text('video_id,duration_sec,dataset\nv,100,epic\nV,100,epic\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                check_directory(outputs, manifest)

    def test_duplicate_json_keys_and_optimized_python(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            path.write_text('{"team_id":"a","team_id":"b"}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_json(path)
        code = "from submission import check_submission; check_submission({})"
        run = subprocess.run([sys.executable, '-O', '-c', code], cwd=Path(__file__).resolve().parents[1] / 'code', capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('ValueError', run.stderr)

    def test_cli_returns_nonzero_for_unresolved_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outputs = root / 'outputs'
            outputs.mkdir()
            document = example()
            document.update(duration_sec=11.13, highlights=[dict(segment=[0., 4.])])
            (outputs / 'v.json').write_text(json.dumps(document))
            manifest = root / 'manifest.csv'
            manifest.write_text('video_id,duration_sec,dataset\nv,11.13,epic_kitchens_100\n')
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'code/check_output.py'),
                       '--dir', str(outputs), '--manifest', str(manifest), '--team-id', 'team']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertEqual(json.loads(run.stdout)['status'], 'unresolved')
            document = example()
            (outputs / 'v.json').write_text(json.dumps(document))
            manifest.write_text('video_id,duration_sec,dataset\nv,100,epic_kitchens_100\n')
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
