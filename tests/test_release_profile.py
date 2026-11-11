import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from runtime_profile import apply_profile
import main


class ReleaseProfileChecks(unittest.TestCase):
    def test_packaged_profile_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'code').mkdir()
            (root/'code/inference_profile.txt').write_text('improved\n')
            def options(profile=None):
                return argparse.Namespace(profile=profile,selector=None,kitchen_routing=None,
                     temporal_model=None,temporal_decoder=None,temporal_context=None)
            args=options()
            apply_profile(args,root)
            self.assertEqual((args.selector,args.kitchen_routing,args.temporal_context),('preserved','scene',4))
            self.assertEqual(args.temporal_decoder,root/'weights/temporal_decoder.npz')
            args=options('clip')
            apply_profile(args,root)
            self.assertIsNone(args.temporal_model)
            self.assertEqual(args.kitchen_routing,'category')
            (root/'code/inference_profile.txt').write_text('unknown')
            with self.assertRaises(ValueError):apply_profile(options(),root)

    def test_demo_failure_preserves_successful_inference_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'input.mp4';source.touch()
            target=root/'out.json'
            result=Mock(label='cooking',label_score=.5,actions=[],highlights=[dict(start=0.,end=.29,importance=.5)],
                        route='clip',kitchen_gate_accepted=False)
            argv=['main.py',str(source),str(target),'--profile','clip','--demo',str(root/'demo')]
            with patch.object(sys,'argv',argv),patch.object(main,'load_supported_classes',return_value=(None,dict(labels=['cooking'],verbs=['take'],nouns=['cup']))), \
                 patch.object(main,'probe',return_value=(30.,False)),patch.object(main,'Scorer'), \
                 patch.object(main,'frames',return_value=b'raw'),patch.object(main,'infer_video',return_value=result), \
                 patch.object(main,'render',side_effect=ValueError('Simulated encoder failure')),patch.object(main,'log_event') as log:
                main.main()
            self.assertEqual(json.loads(target.read_text())['label']['name'],'cooking')
            self.assertEqual(json.loads(target.read_text())['highlights'],[dict(segment=[0.,4.0])])
            self.assertTrue(any(c.args[0]=='demo_failed' for c in log.call_args_list))
