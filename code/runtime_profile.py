"""Small deployment preset; the inference output format is unaffected."""
from pathlib import Path


def apply_profile(args, root):
    path = root / 'code/inference_profile.txt'
    profile = args.profile or (path.read_text(encoding='utf-8').strip() if path.exists() else 'clip')
    if profile not in ('clip', 'improved'):
        raise ValueError('Inference profile must be clip or improved.')
    args.profile = profile
    if args.selector is None:
        args.selector = 'preserved'  # Selected on development coverage; temporal regressed.
    if args.kitchen_routing is None:
        args.kitchen_routing = 'scene' if profile == 'improved' else 'category'
    if profile == 'improved':
        args.temporal_model = args.temporal_model or root / 'weights/temporal/tsm_rgb.ckpt'
        args.temporal_context = args.temporal_context or 4
