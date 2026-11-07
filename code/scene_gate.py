"""Optional scene-based model-domain routing; these prompts are not output labels."""
import numpy as np
import torch

PROMPTS = [f'A first-person view inside a {place}.' for place in
           ('kitchen', 'bathroom', 'office', 'bedroom', 'living room', 'workshop', 'vehicle')]
PROMPTS.append('A first-person view outdoors.')
MARGIN = .01
FRAME_FRACTION = .5


def decide(similarities):
    means = similarities.mean(axis=0)
    margin = float(means[0] - means[1:].max())
    fraction = float(np.mean(similarities.argmax(axis=1) == 0))
    return margin >= MARGIN and fraction >= FRAME_FRACTION, margin, fraction


@torch.inference_mode()
def confidence(scorer, embeddings):
    if not hasattr(scorer, '_scene_text'):
        tokens = scorer.processor(text=PROMPTS, padding=True, return_tensors='pt')
        vectors = scorer.model.get_text_features(**tokens)
        scorer._scene_text = torch.nn.functional.normalize(vectors, dim=-1).numpy()
    return decide(embeddings @ scorer._scene_text.T)
