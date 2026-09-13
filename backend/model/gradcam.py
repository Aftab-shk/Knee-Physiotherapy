"""
gradcam.py — where the model was looking when it chose a grade.

A number that restricts how far someone bends a healing knee ought to be able to
show its work. Grad-CAM does not explain the decision — nothing does, honestly —
but it answers a question a clinician can actually check: *was it looking at the
joint space, or at the corner of the film?* A heat map centred on the tibial
plateau is worth something. One centred on a radiographic marker says the grade
should be ignored, and nothing else in the system would ever have told you.

How it works, briefly
---------------------
Take the activations of the last convolutional block and the gradient of the
chosen class score with respect to them. Average each gradient channel to get a
weight — how much that feature map pushed the score — then sum the maps by those
weights and keep the positive part. What survives is the evidence *for* the
grade, which is the question being asked.

Computed on the un-flipped pass only. `_real_predict` averages two test-time
augmentations, and a map averaged across a mirror image would be smeared into
symmetry that is not there.

Deliberately separate from inference.py: this needs gradients, which that file
takes care to run without, and a backward pass is a cost nothing should pay by
accident.
"""

import io
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("physio-backend.gradcam")

# Blue → red through translucent green. Low activation stays out of the way so
# the radiograph underneath is still readable; a solid overlay would hide the
# thing the clinician is checking the heat map against.
_COLOURS = np.array([
    [0,   0, 140],
    [0, 110, 200],
    [0, 190, 160],
    [240, 200,  40],
    [230,  90,  30],
    [190,   0,  20],
], dtype=np.float32)

# Below this the region contributed little; painting it adds noise, not meaning.
_ALPHA_FLOOR = 0.15
_ALPHA_CEILING = 0.72


def _last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """
    The layer to read activations from: the deepest one that still has a spatial
    grid. For torchvision's EfficientNet that is the end of `features`; the
    fallback walks the graph for anything else a checkpoint might carry.
    """
    features = getattr(model, "features", None)
    if features is not None and len(features) > 0:
        return features[-1]

    conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv = module
    if conv is None:
        raise ValueError("no convolutional layer to attach Grad-CAM to")
    return conv


def _colourise(cam: np.ndarray) -> np.ndarray:
    """Map a 0-1 heat map to RGBA, transparent where the model was not looking."""
    positions = cam * (len(_COLOURS) - 1)
    lower = np.clip(np.floor(positions).astype(int), 0, len(_COLOURS) - 1)
    upper = np.clip(lower + 1, 0, len(_COLOURS) - 1)
    blend = (positions - lower)[..., None]

    rgb = _COLOURS[lower] * (1 - blend) + _COLOURS[upper] * blend
    alpha = np.clip((cam - _ALPHA_FLOOR) / (1 - _ALPHA_FLOOR), 0, 1) * _ALPHA_CEILING

    out = np.zeros((*cam.shape, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = (alpha * 255).astype(np.uint8)
    return out


def explain(classifier, image_bytes: bytes, size: int = 384) -> dict | None:
    """
    Produce a Grad-CAM overlay for `classifier`'s own prediction.

    Returns {"overlay_png": bytes, "class_index": int, "peak": [x, y]} — the
    overlay being an RGBA PNG at `size`x`size`, meant to be drawn on top of the
    uploaded X-ray at the same box.

    Returns None rather than raising when there is nothing to explain (demo
    mode, no model) or when the attempt fails. An explanation is a courtesy; it
    must never be the reason an analysis does not come back.
    """
    if classifier is None or getattr(classifier, "demo_mode", True):
        return None

    model = classifier.model
    activations: dict = {}
    gradients: dict = {}
    handles: list = []
    was_training = model.training

    try:
        # Inside the guard: a checkpoint with no convolutional layer is one of
        # the things that must degrade to "no picture" rather than to an error.
        target = _last_conv(model)

        def save_activation(_module, _inputs, output):
            activations["value"] = output

        def save_gradient(_module, _grad_in, grad_out):
            gradients["value"] = grad_out[0]

        handles = [
            target.register_forward_hook(save_activation),
            target.register_full_backward_hook(save_gradient),
        ]

        model.eval()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = classifier.transform(image).unsqueeze(0).to(classifier.device)

        # Gradients are needed here, unlike everywhere else in the serving path.
        with torch.enable_grad():
            x.requires_grad_(True)
            logits = model(x)
            class_index = int(torch.argmax(logits, dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()

        acts = activations.get("value")
        grads = gradients.get("value")
        if acts is None or grads is None:
            logger.warning("Grad-CAM hooks captured nothing; skipping the overlay")
            return None

        # One weight per channel: how strongly that feature map pushed the score.
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        span = float(cam.max() - cam.min())
        if span <= 1e-8:
            # A flat map means the layer contributed nothing distinguishable.
            # Normalising it would manufacture a pattern out of rounding error.
            logger.info("Grad-CAM produced a flat map; no overlay to show")
            return None
        cam = (cam - float(cam.min())) / span

        peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)

        buffer = io.BytesIO()
        Image.fromarray(_colourise(cam), mode="RGBA").save(buffer, format="PNG", optimize=True)

        return {
            "overlay_png": buffer.getvalue(),
            "class_index": class_index,
            "peak": [int(peak_x), int(peak_y)],
            "size": size,
        }

    except Exception:
        logger.exception("Grad-CAM failed; returning the analysis without an overlay")
        return None

    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
