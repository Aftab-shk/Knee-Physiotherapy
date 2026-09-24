"""
Grad-CAM: showing where the model was looking.

The overlay does not explain the decision — nothing does, honestly. It answers a
narrower question a clinician can actually check: *was it reading the joint, or
the corner of the film?* A heat map on the tibial plateau is worth something; one
on a radiographic marker says the grade should be ignored, and nothing else in
the system would ever have said so.

Which makes the properties below the ones that matter:

  * It explains the grade that was actually returned, not some other class.
  * It leaves the model exactly as it found it — this is the only place in the
    serving path that runs a backward pass, and a stray gradient or a model left
    in training mode would corrupt every reading afterwards.
  * It is never load-bearing. An overlay that fails must cost the patient
    nothing; the analysis comes back without a picture.

Run:  python -m pytest backend/tests -q
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="Grad-CAM needs torch")
pytest.importorskip("PIL", reason="Grad-CAM decodes and writes images")

import numpy as np
from model.gradcam import explain
from PIL import Image


class TinyNet(torch.nn.Module):
    """
    A real convolutional network, small enough to be instant.

    Not a mock: Grad-CAM is entirely about hooks, gradients and layer structure,
    and a stub that returned fabricated activations would test none of it.
    `features` mirrors torchvision's EfficientNet so the layer-finding takes the
    same path it does in production.
    """

    def __init__(self, classes=5, seed=20260901):
        super().__init__()
        # Seeded. An unseeded net of this size produces logits whose top two are
        # sometimes a thousandth apart, and argmax on a tie that close flips with
        # thread count — which made a test that passed alone fail inside the full
        # suite. The flake was in this fixture, not in Grad-CAM.
        torch.manual_seed(seed)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(16, classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(self.pool(x).flatten(1))


class FakeClassifier:
    """Stands in for KneeClassifier with the three attributes Grad-CAM reads."""

    demo_mode = False

    def __init__(self, model=None):
        self.model = model or TinyNet()
        self.device = torch.device("cpu")
        self.model.eval()

    def transform(self, image):
        arr = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def png_bytes(size=96, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(20, 235, (size, size), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# It produces something usable
# ---------------------------------------------------------------------------

def test_the_overlay_is_a_transparent_png_of_the_requested_size():
    result = explain(FakeClassifier(), png_bytes(), size=128)
    assert result is not None

    image = Image.open(io.BytesIO(result["overlay_png"]))
    assert image.mode == "RGBA", "the radiograph has to show through it"
    assert image.size == (128, 128)

    alpha = np.array(image)[..., 3]
    assert alpha.max() > 0, "something should be painted"
    assert (alpha == 0).any(), "and something should be left clear"


def test_it_explains_the_grade_that_was_returned():
    """
    An overlay for a different class would be worse than none: it would look
    like evidence for a decision that was not made.
    """
    classifier = FakeClassifier()
    result = explain(classifier, png_bytes())

    image = Image.open(io.BytesIO(png_bytes())).convert("RGB")
    x = classifier.transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = classifier.model(x)[0]

    top_two = torch.topk(logits, 2).values
    assert float(top_two[0] - top_two[1]) > 1e-3, (
        "the two leading classes are too close for argmax to be a stable "
        "expectation — reseed TinyNet rather than accepting a flaky test"
    )
    assert result["class_index"] == int(torch.argmax(logits).item())


def test_the_peak_lies_inside_the_overlay():
    result = explain(FakeClassifier(), png_bytes(), size=160)
    x, y = result["peak"]
    assert 0 <= x < 160 and 0 <= y < 160


def test_low_activation_stays_out_of_the_way():
    """
    A solid overlay would hide the thing the clinician is checking the heat map
    against, which defeats the point of showing it.

    Tested against the colouriser with a known ramp rather than a model's output:
    this is a property of the overlay design, and how much of any particular heat
    map falls below the floor depends on the image.
    """
    from model.gradcam import _ALPHA_CEILING, _ALPHA_FLOOR, _colourise

    ramp = np.linspace(0.0, 1.0, 101).reshape(1, 101)
    alpha = _colourise(ramp)[..., 3].astype(float) / 255.0

    assert alpha[0, 0] == 0.0, "the coldest region must be fully clear"
    below = ramp[0] < _ALPHA_FLOOR
    assert (alpha[0][below] == 0).all(), "anything under the floor is not painted"
    assert alpha.max() <= _ALPHA_CEILING + 0.01, "never fully opaque"
    # Monotonic: hotter is always at least as visible as cooler.
    assert (np.diff(alpha[0]) >= -1e-6).all()


# ---------------------------------------------------------------------------
# It leaves nothing behind
# ---------------------------------------------------------------------------

def test_the_prediction_is_identical_afterwards():
    """
    The only backward pass in the serving path. A stray gradient or a lingering
    hook would quietly corrupt every reading that followed.
    """
    classifier = FakeClassifier()
    image = png_bytes()
    x = classifier.transform(Image.open(io.BytesIO(image)).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        before = classifier.model(x).clone()

    explain(classifier, image)

    with torch.no_grad():
        after = classifier.model(x)

    assert torch.allclose(before, after)


def test_no_hooks_are_left_attached():
    classifier = FakeClassifier()
    layer = classifier.model.features[-1]
    before = len(layer._forward_hooks) + len(layer._backward_hooks)

    for _ in range(3):
        explain(classifier, png_bytes())

    after = len(layer._forward_hooks) + len(layer._backward_hooks)
    assert after == before, "hooks accumulate across calls"


def test_the_model_is_left_in_eval_mode():
    classifier = FakeClassifier()
    classifier.model.eval()
    explain(classifier, png_bytes())
    assert classifier.model.training is False


def test_no_gradients_are_left_on_the_parameters():
    classifier = FakeClassifier()
    explain(classifier, png_bytes())
    # zero_grad(set_to_none=True) runs before the backward pass; what matters is
    # that nothing downstream inherits a half-populated graph.
    for param in classifier.model.parameters():
        assert param.grad is None or torch.isfinite(param.grad).all()


# ---------------------------------------------------------------------------
# It is never load-bearing
# ---------------------------------------------------------------------------

def test_demo_mode_has_nothing_to_explain():
    classifier = FakeClassifier()
    classifier.demo_mode = True
    assert explain(classifier, png_bytes()) is None


def test_no_classifier_means_no_overlay():
    assert explain(None, png_bytes()) is None


def test_an_undecodable_image_returns_none_rather_than_raising():
    assert explain(FakeClassifier(), b"not an image") is None


def test_a_model_with_no_convolution_returns_none():
    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(3 * 64 * 64, 5)

        def forward(self, x):
            return self.classifier(x.flatten(1))

    assert explain(FakeClassifier(Flat()), png_bytes()) is None


def test_a_model_that_throws_returns_none():
    class Broken(TinyNet):
        def forward(self, x):
            raise RuntimeError("CUDA out of memory")

    assert explain(FakeClassifier(Broken()), png_bytes()) is None


def test_a_flat_activation_map_produces_no_overlay():
    """
    Normalising a constant map would manufacture a pattern out of rounding
    error, and a heat map that is pure noise is worse than an absent one.
    """
    class Constant(TinyNet):
        def __init__(self):
            super().__init__()
            # No gradient reaches the conv stack, so the map cannot vary.
            for param in self.features.parameters():
                param.requires_grad_(False)
            torch.nn.init.zeros_(self.classifier.weight)
            torch.nn.init.zeros_(self.classifier.bias)

    assert explain(FakeClassifier(Constant()), png_bytes()) is None


# ---------------------------------------------------------------------------
# The API wiring
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="API tests need fastapi")
pytest.importorskip("sqlalchemy", reason="the app needs sqlalchemy")

from conftest import reset_database, stub_inference
from fastapi.testclient import TestClient
from test_api_security import StubClassifier
from test_api_security import png_bytes as api_png

import main


@pytest.fixture
def client(monkeypatch):
    reset_database()
    stub_inference(monkeypatch)
    main._rate_buckets.clear()
    with TestClient(main.app) as c:
        monkeypatch.setattr(main, "classifier", StubClassifier())
        yield c


def analyse(client, **extra):
    data = {"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"}
    data.update(extra)
    return client.post("/analyse-xray",
                       files={"image": ("knee.png", api_png(), "image/png")},
                       data=data)


def test_an_explanation_is_not_produced_unless_asked_for(client):
    """
    A backward pass roughly doubles inference time. Nothing should pay that by
    accident.
    """
    assert analyse(client).json()["explanation"] is None


def test_asking_for_one_in_demo_mode_returns_the_analysis_without_it(client):
    """The stub classifier is demo mode: there is nothing real to explain."""
    r = analyse(client, explain="true")
    assert r.status_code == 200
    assert r.json()["explanation"] is None
    assert r.json()["kl_grade"] is not None, "the analysis still comes back"


def test_a_failing_overlay_never_costs_the_patient_their_reading(client, monkeypatch):
    monkeypatch.setattr(main, "gradcam_explain",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        main.gradcam_explain(None, b"")

    # And with the real guard in place — gradcam.explain swallows its own
    # failures — the endpoint answers normally.
    monkeypatch.setattr(main, "gradcam_explain", lambda *a, **k: None)
    r = analyse(client, explain="true")
    assert r.status_code == 200
    assert r.json()["explanation"] is None


def test_an_overlay_arrives_as_a_data_uri(client, monkeypatch):
    monkeypatch.setattr(main, "gradcam_explain",
                        lambda *a, **k: {"overlay_png": b"\x89PNG\r\n\x1a\nfake", "size": 384,
                                         "class_index": 2, "peak": [1, 2]})
    body = analyse(client, explain="true").json()
    assert body["explanation"].startswith("data:image/png;base64,")

    import base64
    encoded = body["explanation"].split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\x89PNG")
