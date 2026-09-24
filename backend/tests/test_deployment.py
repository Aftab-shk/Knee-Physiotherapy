"""
The two deployment guards: refusing an unsafe production start, and serving the
frontend from the API when its pages are bundled alongside it.

Run:  python -m pytest backend/tests/test_deployment.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient

import auth
import main


class RealEnoughClassifier:
    """A classifier that loaded actual weights."""
    demo_mode = False


@pytest.fixture
def prod(monkeypatch):
    """ENV=production with every guard satisfied; each test breaks one."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("MAIL_BACKEND", "smtp")
    monkeypatch.setattr(auth, "JWT_SECRET_IS_EPHEMERAL", False)
    monkeypatch.setattr(main, "classifier", RealEnoughClassifier())


def test_a_configured_production_start_is_allowed(prod):
    main.refuse_unsafe_production()


def test_development_tolerates_everything(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setattr(auth, "JWT_SECRET_IS_EPHEMERAL", True)
    monkeypatch.setattr(main, "classifier", None)
    main.refuse_unsafe_production()


def test_production_refuses_a_generated_jwt_secret(prod, monkeypatch):
    monkeypatch.setattr(auth, "JWT_SECRET_IS_EPHEMERAL", True)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        main.refuse_unsafe_production()


def test_production_refuses_demo_mode(prod, monkeypatch):
    """The one that matters: mock grades are shaped exactly like real readings."""
    stub = RealEnoughClassifier()
    stub.demo_mode = True
    monkeypatch.setattr(main, "classifier", stub)
    with pytest.raises(RuntimeError, match="demo mode"):
        main.refuse_unsafe_production()


def test_production_refuses_a_missing_classifier(prod, monkeypatch):
    monkeypatch.setattr(main, "classifier", None)
    with pytest.raises(RuntimeError, match="torch"):
        main.refuse_unsafe_production()


# ---------------------------------------------------------------------------
# Serving the frontend
# ---------------------------------------------------------------------------

def test_the_frontend_is_bundled_and_served():
    """
    The repository layout puts the pages one level up from backend/, which is
    where FRONTEND_DIR looks by default. If that stops being true the app still
    starts — it just silently serves the API docs at / instead of the app.
    """
    assert main.FRONTEND_DIR is not None, "frontend/index.html was not found"

    with TestClient(main.app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "<!DOCTYPE html>" in root.text[:200].upper() or "<html" in root.text[:400].lower()

        # A page below the root, and an asset: both have to come from disk
        # rather than being swallowed by the API's own 404.
        assert client.get("/login.html").status_code == 200
        assert client.get("/assets/config.js").status_code == 200

        # And the mount must not have shadowed the API it sits under.
        assert client.get("/health").json()["status"] in ("ok", "degraded")
        assert client.get("/docs").status_code == 200


def test_production_refuses_to_print_reset_links_into_the_log(prod, monkeypatch):
    """
    The default mail backend writes a working credential to the log. Fine on a
    laptop, an account takeover anywhere the log is shipped or shared.
    """
    monkeypatch.setenv("MAIL_BACKEND", "log")
    with pytest.raises(RuntimeError, match="MAIL_BACKEND"):
        main.refuse_unsafe_production()


# ---------------------------------------------------------------------------
# The torch / numpy pairing
# ---------------------------------------------------------------------------

def test_torch_can_still_hand_a_tensor_to_numpy():
    """
    The pin that drifted, checked the only way that actually means anything.

    torch before 2.5 was built against NumPy 1. Install NumPy 2 beside it and
    nothing complains: the import succeeds, the model loads, a forward pass
    returns. It breaks at the first `.numpy()` — which on this codebase is
    model/gradcam.py:149, so the symptom is the explanation overlay throwing on
    a live request while every other endpoint looks healthy.

    requirements.txt pins the pair and requirements-dev.txt back-pins matplotlib
    and scikit-learn to keep numpy from being dragged forward underneath it. Both
    of those are statements about versions, and versions drift — the image and
    the checked-in virtualenvs hold torch 2.3.0 while a current machine resolves
    2.12.1, and both are correct because each brings its matching numpy.

    So this asserts the bridge rather than the numbers. It needs no table of
    known-good combinations and cannot go stale.
    """
    torch = pytest.importorskip("torch", reason="the API tests run without torch on CI")

    assert torch.zeros(3).numpy().shape == (3,)
