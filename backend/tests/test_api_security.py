"""
API surface: CORS, upload limits, rate limiting, and input validation.

These run against the real app with the classifier stubbed out, so no torch and
no checkpoint are needed.

Run:  python -m pytest backend/tests -q
"""

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi", reason="API tests need fastapi")
from fastapi.testclient import TestClient

import main


class StubClassifier:
    """Stands in for KneeClassifier so the API can be exercised without weights."""
    demo_mode = True
    model_version = "stub"
    calibration = {"temperature": 1.0, "calibrated": False,
                   "warn_threshold": None, "reject_threshold": None, "ece": None}

    def __init__(self, **overrides):
        self.overrides = overrides

    def predict(self, image_bytes):
        result = {
            "kl_grade": 2, "health_score": 60, "max_angle": 90,
            "confidence": 0.7, "confidence_band": "moderate", "calibrated": False,
            "energy": None, "ood_suspected": False, "ood_reject": False,
            "ood_screened": False, "demo_mode": True,
        }
        result.update(self.overrides)
        return result


def png_bytes(size_px=64, payload=None):
    """A valid PNG that passes validate_image (needs contrast, not blank)."""
    np = pytest.importorskip("numpy")
    from PIL import Image
    rng = np.random.default_rng(0)
    arr = rng.integers(20, 235, (size_px, size_px), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="PNG")
    data = buf.getvalue()
    return data + (payload or b"")


@pytest.fixture
def client(monkeypatch):
    """
    The stub must be installed AFTER entering TestClient: the lifespan handler
    constructs a real KneeClassifier and assigns it to the module global, so
    patching beforehand is silently undone — which had these tests running real
    B4 inference instead of the stub.
    """
    main._rate_buckets.clear()
    with TestClient(main.app) as c:
        monkeypatch.setattr(main, "classifier", StubClassifier())
        yield c


def form(**over):
    d = {"knee_side": "left", "surgery_type": "tkr", "weeks_post_op": "3"}
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def test_null_origin_is_not_allowed():
    """
    "null" is the origin of sandboxed iframes and file:// pages. Allowing it —
    as this API used to, alongside allow_credentials — lets any local HTML file
    or hostile sandboxed frame call the API.
    """
    assert "null" not in main.CORS_ORIGINS


def test_credentials_are_disabled():
    """No cookies or auth headers are used, so credentials must stay off:
    with them on, an allowed origin can ride a browser session."""
    cors = [m for m in main.app.user_middleware if "CORS" in str(m)]
    assert cors, "CORS middleware should be installed"
    assert cors[0].kwargs["allow_credentials"] is False


def test_methods_and_headers_are_not_wildcarded():
    cors = next(m for m in main.app.user_middleware if "CORS" in str(m))
    assert cors.kwargs["allow_methods"] == ["GET", "POST", "OPTIONS"]
    assert "*" not in cors.kwargs["allow_headers"]


def test_allowed_origin_gets_cors_header(client):
    r = client.get("/health", headers={"Origin": "http://localhost:8080"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8080"


def test_disallowed_origin_gets_no_cors_header(client):
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert r.headers.get("access-control-allow-origin") is None


# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------

def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 50_000)
    big = png_bytes(payload=b"\0" * 80_000)
    r = client.post("/analyse-xray", data=form(),
                    files={"image": ("x.png", big, "image/png")})
    assert r.status_code == 413


def test_upload_under_the_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 5_000_000)
    r = client.post("/analyse-xray", data=form(),
                    files={"image": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text


def test_read_capped_aborts_before_buffering_everything():
    """
    The point of read_capped: it must stop reading at the ceiling rather than
    buffer the whole body and check afterwards.
    """
    import asyncio

    from fastapi import HTTPException

    class EndlessUpload:
        def __init__(self):
            self.read_bytes = 0

        async def read(self, n):
            self.read_bytes += n
            return b"\0" * n

    up = EndlessUpload()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.read_capped(up, 3 << 20))
    assert exc.value.status_code == 413
    # Stopped near the limit, not at some unbounded size.
    assert up.read_bytes <= (3 << 20) + (1 << 20)


def test_wrong_content_type_is_rejected(client):
    r = client.post("/analyse-xray", data=form(),
                    files={"image": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_kicks_in(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(main, "RATE_LIMIT_WINDOW_S", 60)
    main._rate_buckets.clear()

    codes = [
        client.post("/analyse-xray", data=form(),
                    files={"image": ("x.png", png_bytes(), "image/png")}).status_code
        for _ in range(5)
    ]
    assert codes[:3] == [200, 200, 200], codes
    assert codes[3:] == [429, 429], codes


def test_rate_limited_response_carries_retry_after(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 1)
    main._rate_buckets.clear()

    client.post("/analyse-xray", data=form(), files={"image": ("x.png", png_bytes(), "image/png")})
    r = client.post("/analyse-xray", data=form(), files={"image": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_rate_limit_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 0)
    main._rate_buckets.clear()
    for _ in range(6):
        r = client.post("/analyse-xray", data=form(),
                        files={"image": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200


def test_forwarded_for_is_ignored_unless_proxy_is_trusted(monkeypatch):
    """Otherwise a caller spoofs the header and gets a fresh bucket each time,
    which makes the limiter worse than none at all."""
    from starlette.datastructures import Headers

    class FakeRequest:
        headers = Headers({"x-forwarded-for": "1.2.3.4"})
        client = type("C", (), {"host": "10.0.0.1"})()

    monkeypatch.delenv("TRUST_PROXY", raising=False)
    assert main._client_key(FakeRequest()) == "10.0.0.1"

    monkeypatch.setenv("TRUST_PROXY", "true")
    assert main._client_key(FakeRequest()) == "1.2.3.4"


# ---------------------------------------------------------------------------
# OOD gate + validation
# ---------------------------------------------------------------------------

def test_ood_rejected_image_returns_422(client, monkeypatch):
    monkeypatch.setattr(main, "classifier", StubClassifier(ood_reject=True, energy=3.2))
    main._rate_buckets.clear()
    r = client.post("/analyse-xray", data=form(),
                    files={"image": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 422
    assert "knee" in r.json()["detail"].lower()


def test_missing_weeks_post_op_is_rejected(client):
    r = client.post("/analyse-xray", data=form(weeks_post_op=""),
                    files={"image": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 422


def test_health_reports_calibration_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["calibrated"] is False
    assert body["ood_screening"] is False


def test_expired_rate_limit_buckets_are_reclaimed(monkeypatch):
    """
    Regression: cleanup used to delete only already-empty buckets. Entries are
    trimmed on a key's own next request, so a client that calls once and never
    returns left a non-empty deque behind forever and the sweep freed nothing.

    Deliberately not timing-dependent — an earlier version of this test raced
    the loop against a one-second window and passed or failed on how warm the
    machine was. Here the clock is moved forward explicitly to force expiry.
    """
    import asyncio
    import time

    from starlette.datastructures import Headers

    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 5)
    monkeypatch.setattr(main, "_RATE_BUCKET_SOFT_CAP", 200)
    monkeypatch.setattr(main, "_RATE_BUCKET_HARD_CAP", 100_000)  # isolate the sweep
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    main._rate_buckets.clear()

    def request_from(host):
        return type("R", (), {
            "headers": Headers({}),
            "client": type("C", (), {"host": host})(),
        })()

    async def drive():
        # Long window: nothing expires, so every client keeps its bucket.
        monkeypatch.setattr(main, "RATE_LIMIT_WINDOW_S", 3600)
        for i in range(1000):
            await main.enforce_rate_limit(request_from(f"10.0.{i // 256}.{i % 256}"))
        populated = len(main._rate_buckets)

        # Age every existing entry out by jumping the clock a full window
        # forward, then make one more call — which is above the soft cap and
        # therefore triggers the sweep.
        #
        # Not by setting the window to 0: that puts the cutoff exactly on `now`,
        # and the sweep's `stale[0] < cutoff` is strict, so every bucket stamped
        # in the same clock tick as the final call survives. time.monotonic()
        # ticks at ~15.6ms on Windows, which covered the last few hundred of the
        # loop above and left them behind.
        monkeypatch.setattr(
            main.time, "monotonic", lambda base=time.monotonic(): base + 2 * 3600
        )
        await main.enforce_rate_limit(request_from("10.9.9.9"))
        return populated, len(main._rate_buckets)

    populated, after = asyncio.run(drive())

    assert populated == 1000, f"expected 1000 live buckets, got {populated}"
    # Under the old cleanup this stayed at 1001: nothing was ever reclaimed.
    assert after <= 2, f"expired buckets should have been swept, {after} remain"

    main._rate_buckets.clear()


def test_rate_limit_table_is_hard_capped_against_a_distributed_flood(monkeypatch):
    """
    Sweeping expired entries is not enough on its own: a flood spread across
    many source addresses inside the window leaves every bucket live, so the
    sweep frees nothing and the table grows with attacker-chosen input. That
    turns a DoS defence into a memory-exhaustion vector, so there is a hard cap.
    """
    import asyncio

    from starlette.datastructures import Headers

    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 5)
    monkeypatch.setattr(main, "RATE_LIMIT_WINDOW_S", 3600)   # nothing expires
    monkeypatch.setattr(main, "_RATE_BUCKET_SOFT_CAP", 200)
    monkeypatch.setattr(main, "_RATE_BUCKET_HARD_CAP", 400)
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    main._rate_buckets.clear()

    async def drive():
        for i in range(3000):
            req = type("R", (), {
                "headers": Headers({}),
                "client": type("C", (), {"host": f"172.16.{i // 256}.{i % 256}"})(),
            })()
            await main.enforce_rate_limit(req)
        return len(main._rate_buckets)

    final = asyncio.run(drive())

    assert final <= main._RATE_BUCKET_HARD_CAP, (
        f"3000 distinct live clients grew the table to {final}, above the "
        f"{main._RATE_BUCKET_HARD_CAP} hard cap"
    )
    main._rate_buckets.clear()
