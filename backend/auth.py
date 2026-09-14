"""
auth.py — password hashing, token issue/verify, and the request dependencies.

Configuration
-------------
  JWT_SECRET         signing key. **Required in production.** If unset, a random
                     one is generated per process: development keeps working, and
                     a deployment that forgot to set it logs loudly and logs
                     everyone out on restart rather than shipping a key that is
                     public knowledge because it was committed as a default.
  JWT_EXPIRY_HOURS   token lifetime, default 336 (14 days). Rehab is a daily
                     habit over months; a session that dies overnight trains
                     people to stop opening the app.
  MIN_PASSWORD_LEN   default 8.

Passwords are hashed with Argon2id — the current OWASP first choice, and it has
no equivalent of bcrypt's silent 72-byte truncation, where two different long
passwords hash identically.
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from db import get_db
from models import Clinician, Patient, utcnow

logger = logging.getLogger("physio-backend.auth")

ALGORITHM = "HS256"
MIN_PASSWORD_LEN = int(os.getenv("MIN_PASSWORD_LEN", "8"))
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "336"))

_ph = PasswordHasher()


def _load_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if secret:
        return secret
    logger.warning(
        "⚠️  JWT_SECRET is not set — generated a random key for this process. "
        "Every existing token is now invalid and will be again on the next "
        "restart. Set JWT_SECRET before deploying."
    )
    return secrets.token_urlsafe(48)


JWT_SECRET = _load_secret()

# True when the key above was generated rather than configured. main.py refuses
# to start with ENV=production in that state — a warning in a log nobody reads
# is not enough when the symptom is everyone being signed out on each deploy.
JWT_SECRET_IS_EPHEMERAL = not os.getenv("JWT_SECRET", "").strip()


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def normalise_email(email: str) -> str:
    """Lower-case and strip, so one address cannot become two accounts."""
    return email.strip().lower()


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        _ph.verify(password_hash, password)
        return True
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash predates the current Argon2 parameters."""
    try:
        return _ph.check_needs_rehash(password_hash)
    except InvalidHashError:
        return False


# A pre-computed hash of a throwaway value. Verifying against it on an unknown
# email makes a failed login cost the same as a successful one; without it the
# response time alone tells an attacker which addresses have accounts.
_DUMMY_HASH = _ph.hash("not-a-real-password-" + secrets.token_hex(8))


def waste_time_like_a_real_verify() -> None:
    try:
        _ph.verify(_DUMMY_HASH, "wrong")
    except VerifyMismatchError:
        pass


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

# Which table `sub` names. Carried in the token because patients and clinicians
# live in separate tables, and an id alone does not say which one to look in.
ROLE_PATIENT = "patient"
ROLE_CLINICIAN = "clinician"


def create_access_token(
    subject_id: str, role: str = ROLE_PATIENT, token_version: int = 0
) -> tuple[str, int]:
    """
    Return (token, seconds_until_expiry).

    token_version is the account's counter at the moment of issue. Bumping that
    counter is what invalidates every token already out there — see
    revoke_tokens() and _still_valid().
    """
    expires_delta = timedelta(hours=JWT_EXPIRY_HOURS)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject_id,
        "role": role,
        "ver": int(token_version),
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM), int(expires_delta.total_seconds())


def decode_token(token: str) -> Optional[tuple[str, str]]:
    """
    Return (subject_id, role, token_version), or None if the token is invalid
    or expired. The version is what the lookups below compare against the
    account's own counter, which is how a token gets taken back.

    A token issued before roles existed carries no claim; those are patients,
    which is what every account was at the time. Anything else — a role this
    build does not know, or a non-string — is refused rather than guessed at,
    since guessing here would be guessing which table to trust someone against.
    """
    try:
        # algorithms is pinned: without it a token could name "none" and be
        # accepted unsigned.
        claims = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        return None

    role = claims.get("role", ROLE_PATIENT)
    if role not in (ROLE_PATIENT, ROLE_CLINICIAN):
        return None

    # The version claim is what makes a token revocable. One issued before this
    # existed carries none, and is read as version 0 — the value every account
    # starts at, so nothing is signed out merely by this shipping.
    ver = claims.get("ver", 0)
    if not isinstance(ver, int) or isinstance(ver, bool):
        return None
    return sub, role, ver


# ---------------------------------------------------------------------------
# Password reset tokens
# ---------------------------------------------------------------------------

# Long enough that guessing is not a strategy, short enough to survive being
# pasted out of an email client that has wrapped the line.
RESET_TOKEN_BYTES = 32
RESET_TOKEN_TTL_MINUTES = int(os.getenv("RESET_TOKEN_TTL_MINUTES", "60"))


def new_reset_token() -> tuple[str, str]:
    """
    Return (token, hash). Only the hash is stored.

    SHA-256 rather than Argon2: this is a 256-bit random value with an hour to
    live, not a human-chosen password, so there is nothing for a slow hash to
    protect against — and the verify happens on an unauthenticated endpoint
    where a 85 ms hash would be a denial-of-service lever.
    """
    token = secrets.token_urlsafe(RESET_TOKEN_BYTES)
    return token, hash_reset_token(token)


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def reset_token_is_live(account) -> bool:
    """True when the account has a reset outstanding that has not expired."""
    if not account.reset_token_hash or account.reset_requested_at is None:
        return False
    requested = account.reset_requested_at
    if requested.tzinfo is None:
        requested = requested.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - requested
    return age <= timedelta(minutes=RESET_TOKEN_TTL_MINUTES)


def revoke_tokens(account) -> None:
    """
    Take back every token issued to this account so far.

    Called on sign-out, on a password change, and on a completed reset. The
    caller commits, and anything it hands back afterwards must be issued from
    the new counter value.
    """
    account.token_version = (account.token_version or 0) + 1


# ---------------------------------------------------------------------------
# Request dependencies
# ---------------------------------------------------------------------------

# auto_error=False on both: a missing header should produce this module's own
# 401 shape, not HTTPBearer's, and the optional variant must not raise at all.
_required_scheme = HTTPBearer(auto_error=False, description="Bearer token from POST /auth/login")
_optional_scheme = HTTPBearer(auto_error=False)

_UNAUTHORISED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Sign in to continue.",
    headers={"WWW-Authenticate": "Bearer"},
)


def _still_valid(account, token_version: int) -> bool:
    """
    False once the account has been bumped past the version this token carries.

    Signing out, changing a password and completing a reset all increment the
    counter, which takes back every token issued before. Equality rather than a
    comparison: a token from the future is as wrong as one from the past, and
    both mean something has gone astray.
    """
    return token_version == (getattr(account, "token_version", 0) or 0)


def _lookup(db: Session, token: str) -> Optional[Patient]:
    decoded = decode_token(token)
    if decoded is None:
        return None
    subject_id, role, token_version = decoded
    # A clinician's token must not open a patient's door, however valid it is.
    if role != ROLE_PATIENT:
        return None
    # A token can outlive the account it names — deletion does not reach back
    # and revoke tokens already issued.
    patient = db.get(Patient, subject_id)
    if patient is None or not _still_valid(patient, token_version):
        return None
    return patient


def _lookup_clinician(db: Session, token: str) -> Optional[Clinician]:
    decoded = decode_token(token)
    if decoded is None:
        return None
    subject_id, role, token_version = decoded
    if role != ROLE_CLINICIAN:
        return None
    clinician = db.get(Clinician, subject_id)
    if clinician is None or not _still_valid(clinician, token_version):
        return None
    return clinician


def current_patient(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_required_scheme),
    db: Session = Depends(get_db),
) -> Patient:
    """Require a signed-in patient. Raises 401 otherwise."""
    if creds is None:
        raise _UNAUTHORISED
    patient = _lookup(db, creds.credentials)
    if patient is None:
        raise _UNAUTHORISED
    return patient


def optional_patient(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_optional_scheme),
    db: Session = Depends(get_db),
) -> Optional[Patient]:
    """
    Resolve a patient if one is signed in, otherwise None.

    Used by /analyse-xray, which stays open to guests: login.html offers a guest
    route, and an anonymous analysis is still worth returning. A bad or expired
    token is treated as absent rather than rejected — the alternative is that a
    stale token in an old tab blocks a guest from using the app at all.
    """
    if creds is None:
        return None
    return _lookup(db, creds.credentials)


def current_clinician(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_required_scheme),
    db: Session = Depends(get_db),
) -> Clinician:
    """Require a signed-in clinician. Raises 401 otherwise."""
    if creds is None:
        raise _UNAUTHORISED
    clinician = _lookup_clinician(db, creds.credentials)
    if clinician is None:
        raise _UNAUTHORISED
    return clinician


def find_clinician_by_email(db: Session, email: str) -> Optional[Clinician]:
    return db.scalar(select(Clinician).where(Clinician.email == normalise_email(email)))


def touch_last_login(db: Session, account) -> None:
    """Works for either account type; both carry last_login_at."""
    account.last_login_at = utcnow()
    db.commit()


def find_by_email(db: Session, email: str) -> Optional[Patient]:
    return db.scalar(select(Patient).where(Patient.email == normalise_email(email)))
