"""Process-wide Requests configuration for MOEX ISS.

MOEX_SSL_VERIFY defaults to false as a temporary compatibility workaround for
networks that replace MOEX's TLS certificate with a private/self-signed CA.
Set MOEX_SSL_VERIFY=true to restore normal certificate validation and use the
configured certifi/custom CA bundle.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _configure_requests_ca() -> None:
    try:
        import certifi
        import requests.adapters
        import requests.sessions
    except ImportError:
        return

    custom_ca = (
        os.environ.get("MOEX_CA_BUNDLE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    certifi_bundle = Path(certifi.where())

    if custom_ca:
        custom_path = Path(custom_ca).expanduser()
        if not custom_path.is_file():
            raise RuntimeError(
                "MOEX TLS: CA bundle not found. Check MOEX_CA_BUNDLE: "
                f"{custom_path}"
            )
        combined = Path(tempfile.gettempdir()) / (
            f"bond_date_requests_ca_{os.getpid()}.pem"
        )
        combined.write_bytes(
            certifi_bundle.read_bytes()
            + b"\\n"
            + custom_path.read_bytes()
            + b"\\n"
        )
        ca_path = str(combined)
    else:
        ca_path = str(certifi_bundle)

    requests.adapters.DEFAULT_CA_BUNDLE_PATH = ca_path
    os.environ["REQUESTS_CA_BUNDLE"] = ca_path

    # Apply one consistent policy to every requests.get/post/Session call in
    # the repository, rather than relying on each MOEX loader to pass verify.
    if getattr(requests.sessions.Session.request, "_moex_tls_policy", False):
        return

    original_request = requests.sessions.Session.request

    def request_with_moex_tls(self, method, url, *args, **kwargs):
        hostname = (urlparse(str(url)).hostname or "").lower()
        if hostname == "iss.moex.com":
            # Explicit override is available for deployments with a trusted CA.
            kwargs["verify"] = _env_bool("MOEX_SSL_VERIFY", default=False)
        return original_request(self, method, url, *args, **kwargs)

    request_with_moex_tls._moex_tls_policy = True
    requests.sessions.Session.request = request_with_moex_tls


_configure_requests_ca()
