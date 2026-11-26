"""Process-wide TLS configuration for Python Requests used by MOEX ISS loaders.

Requests verifies TLS certificates by default. If the runtime is behind a TLS-
inspecting proxy, set MOEX_CA_BUNDLE to the PEM file containing that proxy's
root/intermediate CA certificate(s). The custom CA is appended to certifi's
public roots; certificate verification is never disabled.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _configure_requests_ca() -> None:
    try:
        import certifi
        import requests.adapters
    except ImportError:
        # Requests/certifi are application dependencies; don't prevent unrelated
        # tooling from starting if this optional startup hook runs without them.
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
        # Preserve public Web PKI roots while adding the private CA used by a
        # corporate proxy/network gateway.
        combined = Path(tempfile.gettempdir()) / (
            f"bond_date_requests_ca_{os.getpid()}.pem"
        )
        combined.write_bytes(
            certifi_bundle.read_bytes()
            + b"\n"
            + custom_path.read_bytes()
            + b"\n"
        )
        ca_path = str(combined)
    else:
        # Explicitly pin Requests to certifi's maintained Mozilla CA bundle.
        ca_path = str(certifi_bundle)

    # This also covers Sessions with trust_env=False (e.g. news_service.py),
    # for which REQUESTS_CA_BUNDLE environment overrides are otherwise ignored.
    requests.adapters.DEFAULT_CA_BUNDLE_PATH = ca_path


_configure_requests_ca()
