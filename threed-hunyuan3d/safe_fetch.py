"""Fetch an image from a job input: a data URL, raw base64, or a public https:// URL.

Job inputs can arrive from an LLM agent (Hermes), not just a human, so a URL in the input must
never be able to reach an internal address -- cloud metadata endpoints (169.254.169.254),
localhost, RFC1918 ranges, link-local, etc. ("SSRF"). This file is the one shared helper both
Hunyuan workers use, copied verbatim into each image rather than packaged, so neither depends on
the other or on a shared base layer. build_check.py COPYs it in and imports + exercises it at
build time, on literal addresses rather than live DNS, so the checks run with no network on the
ACR build agent.

    fetch_image_bytes(src: str, *, max_bytes=64*1024*1024, timeout=60) -> bytes

Raises ValueError with a message safe to put straight into a job's {"error": ...} -- it never
echoes fetched bytes or a remote server's response body back to the caller.
"""
from __future__ import annotations

import base64
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit

MAX_BYTES_DEFAULT = 64 * 1024 * 1024
TIMEOUT_DEFAULT = 60.0


def _is_public(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        or ip.is_multicast or ip.is_unspecified or not ip.is_global
    )


def _resolve_and_validate(host: str, port: int) -> str:
    """Every address a hostname resolves to must be public, or the host is refused outright -- a
    name that resolves to both a public and a private address (the classic DNS-rebinding shape,
    and not rare by accident either) is not made safe by one good answer among several."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host: {e}") from None
    addrs = {info[4][0] for info in infos}
    if not addrs:
        raise ValueError("host did not resolve to any address")
    for addr in addrs:
        if not _is_public(addr):
            raise ValueError("host resolves to a non-public address; refused")
    return sorted(addrs)[0]  # deterministic pick, not whatever order the resolver happened to list


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects to a pre-resolved, pre-validated IP, but still TLS-handshakes (SNI) and lets the
    Host header default to the original hostname -- otherwise this would both fail certificate
    validation against virtual-hosted/CDN-fronted sites and reopen the exact gap it exists to
    close (validate a name, then let the HTTP client resolve that same name again itself, which a
    DNS-rebinding attacker can answer differently the second time)."""

    def __init__(self, hostname: str, pinned_ip: str, port: int, timeout: float):
        super().__init__(hostname, port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        context = ssl.create_default_context()
        self.sock = context.wrap_socket(sock, server_hostname=self.host)


def _fetch_https(url: str, max_bytes: int, timeout: float) -> bytes:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ValueError(f"only https:// links are accepted, got {parts.scheme or '(none)'}://")
    if not parts.hostname:
        raise ValueError("url has no host")
    port = parts.port or 443
    ip = _resolve_and_validate(parts.hostname, port)
    path = parts.path or "/"
    if parts.query:
        path += f"?{parts.query}"
    conn = _PinnedHTTPSConnection(parts.hostname, ip, port, timeout)
    try:
        conn.request("GET", path, headers={"User-Agent": "hunyuan-worker/1.0"})
        resp = conn.getresponse()
        # http.client never auto-follows a redirect (unlike urllib.request/requests) -- a 3xx
        # comes back as a plain response here. Refuse it rather than following it: a followed
        # redirect is a second fetch this function would have to re-validate from scratch anyway,
        # and a silently-followed one is exactly how an SSRF check gets bypassed.
        if 300 <= resp.status < 400:
            raise ValueError("redirects are not followed; fetch the final URL directly")
        if resp.status >= 400:
            raise ValueError(f"fetch failed: HTTP {resp.status}")
        body = resp.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"response exceeds the {max_bytes} byte limit")
        return body
    finally:
        conn.close()


def fetch_image_bytes(src: str, *, max_bytes: int = MAX_BYTES_DEFAULT, timeout: float = TIMEOUT_DEFAULT) -> bytes:
    if not src:
        raise ValueError("no image provided")
    if src.startswith("data:"):
        try:
            _, b64 = src.split(",", 1)
        except ValueError:
            raise ValueError("malformed data: URL") from None
        try:
            return base64.b64decode(b64, validate=True)
        except Exception as e:
            raise ValueError(f"could not decode the data URL: {type(e).__name__}") from None
    if src.startswith("https://"):
        try:
            return _fetch_https(src, max_bytes, timeout)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 - never surface the underlying exception verbatim;
                                 # it can carry response/body content
            raise ValueError(f"could not fetch the image: {type(e).__name__}") from None
    if src.startswith("http://"):
        raise ValueError("http:// links are not accepted; use https:// or send the image inline")
    # anything else: assumed to be raw base64 with no data: prefix
    try:
        return base64.b64decode(src, validate=True)
    except Exception as e:
        raise ValueError(f"could not decode as base64: {type(e).__name__}") from None
