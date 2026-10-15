"""Shared operator endpoint policy, with no DNS, environment or network I/O."""
from __future__ import annotations

import ipaddress
from urllib.parse import unquote, urlsplit, urlunsplit


def _parsed_http_url(value: str):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or any(char in value for char in ('\\', '?', '#'))):
        raise ValueError('Use an HTTP(S) URL without whitespace, credentials, query or fragment.')
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError('Service URL has an invalid host or port.') from exc
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or '%' in parsed.netloc
            or parsed.netloc.endswith(':') or (port is not None and not 1 <= port <= 65535)):
        raise ValueError('Use an HTTP(S) URL without credentials, query or fragment and with a valid port.')
    decoded_path = unquote(parsed.path)
    if (any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in decoded_path)
            or '\\' in decoded_path or '//' in decoded_path
            or any(part in {'.', '..'} for part in decoded_path.split('/'))):
        raise ValueError('Service URL path is not a valid API base path.')
    return parsed


def validate_http_service_url(value: str) -> str:
    """A manually managed Core may use remote HTTP; it receives no API key."""
    parsed = _parsed_http_url(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', ''))


def _loopback(host: str) -> bool:
    if host.lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_authoring_base_url(value: str) -> str:
    """Validate before reading a key; all remote authoring requires HTTPS."""
    parsed = _parsed_http_url(value)
    if not _loopback(parsed.hostname) and parsed.scheme != 'https':
        raise ValueError('Remote authoring services require HTTPS; HTTP is only allowed for loopback.')
    path = parsed.path.rstrip('/')
    if path.endswith(('/chat/completions', '/models', '/completions')):
        raise ValueError('Configure the authoring API base URL, not a chat or model-list endpoint.')
    if not path:
        path = '/v1'
    return urlunsplit((parsed.scheme, parsed.netloc, path, '', ''))


def authoring_chat_url(value: str) -> str:
    return validate_authoring_base_url(value) + '/chat/completions'


def validate_autostart_comfy_url(value: str, engine_port: int | None = None) -> str:
    """Owned Core binds IPv4 HTTP 127.0.0.1, never DNS, IPv6 or TLS."""
    parsed = _parsed_http_url(value)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path not in {'', '/'}:
        raise ValueError('Automatic Core startup requires http://127.0.0.1 with no API path; disable auto-start for other existing services.')
    port = parsed.port or 80
    if engine_port is not None and (type(engine_port) is not int or port != engine_port):
        raise ValueError('Automatic Core URL port must match engine_port.')
    return urlunsplit(('http', parsed.netloc, '', '', ''))
