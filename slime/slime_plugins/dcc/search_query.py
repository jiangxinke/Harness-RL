"""Conservative, explicit source scoping for generic public search queries."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
import urllib.parse


def _hostname(value):
    try:
        host = value.rstrip(".").encode("idna").decode("ascii").lower()
    except (UnicodeError, AttributeError):
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return None
    labels = host.split(".")
    if len(labels) < 2 or len(host) > 253 or any(
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels
    ) or not re.fullmatch(r"[a-z]{2,63}|xn--[a-z0-9-]+", labels[-1]):
        return None
    return host


@dataclass(frozen=True)
class SearchSite:
    host: str
    path: str = ""

    def matches(self, url):
        try:
            parsed = urllib.parse.urlsplit(url)
            host = _hostname(parsed.hostname)
        except (TypeError, ValueError):
            return False
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or not host:
            return False
        if host != self.host and not host.endswith("." + self.host):
            return False
        path = urllib.parse.unquote(parsed.path).rstrip("/")
        return not self.path or path == self.path or path.startswith(self.path + "/")

    def operator(self):
        return "site:" + self.host + self.path


def _site(value):
    try:
        parsed = urllib.parse.urlsplit(value if "://" in value else "https://" + value)
        host = _hostname(parsed.hostname)
        port = parsed.port
    except (ValueError, TypeError):
        return None
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or port is not None or not host:
        return None
    path = urllib.parse.unquote(parsed.path).rstrip("/")
    if any(character.isspace() for character in path):
        return None
    return SearchSite(host, path)


@dataclass(frozen=True)
class SearchScope:
    site: SearchSite | None
    excluded: tuple[SearchSite, ...] = ()

    def filter(self, hits):
        return [hit for hit in hits if (self.site is None or self.site.matches(hit.get("url", "")))
                and not any(site.matches(hit.get("url", "")) for site in self.excluded)]

    def metadata(self):
        return {"site": self.site.operator()[5:] if self.site else None,
                "excluded_sites": [site.operator()[5:] for site in self.excluded]}


def prepare_search_query(query):
    """Rewrite one unambiguous explicit host; leave complex provider syntax alone.

    Only site:/domain: operators and absolute HTTP(S) URLs request a scope.
    Bare dotted text, including hostnames and technical identifiers, is kept
    byte-for-byte. OR/multiple positive scopes, parentheses and quoted operators
    retain the provider's semantics and receive no local scope filter.
    """
    if re.search(r"[()|]|\b(?:OR|AND|NOT)\b", query, re.IGNORECASE):
        return query, None
    tokens = list(re.finditer(r"\S+", query))
    positive, negative, changes = [], [], []
    for token in tokens:
        value = token.group()
        # A quoted hostname may be a literal phrase, not an operator.
        if '"' in value or "'" in value:
            if re.search(r"(?:site|domain):", value, re.IGNORECASE):
                return query, None
            continue
        operator = re.fullmatch(r"(-?)(?:site|domain):(.+)", value, re.IGNORECASE)
        excluded = bool(operator and operator[1])
        candidate = operator[2] if operator else value
        absolute = candidate.lower().startswith(("https://", "http://"))
        # Dotted tokens can be identifiers, filenames or literal domain names.
        # A suffix cannot establish that the user requested a source filter.
        if not operator and not absolute:
            continue
        site = _site(candidate)
        if site is None:
            if operator:
                return query, None
            continue
        (negative if excluded else positive).append(site)
        changes.append((token.start(), token.end(), ("-" if excluded else "") + site.operator()))
    if len(positive) > 1 or not positive and not negative:
        return query, None
    rewritten = query
    for start, end, replacement in reversed(changes):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten, SearchScope(positive[0] if positive else None, tuple(negative))
