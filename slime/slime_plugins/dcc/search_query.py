"""Conservative, explicit source scoping for generic public search queries."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
import urllib.parse


# Bare filenames are common search terms. Explicit site:/domain: operators and
# absolute URLs do not need this ambiguity guard.
_FILE_SUFFIXES = set("csv doc docx gz htm html ipynb jpeg jpg js json jsonl log md out parquet pdf png py rst sh sql tar tex ts tsv txt xls xlsx xml yaml yml zip".split())


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

    Ordinary text is returned byte-for-byte. OR/multiple positive scopes,
    parentheses and quoted operators retain the provider's own semantics and
    receive no local scope filter. No domain is inferred from an entity name.
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
        absolute = candidate.startswith(("https://", "http://"))
        bare = not operator and not absolute and "/" not in candidate and "@" not in candidate and ":" not in candidate
        if not operator and not absolute and (not bare or candidate.rsplit(".", 1)[-1].lower() in _FILE_SUFFIXES):
            continue
        # Arbitrary dotted words can be version strings or filenames. Keep
        # bare-host inference conservative; explicit operators/URLs support
        # every valid IDNA suffix, including newer generic top-level domains.
        if bare and not operator and not re.fullmatch(r"[a-z]{2}|com|org|net|edu|gov|int|mil", candidate.rsplit(".", 1)[-1], re.IGNORECASE):
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
