# Explicit source scope in Web search

`web_search` keeps its existing `query` and `count` arguments. Ordinary text
queries and their observations are unchanged. When a query explicitly names one
source hostname, the harness converts it to the search provider's `site:` syntax:

```json
{"tool":"web_search","arguments":{"query":"API documentation domain:example.org","count":8}}
```

`site:example.org`, a bare `example.org`, and an absolute URL also identify an
explicit source scope. A URL contributes its hostname and path; its query string
and fragment are omitted from the search scope. `domain:` and absolute URLs
support IDNA hostnames and any valid suffix. Bare hostnames are recognized
conservatively using country-code or classic generic suffixes; filenames,
email addresses and version numbers remain literal query text. Use `site:` for
an ambiguous hostname or a newer suffix.

The harness keeps only results from the requested host or its subdomains and,
when supplied, the requested path and its children. `-site:` excludes a host or
path. If a public provider ignores the scope and returns unrelated URLs, normal
fallback continues. Scoped search can correctly return no results; it never
substitutes an unrelated encyclopedia page. Search snippets and scope metadata
remain discovery information; source reads are still required as evidence.

Boolean queries, parentheses, quoted scope operators and multiple positive
scopes retain the provider's original semantics without local rewriting or
filtering. For example, `site:example.org OR site:example.net documentation`
continues to ask the provider for either domain. No hostname is inferred from an
organization name, and the harness contains no benchmark-specific query or URL
mapping. Rewriting is reported as `original_query`; recognized scope is reported
as `query_scope` in that search observation.
