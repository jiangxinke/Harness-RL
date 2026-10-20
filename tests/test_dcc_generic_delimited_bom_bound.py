"""The returned original UTF-8 BOM counts toward the decoded source bound."""
import pytest
from slime_plugins.dcc import tools


def _csv_text_chars(total):
    parts = ['Country,note\n']
    remaining = total - len(parts[0])
    while remaining > 32771:
        parts.append('A,' + 'x' * 32768 + '\n')
        remaining -= 32771
    assert remaining >= 3
    parts.append('A,' + 'x' * (remaining - 3) + '\n')
    return ''.join(parts)


def test_generic_csv_bom_is_preserved_at_total_source_character_limit():
    text = '\ufeff' + _csv_text_chars(999_999)
    assert len(text) == 1_000_000
    assert tools.extract_page(text.encode(), 'application/octet-stream') == text


def test_generic_csv_bom_counts_toward_total_source_character_limit():
    text = '\ufeff' + _csv_text_chars(1_000_000)
    assert len(text) == 1_000_001
    with pytest.raises(ValueError, match='unsupported page content type'):
        tools.extract_page(text.encode(), 'application/octet-stream')
