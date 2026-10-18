"""Malformed extracted page text must remain usable by real Rust tokenizers."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from slime_plugins.dcc import tools

URL = "https://example.org/malformed.pdf"
EXTRACTED = "PDF source A\ud800B\udfffC. 中文 and 😀 remain intact."


@pytest.fixture
def tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    value = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    value.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    return value


@pytest.fixture
def malformed_pdf(monkeypatch):
    pypdf = pytest.importorskip("pypdf")
    calls = []

    def reader(stream):
        assert isinstance(stream, io.BytesIO)
        calls.append("pdf_text")
        return SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: EXTRACTED)])

    def download(url):
        calls.append(url)
        return b"%PDF-1.7\nstub", "application/pdf", "utf-8"

    monkeypatch.setattr(pypdf, "PdfReader", reader)
    monkeypatch.setattr(tools, "_download_public", download)
    return calls


def test_pdf_surrogates_reproduce_rust_error_then_real_reader_and_tokenizer_succeed(tokenizer, malformed_pdf):
    pytest.importorskip("torch")
    from slime_plugins.dcc.hf_policy import encode_messages

    raw = tools.extract_page(b"%PDF-1.7\nstub", "application/pdf")
    assert isinstance(raw, str) and raw == EXTRACTED
    raw_messages = [{"role": "user", "content": json.dumps({"text": raw}, ensure_ascii=False)}]
    assert all(isinstance(message["content"], str) for message in raw_messages)
    with pytest.raises(TypeError, match="TextEncodeInput"):
        encode_messages(tokenizer, raw_messages, 4096, allow_truncation=False)

    web = tools.WebTools(max_page_chars=18)
    first = asyncio.run(web.fetch_url(URL))
    expected = EXTRACTED.encode("utf-8", errors="replace").decode("utf-8")
    assert web.pages[URL] == expected
    assert first["text"] == expected[:18] and first["next_start"] == 18
    assert first["total_characters"] == len(expected)
    second = asyncio.run(web.fetch_url(URL, start=first["next_start"]))
    assert second["text"] == expected[18:36]
    assert malformed_pdf.count(URL) == 1
    assert web.pages[URL][36:] == expected[36:] and "😀" in web.pages[URL]
    assert not any(0xD800 <= ord(character) <= 0xDFFF for character in web.pages[URL])
    for page in (first, second):
        assert page["end"] - page["start"] == len(page["text"])
        messages = [{"role": "user", "content": "Tool observation (source material):\n" + json.dumps(page, ensure_ascii=False)}]
        assert encode_messages(tokenizer, messages, 4096, allow_truncation=False)


def test_valid_non_ascii_web_text_and_cuad_document_offsets_are_unchanged(monkeypatch):
    text = "中文 evidence 😀 café\nExact contract clause."
    monkeypatch.setattr(tools, "_download_public", lambda url: (text.encode(), "text/plain", "utf-8"))
    web = tools.WebTools(max_page_chars=100)
    result = asyncio.run(web.fetch_url("https://example.org/plain"))
    assert result["text"] == text and web.pages[result["url"]] == text
    document = "Contract \ud800 source clause."
    reader = tools.DocumentTools(document)
    observation = reader.read_document(0, len(document))
    assert observation == {"start": 0, "end": len(document), "text": document, "document_length": len(document)}
