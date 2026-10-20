"""Offline navigation recovery, safe browser candidates, and evidence boundaries."""
import asyncio
from copy import deepcopy
import json

import pytest

from slime_plugins.dcc.browser_reader import BROWSER_READER_VERSION
from slime_plugins.dcc.web_navigation import (
    WEB_NAVIGATION_LIMITS, WEB_NAVIGATION_PREFIX, WEB_NAVIGATION_VERSION,
    WebNavigationLedger, validate_web_navigation_records, web_navigation_configuration,
)

PAGE = "https://fixture.example/page"
EXPORT = "https://fixture.example/table.csv"
API = "https://fixture.example/collection?country=A&year=2020"


def event(tool, observation, **arguments):
    return dict(tool=tool, observation=observation, arguments=arguments)


def located(url=PAGE, start=0, text="alpha", **metadata):
    return dict(url=url, start=start, end=start+len(text), text=text, **metadata)


def browser(rows, **metadata):
    return dict(method="visible_dom", browser_reader_version=BROWSER_READER_VERSION,
                observed_json_urls=rows, **metadata)


def observed(url=API, **metadata):
    return dict(url=url, status=200, navigation_only=True, **metadata)


def snapshot(ledger):
    rendered=ledger.render()
    assert len(rendered)<=WEB_NAVIGATION_LIMITS["max_chars"]
    return json.loads(rendered.removeprefix(WEB_NAVIGATION_PREFIX))


def frontier(ledger):
    return {(row["url"],row["start"]):row for row in snapshot(ledger)["frontier"]}


def test_actual_403_is_failed_and_untried_html_precedes_failed_structured_export():
    ledger=WebNavigationLedger()
    ledger.observe_events([event("web_search",dict(results=[dict(url=EXPORT),dict(url=PAGE)])),
        event("fetch_url",dict(url=EXPORT,error="HTTP rejection",http_status=403,retryable=False),url=EXPORT)])
    data=snapshot(ledger)
    assert data["frontier"][0]["url"]==PAGE
    failure=frontier(ledger)[(EXPORT,0)]
    assert failure["state"]=="failed" and failure["http_status"]==403 and failure["retryable"] is False
    assert data["recent_windows"]==[]


def test_constructed_failed_url_is_metadata_only_and_search_error_does_not_fail_a_fetch():
    ledger=WebNavigationLedger()
    ledger.observe_events([event("web_search",dict(error="upstream search unavailable"),url=PAGE),
        event("fetch_url",dict(url=API,error="HTTP rejection",http_status=403,retryable=False),url=API)])
    assert set(frontier(ledger))=={(API,0)}
    assert frontier(ledger)[(API,0)]["state"]=="failed"
    assert not snapshot(ledger)["recent_windows"]


@pytest.mark.parametrize("status,retryable",[(429,True),(503,True),(None,True),(403,False)])
def test_later_actual_success_restores_read_and_next_window_without_permanent_ban(status,retryable):
    ledger=WebNavigationLedger()
    events=[event("web_search",dict(results=[dict(url=EXPORT)])),
        event("fetch_url",dict(url=EXPORT,error="failure",http_status=status,retryable=retryable),url=EXPORT)]
    ledger.observe_events(events)
    assert frontier(ledger)[(EXPORT,0)]["state"]=="failed"
    events.append(event("fetch_url",located(EXPORT,next_start=5),url=EXPORT))
    ledger.observe_events(events)
    rows=frontier(ledger)
    assert rows[(EXPORT,0)]["state"]=="read" and rows[(EXPORT,5)]["state"]=="unread"
    assert all("http_status" not in row and "retryable" not in row for row in rows.values())
    events.append(event("fetch_url",located(EXPORT,start=5),url=EXPORT,start=5))
    ledger.observe_events(events)
    assert frontier(ledger)[(EXPORT,5)]["state"]=="read"
    assert len(snapshot(ledger)["recent_windows"])==2


def test_unlocated_success_clears_failure_but_never_fabricates_a_read_span():
    ledger=WebNavigationLedger()
    ledger.observe_events([event("fetch_url",dict(url=EXPORT,error="unavailable"),url=EXPORT),
        event("fetch_url",dict(url=EXPORT,text="actual text",start=True,end=999),url=EXPORT)])
    assert frontier(ledger)[(EXPORT,0)]["state"]=="unread"
    assert snapshot(ledger)["recent_windows"][0]["start"] is None


def test_invalid_offset_failure_does_not_poison_an_already_read_other_window():
    ledger=WebNavigationLedger()
    ledger.observe_events([event("web_search",dict(results=[dict(url=PAGE)])),event("fetch_url",located()),
        event("fetch_url",dict(url=PAGE,error="Read offset is beyond this page"),url=PAGE,start=999)])
    assert frontier(ledger)[(PAGE,0)]["state"]=="read"
    assert frontier(ledger)[(PAGE,999)]["state"]=="failed"


def test_errors_keep_only_safe_typed_status_retryability_and_never_echo_credentials():
    secret="credential"+"-value"
    ledger=WebNavigationLedger()
    ledger.observe_events([event("fetch_url",dict(url=EXPORT,error=secret,http_status=True,retryable=1),url=EXPORT)])
    data=snapshot(ledger)
    assert secret not in ledger.render()
    assert data["recent_errors"][0]["http_status"] is None
    assert data["recent_errors"][0]["retryable"] is None
    assert frontier(ledger)[(EXPORT,0)]["state"]=="failed"


def test_real_browser_json_navigation_metadata_is_unread_and_body_or_top_level_urls_are_ignored():
    ledger=WebNavigationLedger()
    ledger.observe_events([event("fetch_url",located(text="https://body.example/not-read",browser_read=browser([
        observed(API),observed(API+"#same-request")]),observed_json_urls=[observed("https://top.example/not-read")]))])
    rows=frontier(ledger)
    assert set(rows)=={(API,0)}
    assert rows[(API,0)]["state"]=="unread" and rows[(API,0)]["kind"]=="browser-json-navigation"
    assert rows[(API,0)]["priority"]=="structured-export-or-docs"
    assert [row["url"] for row in snapshot(ledger)["recent_windows"]]==[PAGE]
    assert "body.example" not in ledger.render() and "top.example" not in ledger.render()


@pytest.mark.parametrize("mutation",["status","bool-status","navigation-only","version","method","error","empty-body"])
def test_unqualified_browser_metadata_never_creates_candidates(mutation):
    row=observed();metadata=browser([row]);observation=located(browser_read=metadata)
    if mutation=="status":row["status"]=403
    elif mutation=="bool-status":row["status"]=True
    elif mutation=="navigation-only":row["navigation_only"]=1
    elif mutation=="version":metadata["browser_reader_version"]="unsupported"
    elif mutation=="method":metadata["method"]="http_fallback"
    elif mutation=="error":observation["error"]="failed fetch"
    else:observation["text"]=""
    ledger=WebNavigationLedger();ledger.observe_events([event("fetch_url",observation,url=PAGE)])
    assert all(row["url"]!=API for row in snapshot(ledger)["frontier"])


@pytest.mark.parametrize("bad",[
    "javascript:alert(1)","file:///private","https://"+"user:password"+"@fixture.example/data", "https://@fixture.example/data",
    "https://fixture.example:bad/data","https://fixture.example/data\nnext","https://fixture.example/"+"a"*2050,
    "https://fixture.example/data?access_token=private", "https://fixture.example/data?%61pi_key=private",
    "https://fixture.example/data?X-Amz-Signature=private", "https://fixture.example/data?key=private",
    "https://fixture.example/data?next=https%3A%2F%2Fuser%3Apassword%40other.example",
])
def test_sensitive_or_invalid_browser_url_is_omitted_from_summary_only(bad):
    ledger=WebNavigationLedger()
    ledger.observe_events([event("fetch_url",located(browser_read=browser([observed(bad)])))])
    assert not snapshot(ledger)["frontier"]
    assert bad not in ledger.render()


def test_fragment_aliases_share_failure_and_recovery_but_distinct_queries_remain_distinct():
    other=API+"&page=2"
    ledger=WebNavigationLedger()
    events=[event("fetch_url",located(browser_read=browser([observed(API+"#one"),observed(API+"#two"),observed(other)]))),
        event("fetch_url",dict(url=API,error="429",http_status=429,retryable=True),url=API+"#two")]
    ledger.observe_events(events)
    rows=frontier(ledger)
    assert set(rows)=={(API,0),(other,0)}
    assert rows[(API,0)]["state"]=="failed" and rows[(other,0)]["state"]=="unread"
    events.append(event("fetch_url",located(API+"#three",next_start=5),url=API))
    ledger.observe_events(events)
    assert frontier(ledger)[(API,0)]["state"]=="read"
    assert frontier(ledger)[(API,5)]["state"]=="unread"


def test_candidates_and_snapshot_remain_bounded_and_incremental_under_truncation():
    urls=["https://fixture.example/data/"+str(i)+"?column="+"a"*500 for i in range(100)]
    ledger=WebNavigationLedger()
    events=[event("fetch_url",located(browser_read=browser([observed(url) for url in urls],observed_json_urls_truncated=True)))]
    ledger.observe_events(events)
    data=snapshot(ledger)
    assert len(ledger.frontier)==WEB_NAVIGATION_LIMITS["max_frontier"]
    assert data["evicted_records"]["frontier"]==52
    assert data["omitted_from_snapshot"]["frontier"]>0
    assert all(row["url"] in urls for row in data["frontier"])
    before=ledger.render();ledger.observe_events(events);assert ledger.render()==before


def test_enabled_version_changes_but_disabled_configuration_is_exact_legacy_shape():
    assert WEB_NAVIGATION_VERSION=="observed-web-frontier-ledger-v2"
    assert web_navigation_configuration(False)==dict(enabled=False,version=None,scope=None,limits=None,candidate_sources=None)
    configuration=web_navigation_configuration(True)
    assert "browser_read.observed_json_urls" in configuration["candidate_sources"]
    from slime_plugins.dcc.web_structured import web_structured_actions_configuration,web_structured_configuration
    record=dict(web_navigation_ledger=configuration,web_structured_final=web_structured_configuration(True),
        web_structured_actions=web_structured_actions_configuration(True),backend="vllm")
    assert validate_web_navigation_records([record,deepcopy(record)])==configuration
    legacy=deepcopy(record);legacy["web_navigation_ledger"]["version"]="observed-web-frontier-ledger-v1"
    with pytest.raises(ValueError,match="supported configuration"):validate_web_navigation_records([legacy])


def test_real_tools_browser_metadata_does_not_make_api_contents_or_citation_evidence(monkeypatch):
    from slime_plugins.dcc import tools
    from slime_plugins.dcc import search_provider
    monkeypatch.setattr(search_provider,"wikipedia_page",lambda url:None)
    monkeypatch.setattr(tools,"_download_public",lambda url:(b"<html>ordinary page</html>","text/html","utf-8"))
    class Reader:
        async def read(self,url):
            return dict(text="Visible page",truncated=False,total_characters=12,final_url=url,
                outgoing_links=[],observed_json_urls=[observed()],
                browser_runtime=dict(version=BROWSER_READER_VERSION,network={}))
    web=tools.WebTools(browser_reader=Reader())
    result=asyncio.run(web.fetch_url(PAGE))
    assert result["browser_read"]["observed_json_urls"]==[observed()]
    ledger=WebNavigationLedger();ledger.observe_events([event("fetch_url",result,url=PAGE)])
    assert frontier(ledger)[(API,0)]["state"]=="unread"
    assert API not in web.pages and API not in web.sources
    assert web.citation_validity([dict(citations=[API])])==0
    assert web.citation_validity([dict(citations=[PAGE])])==1


@pytest.mark.parametrize("nested", [
    "https://alice@public.example/data",
    "https://@public.example/data",
    "https://public.example/data?api_key=private",
    "https://public.example/data?X-Amz-Signature=private",
])
def test_nested_url_userinfo_and_sensitive_query_values_are_omitted(nested):
    from urllib.parse import urlencode

    for depth in range(1, 4):
        url = nested
        for _ in range(depth):
            url = PAGE + "?" + urlencode({"next": url})
        ledger = WebNavigationLedger()
        ledger.observe_events([event("web_search", dict(results=[dict(url=url)]))])
        assert not snapshot(ledger)["frontier"]


@pytest.mark.parametrize("nested", [
    "https://public.example/data?country=CA&year=2024",
    "ftp://public.example/data?format=csv",
])
def test_public_nested_url_queries_remain_literal(nested):
    from urllib.parse import urlencode

    url = PAGE + "?" + urlencode({"next": nested, "offset": 10, "limit": 50})
    ledger = WebNavigationLedger()
    ledger.observe_events([event("web_search", dict(results=[dict(url=url)]))])
    assert snapshot(ledger)["frontier"][0]["url"] == url
