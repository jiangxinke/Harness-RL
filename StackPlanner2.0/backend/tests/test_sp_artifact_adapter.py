"""Tests for SP mid-term artifact adapter."""

from pathlib import Path

import pytest

from deerflow.sp.artifacts import SPArtifactAdapter, SPArtifactMetadata


def _state(tmp_path: Path) -> dict:
    return {
        "thread_data": {
            "outputs_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "outputs"),
        }
    }


def test_write_text_artifact_stores_body_in_outputs_and_returns_lightweight_refs(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)

    result = adapter.write_text_artifact(
        "# Report\n\nLarge body",
        artifact_type="report",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
        created_by="reporter",
        stage="reporting",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report"]
    file_path = Path(result.metadata.workspace_path)
    assert file_path.read_text(encoding="utf-8") == "# Report\n\nLarge body"
    assert ref["virtual_path"].startswith("/mnt/user-data/outputs/sp/report/")
    assert ref["artifact_url"].startswith("/api/threads/thread-1/artifacts/")
    assert ref["summary"] == "# Report Large body"
    assert ref["is_current"] is True
    assert "Large body" not in result.state_update
    assert result.state_update["artifacts"] == [ref["virtual_path"]]
    assert result.state_update["sp_current_report_version"] == "1"


def test_read_text_artifact_materializes_bounded_head_and_tail(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)
    body = "HEAD-MARKER\n" + ("middle evidence\n" * 200) + "TAIL-MARKER\n"
    written = adapter.write_text_artifact(
        body,
        artifact_type="research_observation",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
    )

    content, truncated = adapter.read_text_artifact(
        written.metadata.to_ref(),
        state={**state, **written.state_update},
        thread_id="thread-1",
        max_chars=240,
    )

    assert truncated is True
    assert content.startswith("HEAD-MARKER")
    assert "<artifact-content-truncated" in content
    assert content.endswith("TAIL-MARKER\n")
    assert len(content) <= 240


def test_read_text_artifact_rejects_non_thread_and_binary_paths(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)
    outputs = Path(state["thread_data"]["outputs_path"])
    outputs.mkdir(parents=True)
    binary = outputs / "image.png"
    binary.write_bytes(b"\x89PNG\x00\x01\x02")

    assert (
        adapter.read_text_artifact(
            {"virtual_path": "/mnt/user-data/outputs/image.png", "type": "generated_file"},
            state=state,
            thread_id="thread-1",
        )
        is None
    )
    assert (
        adapter.read_text_artifact(
            {"virtual_path": "/tmp/outside.md", "type": "report"},
            state=state,
            thread_id="thread-1",
        )
        is None
    )


def test_write_text_artifact_versions_current_ref_and_keeps_history(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)
    first = adapter.write_text_artifact("Outline v1", artifact_type="outline", state=state, thread_id="thread-1")
    second = adapter.write_text_artifact(
        "Outline v2",
        artifact_type="outline",
        state={**state, **first.state_update},
        thread_id="thread-1",
    )

    refs = second.state_update["sp_current_artifact_refs"]
    assert refs["outline"]["version"] == 2
    assert [item["version"] for item in refs["_history"] if item["type"] == "outline"] == [1, 2]
    assert [item["is_current"] for item in refs["_history"] if item["type"] == "outline"] == [False, True]


def test_report_and_report_revision_share_one_version_lineage(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)
    report = adapter.write_text_artifact("Report v1", artifact_type="report", state=state, thread_id="thread-1")
    revision = adapter.write_text_artifact(
        "Report v2",
        artifact_type="report_revision",
        state={**state, **report.state_update},
        thread_id="thread-1",
        parent_artifact_ids=[report.metadata.artifact_id],
    )

    refs = revision.state_update["sp_current_artifact_refs"]
    assert report.metadata.version == 1
    assert revision.metadata.version == 2
    assert revision.metadata.parent_artifact_ids == [report.metadata.artifact_id]
    assert revision.state_update["sp_current_report_version"] == "2"
    assert report.metadata.artifact_id != revision.metadata.artifact_id
    assert [item["version"] for item in refs["_history"] if item["type"] in {"report", "report_revision"}] == [1, 2]
    assert refs["report"]["is_current"] is False
    assert refs["report_revision"]["is_current"] is True


def test_same_report_content_in_two_versions_has_distinct_artifact_ids(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)
    first = adapter.write_text_artifact("Same body", artifact_type="report", state=state, thread_id="thread-1")
    second = adapter.write_text_artifact(
        "Same body",
        artifact_type="report",
        state={**state, **first.state_update},
        thread_id="thread-1",
    )

    assert first.metadata.version == 1
    assert second.metadata.version == 2
    assert first.metadata.artifact_id != second.metadata.artifact_id


def test_remerging_same_artifact_keeps_history_entry_current():
    adapter = SPArtifactAdapter()
    metadata = SPArtifactMetadata(
        artifact_id="report-v1",
        type="report",
        version=1,
        virtual_path="/mnt/user-data/outputs/report-v1.md",
    )

    first = adapter._merge_ref({}, metadata)
    repeated = adapter._merge_ref(first, metadata)

    assert repeated["report"]["is_current"] is True
    assert repeated["_history"] == [repeated["report"]]


def test_migrate_legacy_midterm_state_externalizes_large_fields(tmp_path):
    adapter = SPArtifactAdapter()
    update = adapter.migrate_legacy_midterm_state(
        {
            "report_outline": "## Outline",
            "observations": ["observation one", "observation two"],
            "data_collections": [{"source": "search", "content": "result"}],
            "final_report": "# Final report",
            "memory_stack": "must stay out of artifact migration",
        },
        state=_state(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    refs = update["sp_current_artifact_refs"]
    assert set(refs) >= {"outline", "research_observation", "data_collection", "report_revision", "_history"}
    assert len(update["artifacts"]) == 4
    assert "report_outline" not in update
    assert "final_report" not in update
    assert "observations" not in update
    for virtual_path in update["artifacts"]:
        assert virtual_path.startswith("/mnt/user-data/outputs/sp/")


def test_metadata_ref_is_json_safe_and_excludes_workspace_path_from_thread_ref():
    metadata = SPArtifactMetadata(
        type="data_collection",
        workspace_path="/tmp/secret-host-path/data.json",
        virtual_path="/mnt/user-data/outputs/sp/data/data.json",
        metadata={"set": {"b", "a"}},
    )

    ref = metadata.to_ref()

    assert "workspace_path" not in ref
    assert ref["metadata"] == {"set": ["a", "b"]}
    assert ref["is_current"] is True


def test_bind_feedback_updates_current_ref_and_history():
    refs = {
        "report": {"artifact_id": "report-1", "type": "report", "feedback_entry_ids": []},
        "_history": [{"artifact_id": "report-1", "type": "report", "feedback_entry_ids": []}],
    }

    updated = SPArtifactAdapter().bind_feedback(refs, "feedback-1", artifact_ids=["report-1"])

    assert updated["report"]["feedback_entry_ids"] == ["feedback-1"]
    assert updated["_history"][0]["feedback_entry_ids"] == ["feedback-1"]


def test_write_text_artifact_requires_dr2_thread_context_when_no_outputs_path():
    with pytest.raises(ValueError, match="thread_data.outputs_path or thread_id"):
        SPArtifactAdapter().write_text_artifact("body", artifact_type="report", state={})


def test_register_existing_output_file_creates_ref_without_copying_body(tmp_path):
    adapter = SPArtifactAdapter()
    output_file = tmp_path / "threads" / "thread-1" / "user-data" / "outputs" / "generated" / "app.py"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("print('ready')\n", encoding="utf-8")
    result = adapter.register_existing_artifact(
        "/mnt/user-data/outputs/generated/app.py",
        artifact_type="generated_file",
        state=_state(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
        created_by="coder",
    )

    ref = result.state_update["sp_current_artifact_refs"]["generated_file"]
    assert ref["virtual_path"] == "/mnt/user-data/outputs/generated/app.py"
    assert ref["artifact_url"] == "/api/threads/thread-1/artifacts/mnt/user-data/outputs/generated/app.py"
    assert ref["metadata"]["registered_existing_file"] is True
    assert result.state_update["artifacts"] == ["/mnt/user-data/outputs/generated/app.py"]


def test_register_existing_result_text_adds_bounded_finalization_preview(tmp_path):
    adapter = SPArtifactAdapter()
    output_file = tmp_path / "threads" / "thread-1" / "user-data" / "outputs" / "hotel-result.txt"
    output_file.parent.mkdir(parents=True)
    output_file.write_text(
        "A酒店总费用: 5250.48\nB酒店总费用: 5108.50\nB便宜: 141.98\n",
        encoding="utf-8",
    )

    result = adapter.register_existing_artifact(
        "/mnt/user-data/outputs/hotel-result.txt",
        artifact_type="generated_file",
        state=_state(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
        created_by="coder",
        summary="Calculation completed.",
    )

    ref = result.state_update["sp_current_artifact_refs"]["generated_file"]
    assert ref["summary"] == "Calculation completed."
    assert ref["metadata"]["finalization_preview"] == ("A酒店总费用: 5250.48\nB酒店总费用: 5108.50\nB便宜: 141.98\n")
    assert ref["metadata"]["finalization_preview_truncated"] is False


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/not-thread-scoped.txt",
        "/mnt/user-data/outputs/../uploads/secret.txt",
        "relative/file.txt",
    ],
)
def test_register_existing_artifact_rejects_paths_outside_dr2_workspace(path):
    with pytest.raises(ValueError):
        SPArtifactAdapter().register_existing_artifact(path, artifact_type="generated_file", state={})
