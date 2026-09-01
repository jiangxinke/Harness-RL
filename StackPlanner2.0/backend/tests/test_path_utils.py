from app.gateway import path_utils
from deerflow.config.paths import Paths


def test_resolve_thread_virtual_path_reads_legacy_thread_artifact_for_authenticated_user(tmp_path, monkeypatch):
    paths = Paths(tmp_path)
    thread_id = "thread-legacy"
    legacy_file = paths.sandbox_outputs_dir(thread_id) / "report.md"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("legacy report", encoding="utf-8")

    monkeypatch.setattr(path_utils, "get_paths", lambda: paths)
    monkeypatch.setattr(path_utils, "get_effective_user_id", lambda: "user-1")

    resolved = path_utils.resolve_thread_virtual_path(thread_id, "mnt/user-data/outputs/report.md")

    assert resolved == legacy_file.resolve()


def test_resolve_thread_virtual_path_prefers_user_scoped_artifact(tmp_path, monkeypatch):
    paths = Paths(tmp_path)
    thread_id = "thread-scoped"
    legacy_file = paths.sandbox_outputs_dir(thread_id) / "report.md"
    scoped_file = paths.sandbox_outputs_dir(thread_id, user_id="user-1") / "report.md"
    legacy_file.parent.mkdir(parents=True)
    scoped_file.parent.mkdir(parents=True)
    legacy_file.write_text("legacy", encoding="utf-8")
    scoped_file.write_text("scoped", encoding="utf-8")

    monkeypatch.setattr(path_utils, "get_paths", lambda: paths)
    monkeypatch.setattr(path_utils, "get_effective_user_id", lambda: "user-1")

    resolved = path_utils.resolve_thread_virtual_path(thread_id, "/mnt/user-data/outputs/report.md")

    assert resolved == scoped_file.resolve()
