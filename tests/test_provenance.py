"""Run provenance: -dirty git sha, playbook hash, bounded lessons."""
import subprocess

import pytest

from agent import logger, memory


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "agent").mkdir()
    (tmp_path / "run_logs").mkdir()
    (tmp_path / "agent" / "main.py").write_text("x = 1\n")
    (tmp_path / "agent" / "playbook.md").write_text("- a\n")
    (tmp_path / "run_logs" / "state.json").write_text("{}\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(logger, "_ROOT", str(tmp_path))
    monkeypatch.setattr(logger, "_GIT_SHA", None)
    return tmp_path


def _sha(monkeypatch):
    monkeypatch.setattr(logger, "_GIT_SHA", None)   # drop the per-run cache
    return logger.git_sha()


def test_clean_tree_has_plain_sha(repo, monkeypatch):
    sha = _sha(monkeypatch)
    assert sha and not sha.endswith("-dirty")


def test_run_artifacts_and_playbook_do_not_mark_dirty(repo, monkeypatch):
    (repo / "run_logs" / "state.json").write_text('{"iterations": 1}\n')
    (repo / "agent" / "playbook.md").write_text("- a\n- b\n")
    assert not _sha(monkeypatch).endswith("-dirty")


def test_modified_or_untracked_code_marks_dirty(repo, monkeypatch):
    (repo / "agent" / "main.py").write_text("x = 2\n")
    assert _sha(monkeypatch).endswith("-dirty")
    subprocess.run(["git", "checkout", "-q", "agent/main.py"], cwd=repo, check=True)
    assert not _sha(monkeypatch).endswith("-dirty")
    (repo / "agent" / "new_module.py").write_text("y = 1\n")
    assert _sha(monkeypatch).endswith("-dirty")


def test_not_a_repo_degrades_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "_ROOT", str(tmp_path))
    assert _sha(monkeypatch) == ""


def test_playbook_sha_tracks_content(tmp_path, monkeypatch):
    p = tmp_path / "playbook.md"
    monkeypatch.setattr(memory, "PLAYBOOK_PATH", str(p))
    assert memory.playbook_sha() == ""
    p.write_text("- a\n")
    a = memory.playbook_sha()
    p.write_text("- a\n- b\n")
    assert a and memory.playbook_sha() != a


def test_clip_lesson():
    assert memory.clip_lesson("short  lesson.") == "short lesson."
    long = "Finding one holds. " + "Detail sentence follows here. " * 40
    out = memory.clip_lesson(long)
    assert len(out) <= memory.LESSON_MAX_CHARS
    assert out.startswith("Finding one holds.") and out.endswith(".")
    blob = memory.clip_lesson("word " * 400)             # no sentence boundary
    assert len(blob) <= memory.LESSON_MAX_CHARS + 2 and blob.endswith("…")


def test_append_lesson_is_clipped(tmp_path, monkeypatch):
    p = tmp_path / "playbook.md"
    monkeypatch.setattr(memory, "PLAYBOOK_PATH", str(p))
    memory.append_lesson("Finding. " + "More detail here. " * 100)
    line = p.read_text().strip()
    assert len(line) <= memory.LESSON_MAX_CHARS + 2
    assert len(memory.load_playbook().splitlines()) == 1
