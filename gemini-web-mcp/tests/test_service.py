"""Service behaviour with a mocked Gemini backend: parallelism, isolation, deadline, budget, pauses,
edits, end_chat and the TTL sweep."""

from __future__ import annotations

import time

import pytest

from app.errors import AuthProblem, REFRESH_COOKIES
from app.service import UserError
from app.store import Store
from tests.conftest import FakeClock

pytestmark = pytest.mark.anyio


async def test_parallel_success_max_two_at_a_time(make_service, backend):
    service = make_service()
    outcome = await service.generate_images(["one", "two", "three"], "three cats", None)
    assert len(outcome.images) == 3 and not outcome.failures
    assert backend.max_active == 2
    assert len({img.image_id for img in outcome.images}) == 3
    for img in outcome.images:
        assert img.url.startswith(f"https://example.test/i/{outcome.session_id}/{img.image_id}.png")
        assert img.preview.is_file()
        assert service.image_path(outcome.session_id, img.image_id, "png").is_file()
    assert len(service.store.list_chats(outcome.session_id)) == 3
    assert service.store.images_since(0) == 3


async def test_several_images_for_one_prompt_are_all_kept(make_service):
    service = make_service()
    outcome = await service.generate_images(["a [multi3]"], "x", None)
    assert len(outcome.images) == 3


async def test_partial_failure_is_isolated_and_capped_at_two_attempts(make_service, backend):
    service = make_service()
    outcome = await service.generate_images(["ok one", "bad [fail]", "ok two"], "x", None)
    assert len(outcome.images) == 2
    assert [f.label for f in outcome.failures] == ["prompt 2"]
    assert "Failed after 2 attempt(s)" in outcome.failures[0].reason
    assert backend.attempts["bad [fail]"] == 2


async def test_transient_failure_is_retried_once(make_service, backend):
    service = make_service()
    outcome = await service.generate_images(["x [flaky]"], "x", None)
    assert len(outcome.images) == 1 and backend.attempts["x [flaky]"] == 2


async def test_deadline_returns_finished_and_marks_rest_timed_out(make_service):
    service = make_service(deadline_seconds=1.0)
    t0 = time.monotonic()
    outcome = await service.generate_images(["fast", "slow [slow]", "late [slow]"], "x", None)
    assert time.monotonic() - t0 < 5
    assert len(outcome.images) == 1
    assert sorted(f.label for f in outcome.failures) == ["prompt 2", "prompt 3"]
    assert all("Timed out" in f.reason for f in outcome.failures)
    assert service._reserved == 0  # reservations released on cancellation


async def test_refusal_reported_as_is_without_retry(make_service, backend):
    service = make_service()
    outcome = await service.generate_images(["p [refuse]"], "x", None)
    assert not outcome.images
    assert "I can't create that image." in outcome.failures[0].reason
    assert any("Do not rewrite a prompt" in n for n in outcome.notes)
    assert backend.attempts["p [refuse]"] == 1


async def test_hourly_limit_is_enforced_and_persisted(make_service, backend):
    service = make_service(max_images_per_hour=2)
    outcome = await service.generate_images(["a", "b", "c"], "x", None)
    assert len(outcome.images) == 2
    assert "Hourly image limit reached" in outcome.failures[0].reason
    sent = len(backend.sent)
    # Survives a restart: a fresh Store on the same database sees the same count.
    reopened = Store(service.settings.db_path)
    assert reopened.images_since(time.time() - 3600) == 2
    reopened.close()
    again = await service.generate_images(["d"], "x", outcome.session_id)
    assert not again.images and "Hourly image limit" in again.failures[0].reason
    assert len(backend.sent) == sent  # Gemini not called


async def test_auth_error_one_clear_message_and_no_retry_storm(make_service, backend):
    service = make_service()
    backend.ensure_error = AuthProblem(REFRESH_COOKIES)
    first = await service.generate_images(["a", "b", "c"], "x", None)
    assert not first.images and len(first.failures) == 3
    assert all("Refresh your cookies" in f.reason for f in first.failures)
    assert backend.ensure_calls == 1
    second = await service.generate_images(["d"], "x", first.session_id)
    assert "Refresh your cookies" in second.failures[0].reason
    assert backend.ensure_calls == 1 and backend.sent == []


async def test_rate_error_stops_generation_without_retry(make_service, backend):
    service = make_service()
    outcome = await service.generate_images(["x [rate]", "y", "z"], "x", None)
    assert backend.attempts["x [rate]"] == 1
    assert "z" not in backend.attempts  # never sent
    reasons = {f.label: f.reason for f in outcome.failures}
    assert "usage limit" in reasons["prompt 1"] and "Paused until" in reasons["prompt 3"]


async def test_blocked_pause_expires(make_service, backend):
    clock = FakeClock()
    service = make_service(clock=clock)
    await service.generate_images(["x [blocked]"], "x", None)
    blocked = await service.generate_images(["y"], "x", None)
    assert "Paused until" in blocked.failures[0].reason and "y" not in backend.attempts
    clock.advance(31 * 60)
    ok = await service.generate_images(["y"], "x", None)
    assert len(ok.images) == 1


async def test_edit_continues_chat_and_keeps_original(make_service, backend):
    service = make_service()
    gen = await service.generate_images(["a cat"], "a cat", None)
    src = gen.images[0]
    outcome, source = await service.edit_image(gen.session_id, src.image_id, "Change red to blue. Keep everything else exactly the same.")
    assert len(outcome.images) == 1
    new = outcome.images[0]
    assert new.image_id != src.image_id
    last = backend.sent[-1]
    assert last["metadata"] == source.metadata and last["files"] is None
    assert service.image_path(gen.session_id, src.image_id, "png").is_file()
    assert service.store.get_image(gen.session_id, new.image_id).parent_id == src.image_id


async def test_edit_attaches_image_when_ambiguous(make_service, backend):
    service = make_service()
    gen = await service.generate_images(["cats [multi3]"], "x", None)
    second = gen.images[1]
    await service.edit_image(gen.session_id, second.image_id, "Change X to Y.")
    assert backend.sent[-1]["files"] == [service.image_path(gen.session_id, second.image_id, "png")]
    # Editing an older turn of a chat that has moved on also attaches the exact picture.
    gen2 = await service.generate_images(["dog"], "x", gen.session_id)
    first = gen2.images[0]
    await service.edit_image(gen.session_id, first.image_id, "Change A to B.")
    assert backend.sent[-1]["files"] is None
    await service.edit_image(gen.session_id, first.image_id, "Change C to D.")
    assert backend.sent[-1]["files"] == [service.image_path(gen.session_id, first.image_id, "png")]


async def test_edit_rejects_unknown_ids(make_service):
    service = make_service()
    gen = await service.generate_images(["a"], "x", None)
    with pytest.raises(UserError):
        await service.edit_image(gen.session_id, "0" * 16, "Change")
    with pytest.raises(UserError):
        await service.edit_image("f" * 32, gen.images[0].image_id, "Change")


async def test_unknown_session_rejected(make_service):
    service = make_service()
    with pytest.raises(UserError):
        await service.generate_images(["a"], "x", "../../etc")


async def test_end_chat_deletes_gemini_chats_images_and_record(make_service, backend):
    service = make_service()
    gen = await service.generate_images(["a", "b [fail]"], "x", None)
    sid = gen.session_id
    cids = {c.cid for c in service.store.list_chats(sid)}
    assert len(cids) == 3  # includes both failed attempts' chats
    [report] = await service.end_sessions([sid])
    assert set(backend.deleted) == cids
    assert report.chats_deleted == 3 and report.images_deleted == 1 and report.session_deleted
    assert not (service.settings.images_dir / sid).exists()
    assert not service.store.session_exists(sid)
    [again] = await service.end_sessions([sid])
    assert not again.found


async def test_end_chat_failure_keeps_session_for_retry(make_service, backend):
    service = make_service()
    gen = await service.generate_images(["a", "b"], "x", None)
    sid = gen.session_id
    bad = service.store.list_chats(sid)[0].cid
    backend.fail_delete.add(bad)
    [report] = await service.end_sessions([sid])
    assert report.chats_deleted == 1 and len(report.chat_failures) == 1 and not report.session_deleted
    assert report.images_deleted == 2 and not (service.settings.images_dir / sid).exists()
    assert [c.cid for c in service.store.list_chats(sid)] == [bad]
    backend.fail_delete.clear()
    [retry] = await service.end_sessions([sid])
    assert retry.chats_deleted == 1 and retry.session_deleted


async def test_ttl_sweep_ends_only_idle_sessions(make_service, backend):
    clock = FakeClock()
    service = make_service(clock=clock, session_ttl_hours=24)
    old = await service.generate_images(["old"], "x", None)
    clock.advance(23 * 3600)
    fresh = await service.generate_images(["fresh"], "x", None)
    clock.advance(2 * 3600)  # old idle 25 h, fresh idle 2 h
    reports = await service.sweep()
    assert [r.session_id for r in reports] == [old.session_id]
    assert not service.store.session_exists(old.session_id)
    assert service.store.session_exists(fresh.session_id)
    assert len(backend.deleted) == 1


async def test_configured_model_is_used_even_before_first_connect(make_service, backend):
    # The default model is only known after the backend connects (warm-up may not have run yet).
    original = backend.ensure_ready

    async def connect_then_know_model():
        await original()
        backend.default_model = "gemini-pro"

    backend.ensure_ready = connect_then_know_model
    service = make_service()
    gen = await service.generate_images(["a"], "x", None)
    assert backend.sent[-1]["model"] == "gemini-pro"
    assert service.store.list_chats(gen.session_id)[0].model == "gemini-pro"
    backend.default_model = "gemini-flash"  # edits must keep the chat's own model
    await service.edit_image(gen.session_id, gen.images[0].image_id, "Change A to B.")
    assert backend.sent[-1]["model"] == "gemini-pro"
