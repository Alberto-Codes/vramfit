from __future__ import annotations

import gc
import sys
import warnings

import pytest

from vramfit.adapters.outbound.run_log_jsonl import (
    RUNLOG_VERSION,
    JsonlRunLogFile,
    RunLogError,
    read_run_log,
)
from vramfit.domain.errors import VramfitError

pytestmark = pytest.mark.unit


@pytest.fixture
def digit_limit():
    """Pin the interpreter's integer digit limit, then restore it.

    `PYTHONINTMAXSTRDIGITS=0` disables the limit, and a disabled limit
    would let the refusal tests parse their own fixture.
    """
    previous = sys.get_int_max_str_digits()
    limit = sys.int_info.str_digits_check_threshold
    sys.set_int_max_str_digits(limit)
    yield limit
    sys.set_int_max_str_digits(previous)


def test_every_line_carries_the_envelope_and_timestamp(tmp_path) -> None:
    path = tmp_path / "scan.runlog.jsonl"

    JsonlRunLogFile(path).emit("scan_started", {"model": "m"})

    (event,) = read_run_log(path)
    assert event["event"] == "scan_started"
    assert event["vramfit_runlog"] == RUNLOG_VERSION
    assert event["model"] == "m"
    assert event["ts"].endswith("Z") or "+" in event["ts"]


def test_events_append_across_adapter_instances(tmp_path) -> None:
    path = tmp_path / "scan.runlog.jsonl"

    JsonlRunLogFile(path).emit("scan_started", {})
    JsonlRunLogFile(path).emit("scan_finished", {})

    assert [e["event"] for e in read_run_log(path)] == [
        "scan_started",
        "scan_finished",
    ]


def test_emit_into_a_missing_directory_raises_os_error(tmp_path) -> None:
    with pytest.raises(OSError):
        JsonlRunLogFile(tmp_path / "nope" / "x.jsonl").emit("scan_started", {})


def test_events_reach_disk_before_the_handle_closes(tmp_path) -> None:
    # The black-box guarantee: a crash must not lose emitted events.
    # structlog's PrintLogger flushes per line — pin it.
    path = tmp_path / "scan.runlog.jsonl"
    sink = JsonlRunLogFile(path)

    sink.emit("scan_started", {})

    assert path.stat().st_size > 0
    assert read_run_log(path)[0]["event"] == "scan_started"


def test_non_serializable_field_raises_instead_of_degrading(tmp_path) -> None:
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")

    with pytest.raises(RunLogError, match="not JSON serializable"):
        sink.emit("scan_started", {"bad": object()})


def test_non_finite_field_raises_instead_of_corrupting_json(tmp_path) -> None:
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")

    with pytest.raises(RunLogError, match=r"[Nn]a[Nn]|allow_nan"):
        sink.emit("cell_measured", {"damage": float("nan")})


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"bad": object()}, "not JSON serializable"),
        ({"damage": float("nan")}, "nan"),
        ({"damage": float("inf")}, "inf"),
    ],
    ids=["object", "nan", "inf"],
)
def test_emit_refuses_each_bad_field_under_the_error_root(
    tmp_path, fields, match: str
) -> None:
    # #475: the renderer's `TypeError` and `ValueError` are foreign to
    # the port. Each becomes `RunLogError`, and the message names the
    # file and the event. The file stays empty.
    path = tmp_path / "x.jsonl"

    with pytest.raises(VramfitError, match=match) as info:
        JsonlRunLogFile(path).emit("cell_measured", fields)

    assert str(path) in str(info.value)
    assert "'cell_measured'" in str(info.value)
    assert path.read_text(encoding="utf-8") == ""


def test_reader_drops_a_torn_final_line(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    JsonlRunLogFile(path).emit("scan_started", {})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "cell_measu')  # crash mid-write

    events = read_run_log(path)

    assert [e["event"] for e in events] == ["scan_started"]


def test_reader_raises_on_a_torn_middle_line(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"broken\n{"event": "scan_finished", "vramfit_runlog": 2}\n')

    with pytest.raises(ValueError):
        read_run_log(path)


def test_emit_cannot_write_a_duplicate_envelope_key(tmp_path) -> None:
    # The premise behind the refusals below: vramfit never produces the
    # defect, because a colliding field is a duplicate keyword argument
    # at the `msg` call. So a duplicate key means a hand edit or a
    # foreign writer (#283).
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")

    with pytest.raises(RunLogError, match="vramfit_runlog"):
        sink.emit("scan_started", {"vramfit_runlog": 99})
    with pytest.raises(RunLogError, match="event"):
        sink.emit("scan_started", {"event": "other"})


def test_reader_raises_on_a_duplicate_key_in_a_middle_line(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"event": "cell_measured", "damage": 1.0, "damage": 2.0}\n'
        '{"event": "scan_finished", "vramfit_runlog": 2}\n'
    )

    with pytest.raises(ValueError, match='line 1: duplicate key "damage"'):
        read_run_log(path)


def test_reader_names_the_file_line_past_a_blank_line(tmp_path) -> None:
    # The locator must survive the blank-line filter. A hand edit writes
    # the duplicate key, and a hand edit leaves the blank line.
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"event": "scan_started"}\n'
        "\n"
        "\n"
        '{"event": "cell_measured", "damage": 1.0, "damage": 2.0}\n'
    )

    with pytest.raises(ValueError, match='line 4: duplicate key "damage"'):
        read_run_log(path)


def test_reader_raises_on_a_duplicate_key_in_the_final_line(tmp_path) -> None:
    # ADR-0011 decision 2 drops a torn final line, which is the crash
    # signature. A duplicate key parses and no crash writes one, so the
    # drop rule does not reach it (#283).
    path = tmp_path / "x.jsonl"
    JsonlRunLogFile(path).emit("scan_started", {})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "cell_measured", "damage": 1.0, "damage": 2.0}\n')

    with pytest.raises(ValueError, match='line 2: duplicate key "damage"'):
        read_run_log(path)


def test_reader_refuses_an_unparsable_number_in_the_final_line(
    tmp_path, digit_limit
) -> None:
    # #315: the drop rule caught every `ValueError`, scoped by position
    # and never by kind. An integer literal past
    # `sys.get_int_max_str_digits` is complete, well-formed JSON that
    # the parser still refuses. No crash writes one, so it raises.
    huge = "9" * (digit_limit + 1)
    path = tmp_path / "x.jsonl"
    JsonlRunLogFile(path).emit("scan_started", {})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f'{{"event": "cell_measured", "n": {huge}}}\n')

    with pytest.raises(
        ValueError,
        match=r"line 2: cannot parse JSON: Exceeds the limit .*digits\)"
        r" for integer string conversion: value has \d+ digits$",
    ):
        read_run_log(path)


def test_reader_refuses_an_unparsable_number_past_a_blank_line(
    tmp_path, digit_limit
) -> None:
    # The locator must survive the blank-line filter on this path too.
    huge = "9" * (digit_limit + 1)
    path = tmp_path / "x.jsonl"
    path.write_text(
        f'{{"event": "scan_started"}}\n\n{{"event": "cell_measured", "n": {huge}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 3: cannot parse JSON"):
        read_run_log(path)


def test_reader_drops_a_final_line_torn_mid_float(tmp_path) -> None:
    # ADR-0011 decision 2. Narrowing the catch by kind must not reach
    # the crash signature the rule protects.
    path = tmp_path / "x.jsonl"
    JsonlRunLogFile(path).emit("scan_started", {})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "cell_measured", "ppl": 8.')  # crash mid-float

    events = read_run_log(path)

    assert [e["event"] for e in events] == ["scan_started"]


def test_reader_names_the_file_line_for_a_torn_middle_line(tmp_path) -> None:
    # The raw `JSONDecodeError` reports the line inside the string it
    # parsed, which is always 1. The refusal names the file line.
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"event": "scan_started"}\n'
        '{"broken\n'
        '{"event": "scan_finished", "vramfit_runlog": 2}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"line 2: invalid JSON: Unterminated string starting at: column 2$",
    ):
        read_run_log(path)


# Matches the `digit_limit` fixture's limit. Parametrize needs the
# literal at collection time, before any fixture runs.
HUGE = "9" * (sys.int_info.str_digits_check_threshold + 1)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ('{"event": "a", "damage": 1.0, "damage": 2.0}', "line 1: duplicate key"),
        ('{"broken', "line 1: invalid JSON"),
        (f'{{"event": "a", "n": {HUGE}}}', "line 1: cannot parse JSON"),
    ],
    ids=["duplicate_key", "torn_middle_line", "unparsable_number"],
)
def test_reader_refuses_each_defect_under_the_error_root(
    tmp_path, digit_limit, line: str, match: str
) -> None:
    # ADR-0011 decision 5 and its 2026-08-16 amendment: no
    # representability failure escapes the root. Before #346 each of
    # the three refusals raised a plain `ValueError`.
    path = tmp_path / "x.jsonl"
    path.write_text(f'{line}\n{{"event": "scan_finished"}}\n', encoding="utf-8")

    with pytest.raises(VramfitError, match=match):
        read_run_log(path)


def test_run_log_error_keeps_the_value_error_base() -> None:
    # The historical type stays catchable, so no caller's catch
    # changes (ADR-0011 decision 5).
    assert issubclass(RunLogError, ValueError)
    assert issubclass(RunLogError, VramfitError)


def test_emit_leaves_no_open_handle_behind(tmp_path) -> None:
    # #468: the sink used to cache an append handle and never close it.
    # The interpreter then closed it at collection and raised a
    # `ResourceWarning` inside whichever test was running.
    sink = JsonlRunLogFile(tmp_path / "x.jsonl")
    sink.emit("scan_started", {})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del sink
        gc.collect()

    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []
