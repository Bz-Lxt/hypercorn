from __future__ import annotations

import pytest

from hypercorn.protocol.priorities import (
    H3ControlStreamParser,
    H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE,
    H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE,
    parse_priority_field,
    PriorityScheduler,
    pull_uint_var,
)


# ---------------------------------------------------------------------------
# Priority field parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (b"", (3, False)),
        (b"u=0", (0, False)),
        (b"u=7", (7, False)),
        (b"u=3", (3, False)),
        (b"i", (3, True)),
        (b"i=?1", (3, True)),
        (b"i=?0", (3, False)),
        (b"u=5, i", (5, True)),
        (b'i, u=2', (2, True)),
        # Out of range urgency is ignored, default applies.
        (b"u=9", (3, False)),
        (b"u=-1", (3, False)),
        # Unexpected value types are ignored.
        (b"u=x", (3, False)),
        (b'u="1"', (3, False)),
        (b"i=2", (3, False)),
        # Unknown parameters are ignored.
        (b'unknown, u=1', (1, False)),
        (b'foo="a,b", u=2', (2, False)),
        # Structured Fields parameters are tolerated.
        (b"u=1;foo=bar, i=?0;x=y", (1, False)),
        # The first occurrence of a repeated key is used.
        (b"u=1, u=2", (1, False)),
        # Whitespace is tolerated.
        (b" u=4 ,\ti=?1", (4, True)),
        (b"u=4,i", (4, True)),
    ],
)
def test_parse_priority_field(value: bytes, expected: tuple[int, bool]) -> None:
    assert parse_priority_field(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        b"u=1, ",  # empty dictionary member
        b"\xff",  # non-ascii
        b'"unterminated, u=1',
    ],
)
def test_parse_priority_field_invalid(value: bytes) -> None:
    assert parse_priority_field(value) is None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _pop_order(scheduler: PriorityScheduler, count: int) -> list[int]:
    order = []
    for _ in range(count):
        stream_id = scheduler.next_ready()
        order.append(stream_id)
    return order


def test_scheduler_empty() -> None:
    scheduler = PriorityScheduler()
    assert scheduler.next_ready() is None


def test_scheduler_streams_start_blocked() -> None:
    scheduler = PriorityScheduler()
    scheduler.add(1)
    assert scheduler.next_ready() is None
    scheduler.unblock(1)
    assert scheduler.next_ready() == 1


def test_scheduler_urgency_order() -> None:
    scheduler = PriorityScheduler()
    scheduler.add(1, urgency=3)
    scheduler.add(3, urgency=0)
    scheduler.add(5, urgency=7)
    scheduler.add(7, urgency=1)
    for stream_id in (1, 3, 5, 7):
        scheduler.unblock(stream_id)

    order = []
    while True:
        stream_id = scheduler.next_ready()
        if stream_id is None:
            break
        order.append(stream_id)
        scheduler.remove(stream_id)
    assert order == [3, 7, 1, 5]


def test_scheduler_non_incremental_request_order() -> None:
    scheduler = PriorityScheduler()
    for stream_id in (1, 3, 5):
        scheduler.add(stream_id)
        scheduler.unblock(stream_id)

    # The earliest request is served exclusively, in order.
    assert _pop_order(scheduler, 3) == [1, 1, 1]
    scheduler.remove(1)
    assert _pop_order(scheduler, 2) == [3, 3]
    scheduler.remove(3)
    assert scheduler.next_ready() == 5


def test_scheduler_non_incremental_skips_blocked() -> None:
    scheduler = PriorityScheduler()
    for stream_id in (1, 3, 5):
        scheduler.add(stream_id)
        scheduler.unblock(stream_id)

    # Stream 1 is flow-control blocked; the others must not be held up.
    scheduler.block(1)
    assert scheduler.next_ready() == 3

    # When stream 1 recovers it is served first again (request order).
    scheduler.unblock(1)
    assert scheduler.next_ready() == 1


def test_scheduler_incremental_round_robin() -> None:
    scheduler = PriorityScheduler()
    for stream_id in (1, 3, 5):
        scheduler.add(stream_id, incremental=True)
        scheduler.unblock(stream_id)

    assert _pop_order(scheduler, 6) == [1, 3, 5, 1, 3, 5]


def test_scheduler_incremental_skips_blocked() -> None:
    scheduler = PriorityScheduler()
    for stream_id in (1, 3, 5):
        scheduler.add(stream_id, incremental=True)
        scheduler.unblock(stream_id)

    assert scheduler.next_ready() == 1
    scheduler.block(1)
    assert _pop_order(scheduler, 4) == [3, 5, 3, 5]
    scheduler.unblock(1)
    assert scheduler.next_ready() == 1


def test_scheduler_mixed_non_incremental_first() -> None:
    scheduler = PriorityScheduler()
    scheduler.add(1)  # non-incremental
    scheduler.add(3, incremental=True)
    scheduler.add(5, incremental=True)
    for stream_id in (1, 3, 5):
        scheduler.unblock(stream_id)

    # The non-incremental response is served one at a time.
    assert _pop_order(scheduler, 2) == [1, 1]
    # Only once it stalls do the incremental responses progress.
    scheduler.block(1)
    assert _pop_order(scheduler, 4) == [3, 5, 3, 5]


def test_scheduler_closed_stream_skipped() -> None:
    scheduler = PriorityScheduler()
    for stream_id in (1, 3, 5):
        scheduler.add(stream_id)
        scheduler.unblock(stream_id)

    scheduler.remove(1)
    assert scheduler.next_ready() == 3
    scheduler.remove(3)
    assert scheduler.next_ready() == 5
    scheduler.remove(5)
    assert scheduler.next_ready() is None


def test_scheduler_reprioritize_live() -> None:
    scheduler = PriorityScheduler()
    scheduler.add(1, urgency=3)
    scheduler.add(3, urgency=3)
    scheduler.unblock(1)
    scheduler.unblock(3)
    assert scheduler.next_ready() == 1

    # A continuous update during sending takes immediate effect.
    scheduler.set_priority(1, 0, False)
    scheduler.set_priority(3, 7, False)
    assert scheduler.next_ready() == 1
    scheduler.remove(1)
    assert scheduler.next_ready() == 3
    scheduler.remove(3)

    scheduler.add(5, urgency=3, incremental=False)
    scheduler.unblock(5)
    scheduler.set_priority(5, 3, True)
    assert scheduler.next_ready() == 5


def test_scheduler_pending_update_before_open() -> None:
    scheduler = PriorityScheduler()
    # PRIORITY_UPDATE arrives before the HEADERS frame opens the stream.
    scheduler.set_priority(7, 0, True)
    assert 7 in scheduler.pending

    # The header signals defaults, but the buffered update overrides it.
    scheduler.add(7)
    assert 7 not in scheduler.pending
    state = scheduler._streams[7]
    assert state.urgency == 0
    assert state.incremental is True


def test_scheduler_pending_update_replaced() -> None:
    scheduler = PriorityScheduler()
    scheduler.set_priority(9, 0, False)
    scheduler.set_priority(9, 5, True)  # only the latest is retained
    scheduler.add(9, urgency=3, incremental=False)
    state = scheduler._streams[9]
    assert state.urgency == 5
    assert state.incremental is True


def test_scheduler_pending_limit() -> None:
    scheduler = PriorityScheduler(pending_limit=2)
    scheduler.set_priority(1, 0, False)
    scheduler.set_priority(3, 0, False)
    scheduler.set_priority(5, 0, False)
    assert 1 not in scheduler.pending
    assert 3 in scheduler.pending
    assert 5 in scheduler.pending


# ---------------------------------------------------------------------------
# QUIC variable-length integers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoded, value",
    [
        (b"\x00", 0),
        (b"\x01", 1),
        (b"\x3f", 63),
        (b"\x40\x40", 64),
        (b"\x7f\xff", 16383),
        (b"\x80\x00\x40\x00", 16384),
        (b"\xc0\x00\x00\x00\x00\x00\x00\x00", 0),
    ],
)
def test_pull_uint_var(encoded: bytes, value: int) -> None:
    assert pull_uint_var(encoded, 0) == (value, len(encoded))


def test_pull_uint_var_incomplete() -> None:
    assert pull_uint_var(b"\xc0", 0) is None
    assert pull_uint_var(b"\x00", 1) is None


# ---------------------------------------------------------------------------
# HTTP/3 control stream scanning
# ---------------------------------------------------------------------------


def make_collector() -> tuple[list[tuple[int, bytes, bool]], object]:
    updates: list[tuple[int, bytes, bool]] = []

    def collect(stream_id: int, field_value: bytes, is_push: bool) -> None:
        updates.append((stream_id, field_value, is_push))

    return updates, collect


def encode_uint_var(value: int) -> bytes:
    if value <= 63:
        return bytes([value])
    elif value <= 16383:
        return bytes([0x40 | (value >> 8), value & 0xFF])
    elif value <= 1073741823:
        return bytes(
            [0x80 | (value >> 24), (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
        )
    raise AssertionError


def h3_frame(frame_type: int, payload: bytes) -> bytes:
    return encode_uint_var(frame_type) + encode_uint_var(len(payload)) + payload


def test_h3_control_stream_request_update() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    payload = encode_uint_var(0) + b"u=2, i"
    data = b"\x00" + h3_frame(H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, payload)
    parser.feed(2, data)

    assert updates == [(0, b"u=2, i", False)]


def test_h3_control_stream_push_update() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    payload = encode_uint_var(1) + b"u=0"
    data = b"\x00" + h3_frame(H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE, payload)
    parser.feed(2, data)

    # Push ID 1 maps to the server-initiated unidirectional stream 7.
    assert updates == [(7, b"u=0", True)]


def test_h3_control_stream_fragmented() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    payload = encode_uint_var(12) + b"u=7"
    data = b"\x00" + h3_frame(H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, payload)
    for byte in data:
        parser.feed(2, bytes([byte]))
    assert updates == [(12, b"u=7", False)]


def test_h3_control_stream_unknown_frames_ignored() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    data = b"\x00"
    # SETTINGS frame and an unknown frame precede the priority update.
    data += h3_frame(0x04, encode_uint_var(0x01) + encode_uint_var(1))
    data += h3_frame(0x21, b"grease payload")
    data += h3_frame(
        H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, encode_uint_var(4) + b"u=0"
    )
    parser.feed(2, data)

    assert updates == [(4, b"u=0", False)]


def test_h3_control_stream_multiple_updates_latest_wins() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    data = b"\x00"
    data += h3_frame(
        H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, encode_uint_var(0) + b"u=7"
    )
    data += h3_frame(
        H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, encode_uint_var(0) + b"u=0"
    )
    parser.feed(2, data)

    assert updates == [(0, b"u=7", False), (0, b"u=0", False)]


def test_h3_non_control_stream_ignored() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    frame = h3_frame(
        H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, encode_uint_var(0) + b"u=0"
    )
    # A request stream (client initiated bidirectional).
    parser.feed(0, b"\x00" + frame)
    # A server initiated unidirectional stream.
    parser.feed(3, b"\x00" + frame)
    # A non-control (e.g. QPACK encoder) client unidirectional stream.
    parser.feed(2, b"\x02" + frame)

    assert updates == []


def test_h3_control_stream_truncated_update() -> None:
    updates, collect = make_collector()
    parser = H3ControlStreamParser(collect)

    data = b"\x00"
    # The element identifier is split across two deliveries and cannot be
    # parsed until the rest arrives.
    payload = encode_uint_var(1073741823) + b"u=1"
    frame = h3_frame(H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, payload)
    parser.feed(2, b"\x00" + frame[:5])
    assert updates == []
    parser.feed(2, frame[5:])
    assert updates == [(1073741823, b"u=1", False)]
