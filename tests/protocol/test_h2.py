from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, call, Mock

import h2.settings
import pytest
from h2.connection import H2Connection
from h2.events import ConnectionTerminated, DataReceived

from hypercorn.asyncio.worker_context import EventWrapper, WorkerContext
from hypercorn.config import Config
from hypercorn.events import Closed, RawData
from hypercorn.protocol.events import Body, Response, StreamClosed
from hypercorn.protocol.h2 import BUFFER_HIGH_WATER, BufferCompleteError, H2Protocol, StreamBuffer
from hypercorn.protocol.priorities import _PriorityUpdateFrame
from hypercorn.typing import ConnectionState

REQUEST_HEADERS = [
    (":method", "GET"),
    (":path", "/"),
    (":authority", "hypercorn"),
    (":scheme", "https"),
]


def _build_protocol() -> H2Protocol:
    return H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        False,
        None,
        None,
        AsyncMock(),
    )


async def _start_exchange(
    protocol: H2Protocol, client: H2Connection
) -> list[list]:
    """Exchange the HTTP/2 connection preface.

    Returns a mutable marker used by :func:`_drain_server_data` to only
    consider newly sent data.
    """
    client.initiate_connection()
    await protocol.handle(RawData(data=client.data_to_send()))
    marker = [len(protocol.send.call_args_list)]
    for sent in protocol.send.call_args_list:
        event = sent.args[0]
        if isinstance(event, RawData):
            client.receive_data(event.data)
    return marker


def _drain_server_data(
    protocol: H2Protocol, client: H2Connection, marker: list[int]
) -> list[DataReceived]:
    calls = protocol.send.call_args_list[marker[0] :]
    marker[0] = len(protocol.send.call_args_list)
    data_events: list[DataReceived] = []
    for sent in calls:
        event = sent.args[0]
        if isinstance(event, RawData):
            for received in client.receive_data(event.data):
                if isinstance(received, DataReceived) and received.data:
                    data_events.append(received)
    return data_events


async def _open_stream(
    protocol: H2Protocol,
    client: H2Connection,
    stream_id: int,
    priority: str | None = None,
) -> None:
    headers = list(REQUEST_HEADERS)
    if priority is not None:
        headers.append(("priority", priority))
    client.send_headers(stream_id, headers, end_stream=True)
    await protocol.handle(RawData(data=client.data_to_send()))


async def _enqueue_body(protocol: H2Protocol, stream_id: int, data: bytes) -> None:
    await protocol.stream_send(Response(stream_id=stream_id, status_code=200, headers=[]))
    await protocol.stream_send(Body(stream_id=stream_id, data=data))


async def _drain_schedule(
    protocol: H2Protocol, client: H2Connection, marker: list[int], iterations: int = 12
) -> list[int]:
    """Drive the scheduler and report which streams carried data."""
    order: list[int] = []
    for _ in range(iterations):
        stream_id = protocol.scheduler.next_ready()
        if stream_id is None:
            break
        await protocol._send_data(stream_id)
        order.extend(
            event.stream_id for event in _drain_server_data(protocol, client, marker)
        )
    return order


def _priority_update_frame(
    header_stream_id: int, prioritized_stream_id: int, value: bytes
) -> bytes:
    payload = prioritized_stream_id.to_bytes(4, "big") + value
    return (
        len(payload).to_bytes(3, "big")
        + bytes([0x10, 0])
        + header_stream_id.to_bytes(4, "big")
        + payload
    )


@pytest.mark.asyncio
async def test_stream_buffer_push_and_pop() -> None:
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    stream_buffer = StreamBuffer(EventWrapper)

    async def _push_over_limit() -> bool:
        await stream_buffer.push(b"a" * (BUFFER_HIGH_WATER + 1))
        return True

    task = event_loop.create_task(_push_over_limit())
    assert not task.done()  # Blocked as over high water
    await stream_buffer.pop(BUFFER_HIGH_WATER // 4)
    assert not task.done()  # Blocked as over low water
    await stream_buffer.pop(BUFFER_HIGH_WATER // 4)
    assert (await task) is True


@pytest.mark.asyncio
async def test_stream_buffer_drain() -> None:
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    stream_buffer = StreamBuffer(EventWrapper)
    await stream_buffer.push(b"a" * 10)

    async def _drain() -> bool:
        await stream_buffer.drain()
        return True

    task = event_loop.create_task(_drain())
    assert not task.done()  # Blocked
    await stream_buffer.pop(20)
    assert (await task) is True


@pytest.mark.asyncio
async def test_stream_buffer_closed() -> None:
    stream_buffer = StreamBuffer(EventWrapper)
    await stream_buffer.close()
    await stream_buffer._is_empty.wait()
    await stream_buffer._paused.wait()
    assert True
    with pytest.raises(BufferCompleteError):
        await stream_buffer.push(b"a")


@pytest.mark.asyncio
async def test_stream_buffer_complete() -> None:
    stream_buffer = StreamBuffer(EventWrapper)
    await stream_buffer.push(b"a" * 10)
    assert not stream_buffer.complete
    stream_buffer.set_complete()
    assert not stream_buffer.complete
    await stream_buffer.pop(20)
    assert stream_buffer.complete


@pytest.mark.asyncio
async def test_protocol_handle_protocol_error() -> None:
    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        False,
        None,
        None,
        AsyncMock(),
    )
    await protocol.handle(RawData(data=b"broken nonsense\r\n\r\n"))
    protocol.send.assert_awaited()  # type: ignore
    assert protocol.send.call_args_list == [call(Closed())]  # type: ignore


@pytest.mark.asyncio
async def test_protocol_keep_alive_max_requests() -> None:
    protocol = H2Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        AsyncMock(),
        ConnectionState({}),
        False,
        None,
        None,
        AsyncMock(),
    )
    protocol.config.keep_alive_max_requests = 0
    client = H2Connection()
    client.initiate_connection()
    headers = [
        (":method", "GET"),
        (":path", "/reqinfo"),
        (":authority", "hypercorn"),
        (":scheme", "https"),
    ]
    client.send_headers(1, headers, end_stream=True)
    await protocol.handle(RawData(data=client.data_to_send()))
    protocol.send.assert_awaited()  # type: ignore
    events = client.receive_data(protocol.send.call_args_list[1].args[0].data)  # type: ignore
    assert isinstance(events[-1], ConnectionTerminated)


# ---------------------------------------------------------------------------
# RFC 9218 priority scheduling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_header_field_applied() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    await _start_exchange(protocol, client)

    await _open_stream(protocol, client, 1, priority="u=1, i")
    await _open_stream(protocol, client, 3)
    await _open_stream(protocol, client, 5, priority='"')

    assert protocol.scheduler._streams[1].urgency == 1
    assert protocol.scheduler._streams[1].incremental is True
    assert protocol.scheduler._streams[3].urgency == 3
    assert protocol.scheduler._streams[3].incremental is False
    # An unparseable Priority header field is ignored in favour of defaults.
    assert protocol.scheduler._streams[5].urgency == 3
    assert protocol.scheduler._streams[5].incremental is False


@pytest.mark.asyncio
async def test_scheduling_urgency_order() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    await _open_stream(protocol, client, 1, priority="u=5")
    await _open_stream(protocol, client, 3, priority="u=1")
    await _open_stream(protocol, client, 5, priority="u=3")
    for stream_id in (1, 3, 5):
        await _enqueue_body(protocol, stream_id, b"x" * 100)
    _drain_server_data(protocol, client, marker)  # response headers

    assert await _drain_schedule(protocol, client, marker) == [3, 5, 1]


@pytest.mark.asyncio
async def test_scheduling_non_incremental_request_order() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    for stream_id in (1, 3, 5):
        await _open_stream(protocol, client, stream_id)
    for stream_id in (1, 3, 5):
        await _enqueue_body(protocol, stream_id, b"x" * 100)
    _drain_server_data(protocol, client, marker)  # response headers

    assert await _drain_schedule(protocol, client, marker) == [1, 3, 5]


@pytest.mark.asyncio
async def test_scheduling_incremental_round_robin() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    await _open_stream(protocol, client, 1, priority="i")
    await _open_stream(protocol, client, 3, priority="i")
    for stream_id in (1, 3):
        await _enqueue_body(protocol, stream_id, b"x" * 20000)
    _drain_server_data(protocol, client, marker)  # response headers

    assert await _drain_schedule(protocol, client, marker, iterations=8) == [
        1,
        3,
        1,
        3,
    ]


@pytest.mark.asyncio
async def test_flow_control_blocked_stream_is_skipped() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    # Advertise a zero stream-level flow-control window before the streams
    # are opened, so every response starts flow-control blocked.
    client.update_settings({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 0})
    await protocol.handle(RawData(data=client.data_to_send()))
    _drain_server_data(protocol, client, marker)

    await _open_stream(protocol, client, 1)
    await _open_stream(protocol, client, 3)
    for stream_id in (1, 3):
        await _enqueue_body(protocol, stream_id, b"x" * 100)
    _drain_server_data(protocol, client, marker)  # response headers

    assert await _drain_schedule(protocol, client, marker) == []

    # Only stream 3 gets window back; stream 1 must not stop it sending.
    client.increment_flow_control_window(stream_id=3, increment=65535)
    await protocol.handle(RawData(data=client.data_to_send()))
    assert await _drain_schedule(protocol, client, marker) == [3]
    assert protocol.scheduler._streams[1].blocked is True
    assert protocol.scheduler._streams[3].blocked is True


@pytest.mark.asyncio
async def test_priority_update_before_stream_open() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    await _start_exchange(protocol, client)

    for value in (b"u=1, i", b"u=0"):
        frame = _PriorityUpdateFrame(
            stream_id=0, prioritized_stream_id=5, priority_field_value=value
        ).serialize()
        await protocol.handle(RawData(data=frame))

    # Only the most recent update for a not-yet-open stream is retained.
    assert protocol.scheduler.pending[5] == (0, False)

    # A Priority header on the eventual request is overridden by the update.
    await _open_stream(protocol, client, 5, priority="u=6")
    assert protocol.scheduler._streams[5].urgency == 0
    assert protocol.scheduler._streams[5].incremental is False
    assert 5 not in protocol.scheduler.pending


@pytest.mark.asyncio
async def test_priority_update_while_sending() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    await _open_stream(protocol, client, 1, priority="u=3")
    await _open_stream(protocol, client, 3, priority="u=5")
    for stream_id in (1, 3):
        await _enqueue_body(protocol, stream_id, b"x" * 100)
    _drain_server_data(protocol, client, marker)  # response headers

    assert protocol.scheduler.next_ready() == 1

    frame = _PriorityUpdateFrame(
        stream_id=0, prioritized_stream_id=3, priority_field_value=b"u=0"
    ).serialize()
    await protocol.handle(RawData(data=frame))

    assert protocol.scheduler._streams[3].urgency == 0
    assert protocol.scheduler.next_ready() == 3


@pytest.mark.asyncio
async def test_priority_update_malformed_field_ignored() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    await _start_exchange(protocol, client)

    await _open_stream(protocol, client, 1)
    frame = _PriorityUpdateFrame(
        stream_id=0, prioritized_stream_id=1, priority_field_value=b'"'
    ).serialize()
    await protocol.handle(RawData(data=frame))

    assert protocol.scheduler._streams[1].urgency == 3
    assert not any(
        isinstance(sent.args[0], Closed) for sent in protocol.send.call_args_list
    )


@pytest.mark.parametrize(
    "header_stream_id, prioritized_stream_id",
    [(1, 1), (0, 0)],
)
@pytest.mark.asyncio
async def test_priority_update_invalid_frame_is_connection_error(
    header_stream_id: int, prioritized_stream_id: int
) -> None:
    protocol = _build_protocol()
    client = H2Connection()
    await _start_exchange(protocol, client)

    await protocol.handle(
        RawData(
            data=_priority_update_frame(
                header_stream_id, prioritized_stream_id, b"u=1"
            )
        )
    )

    assert any(
        isinstance(sent.args[0], Closed) for sent in protocol.send.call_args_list
    )


@pytest.mark.asyncio
async def test_reset_stream_removed_from_schedule() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    await _open_stream(protocol, client, 1)
    await _open_stream(protocol, client, 3)
    client.reset_stream(1)
    await protocol.handle(RawData(data=client.data_to_send()))

    assert 1 not in protocol.scheduler._streams
    assert 1 not in protocol.stream_buffers

    await _enqueue_body(protocol, 3, b"x" * 100)
    _drain_server_data(protocol, client, marker)
    assert await _drain_schedule(protocol, client, marker) == [3]


@pytest.mark.asyncio
async def test_closed_stream_removed_from_schedule() -> None:
    protocol = _build_protocol()
    client = H2Connection()
    marker = await _start_exchange(protocol, client)

    for stream_id in (1, 3):
        await _open_stream(protocol, client, stream_id)
    await protocol.stream_send(StreamClosed(stream_id=1))

    assert 1 not in protocol.scheduler._streams
    assert 1 not in protocol.stream_buffers

    await _enqueue_body(protocol, 3, b"x" * 100)
    _drain_server_data(protocol, client, marker)
    assert await _drain_schedule(protocol, client, marker) == [3]
