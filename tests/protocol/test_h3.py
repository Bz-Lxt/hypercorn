from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from aioquic.h3.events import HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection

from hypercorn.asyncio.worker_context import WorkerContext
from hypercorn.config import Config
from hypercorn.protocol.events import Body
from hypercorn.protocol.h3 import H3Protocol
from hypercorn.protocol.priorities import (
    H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE,
    H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE,
)
from hypercorn.typing import ConnectionState


def encode_uint_var(value: int) -> bytes:
    if value <= 0x3F:
        return bytes([value])
    if value <= 0x3FFF:
        return (0x4000 | value).to_bytes(2, "big")
    if value <= 0x3FFFFFFF:
        return (0x80000000 | value).to_bytes(4, "big")
    return (0xC000000000000000 | value).to_bytes(8, "big")


def priority_update_frame(frame_type: int, element_id: int, value: bytes) -> bytes:
    payload = encode_uint_var(element_id) + value
    return encode_uint_var(frame_type) + encode_uint_var(len(payload)) + payload


def build_protocol() -> H3Protocol:
    # The client mode connection requires no certificate; none of these
    # tests exchange QUIC packets.
    quic = QuicConnection(configuration=QuicConfiguration(is_client=True))
    task_group = Mock()
    # ``spawn`` must not return an awaitable (the send task is never run),
    # while ``spawn_app`` is awaited while creating streams.
    task_group.spawn_app = AsyncMock()
    return H3Protocol(
        Mock(),
        Config(),
        WorkerContext(None),
        task_group,
        ConnectionState({}),
        None,
        None,
        quic,
        AsyncMock(),
    )


def headers_event(
    stream_id: int, priority: bytes | None = None
) -> HeadersReceived:
    headers = [(b":method", b"GET"), (b":path", b"/")]
    if priority is not None:
        headers.append((b"priority", priority))
    return HeadersReceived(
        stream_id=stream_id, stream_ended=True, headers=headers
    )


@pytest.mark.asyncio
async def test_priority_update_before_request_stream() -> None:
    protocol = build_protocol()

    data = encode_uint_var(0x00) + priority_update_frame(
        H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE, 0, b"u=1, i"
    )
    protocol.control_parser.feed(2, data)

    assert protocol.scheduler.pending[0] == (1, True)

    await protocol._create_stream(headers_event(0, priority=b"u=6"))

    state = protocol.scheduler._streams[0]
    assert state.urgency == 1
    assert state.incremental is True
    assert 0 not in protocol.scheduler.pending


@pytest.mark.asyncio
async def test_push_priority_update_mapped_to_push_stream() -> None:
    protocol = build_protocol()

    data = encode_uint_var(0x00) + priority_update_frame(
        H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE, 1, b"u=2"
    )
    protocol.control_parser.feed(2, data)

    # Push ID 1 is carried by the server push stream with identifier 7.
    assert protocol.scheduler.pending[7] == (2, False)

    await protocol._create_stream(headers_event(7))
    assert protocol.scheduler._streams[7].urgency == 2


@pytest.mark.asyncio
async def test_priority_update_while_sending() -> None:
    protocol = build_protocol()

    await protocol._create_stream(headers_event(0))
    await protocol._create_stream(headers_event(4, priority=b"u=3"))
    await protocol.stream_send(Body(stream_id=0, data=b"abc"))
    await protocol.stream_send(Body(stream_id=4, data=b"abc"))

    assert protocol.scheduler.next_ready() == 0

    protocol._priority_update(4, b"u=0", False)
    assert protocol.scheduler._streams[4].urgency == 0
    assert protocol.scheduler.next_ready() == 4


@pytest.mark.asyncio
async def test_malformed_priority_update_is_ignored() -> None:
    protocol = build_protocol()
    protocol._priority_update(0, b'"', False)
    assert protocol.scheduler.pending == {}


@pytest.mark.asyncio
async def test_body_is_buffered_until_scheduled() -> None:
    protocol = build_protocol()
    await protocol._create_stream(headers_event(0))

    await protocol.stream_send(Body(stream_id=0, data=b"abc"))

    state = protocol.scheduler._streams[0]
    assert state.blocked is False
    assert bytes(protocol.stream_buffers[0].buffer) == b"abc"


@pytest.mark.asyncio
async def test_send_without_flow_control_window_blocks_stream() -> None:
    protocol = build_protocol()
    await protocol._create_stream(headers_event(0))
    await protocol.stream_send(Body(stream_id=0, data=b"abc"))

    # The QUIC connection has no stream state, hence no send window: the
    # stream is marked blocked and keeps its buffered data.
    await protocol._send_data(0)

    state = protocol.scheduler._streams[0]
    assert state.blocked is True
    assert bytes(protocol.stream_buffers[0].buffer) == b"abc"


@pytest.mark.asyncio
async def test_close_stream_removes_it_from_schedule() -> None:
    protocol = build_protocol()
    await protocol._create_stream(headers_event(0))
    await protocol.stream_send(Body(stream_id=0, data=b"abc"))

    await protocol._close_stream(0)

    assert 0 not in protocol.scheduler._streams
    assert 0 not in protocol.stream_buffers
    assert protocol.scheduler.next_ready() is None


@pytest.mark.asyncio
async def test_close_stops_send_task() -> None:
    protocol = build_protocol()
    await protocol.close()
    assert protocol.closed is True
