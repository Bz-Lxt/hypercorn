from __future__ import annotations

from collections.abc import Awaitable, Callable

from aioquic.h3.connection import H3Connection
from aioquic.h3.events import DataReceived, HeadersReceived
from aioquic.h3.exceptions import NoAvailablePushIDError
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import (
    QuicEvent,
    StreamDataReceived,
    StreamReset,
    StopSendingReceived,
)

from .events import (
    Body,
    Data,
    EndBody,
    EndData,
    Event as StreamEvent,
    InformationalResponse,
    Request,
    Response,
    StreamClosed,
    Trailers,
)
from .http_stream import HTTPStream
from .priorities import (
    BufferCompleteError,
    H3ControlStreamParser,
    parse_priority_field,
    PriorityScheduler,
    StreamBuffer,
)
from .ws_stream import WSStream
from ..config import Config
from ..typing import AppWrapper, ConnectionState, TaskGroup, WorkerContext
from ..utils import filter_pseudo_headers

# Per-iteration limit on the data handed to aioquic for a single stream,
# matching the default HTTP/2 maximum frame size and keeping scheduling
# decisions meaningful despite QUIC's unbounded stream frames.
H3_CHUNK_SIZE = 2**14
# An HTTP/3 DATA frame prepends a one-byte type and a (two-byte, at this
# payload size) length to the application data.
H3_DATA_FRAME_OVERHEAD = 3
H3_MAX_DATA_PAYLOAD = H3_CHUNK_SIZE - H3_DATA_FRAME_OVERHEAD
# The terminating empty DATA frame uses a one-byte type and one-byte length.
H3_FIN_FRAME_SIZE = 2


class H3Protocol:
    def __init__(
        self,
        app: AppWrapper,
        config: Config,
        context: WorkerContext,
        task_group: TaskGroup,
        state: ConnectionState,
        client: tuple[str, int] | None,
        server: tuple[str, int] | None,
        quic: QuicConnection,
        send: Callable[[], Awaitable[None]],
    ) -> None:
        self.app = app
        self.client = client
        self.config = config
        self.context = context
        self.connection = H3Connection(quic)
        self.send = send
        self.server = server
        self.streams: dict[int, HTTPStream | WSStream] = {}
        self.task_group = task_group
        self.state = state

        self.closed = False
        # The below are used by the sending task, which chooses when (and
        # for which stream) response data is passed to aioquic.
        self.has_data = self.context.event_class()
        self.scheduler = PriorityScheduler()
        self.stream_buffers: dict[int, StreamBuffer] = {}
        self.control_parser = H3ControlStreamParser(self._priority_update)
        self.task_group.spawn(self.send_task)

    async def handle(self, quic_event: QuicEvent) -> None:
        # aioquic discards frame types it does not know about on the
        # control stream, so scan the raw data for PRIORITY_UPDATE frames
        # first (RFC 9218 section 7.2).
        if isinstance(quic_event, StreamDataReceived):
            self.control_parser.feed(quic_event.stream_id, quic_event.data)

        for event in self.connection.handle_event(quic_event):
            if isinstance(event, HeadersReceived):
                if not self.context.terminated.is_set():
                    await self._create_stream(event)
                    if event.stream_ended:
                        await self.streams[event.stream_id].handle(
                            EndBody(stream_id=event.stream_id)
                        )
            elif isinstance(event, DataReceived):
                await self.streams[event.stream_id].handle(
                    Body(stream_id=event.stream_id, data=event.data)
                )
                if event.stream_ended:
                    await self.streams[event.stream_id].handle(EndBody(stream_id=event.stream_id))

        if isinstance(quic_event, (StreamReset, StopSendingReceived)):
            # The peer no longer wants the response; drop it without
            # letting it interfere with the other streams.
            await self._close_stream(quic_event.stream_id)

        # Incoming QUIC packets may carry MAX_DATA / MAX_STREAM_DATA
        # frames that reopen flow-control room for blocked streams.
        await self.has_data.set()

    def _priority_update(self, stream_id: int, field_value: bytes, is_push: bool) -> None:
        priority = parse_priority_field(field_value)
        if priority is None:
            return
        urgency, incremental = priority
        # *stream_id* is the QUIC stream identifier (push identifiers are
        # mapped by the control stream parser). Updates of streams that
        # have not opened yet are remembered until their HEADERS arrive.
        self.scheduler.set_priority(stream_id, urgency, incremental)

    async def send_task(self) -> None:
        # Separate task that decides in what order response data is sent.
        while not self.closed:
            stream_id = self.scheduler.next_ready()
            if stream_id is None:
                await self.has_data.wait()
                await self.has_data.clear()
            else:
                await self._send_data(stream_id)

    async def _send_data(self, stream_id: int) -> None:
        try:
            quic = self.connection._quic
            stream = quic._streams.get(stream_id)
            if stream is None:
                self.scheduler.block(stream_id)
                return
            # Respect both the stream and connection flow-control windows.
            # A blocked stream is skipped in favour of any other stream.
            stream_window = stream.max_stream_data_remote - stream.sender._buffer_stop
            connection_window = quic._remote_max_data - quic._remote_max_data_used
            available = min(stream_window, connection_window)
            buffer = self.stream_buffers[stream_id]
            if buffer.complete:
                # The final empty HTTP/3 DATA frame still carries two bytes
                # of framing; wait for the window rather than lose the FIN.
                if available < H3_FIN_FRAME_SIZE:
                    self.scheduler.block(stream_id)
                    return
                self.connection.send_data(stream_id, b"", True)
                await self.send()
                del self.stream_buffers[stream_id]
                self.scheduler.remove(stream_id)
                return
            # Each HTTP/3 DATA frame adds type/length framing on top of the
            # application data, which is subject to QUIC flow control too.
            chunk_size = min(available - H3_DATA_FRAME_OVERHEAD, H3_MAX_DATA_PAYLOAD)
            if chunk_size <= 0:
                self.scheduler.block(stream_id)
                return
            data = await buffer.pop(chunk_size)
            if data:
                self.connection.send_data(stream_id, data, False)
                await self.send()
            else:
                self.scheduler.block(stream_id)
        except (KeyError, ValueError):
            # The stream has been reset or finished whilst waiting to send.
            if stream_id in self.stream_buffers:
                await self.stream_buffers[stream_id].close()
                del self.stream_buffers[stream_id]
            self.scheduler.remove(stream_id)

    async def stream_send(self, event: StreamEvent) -> None:
        try:
            if isinstance(event, (InformationalResponse, Response)):
                self.connection.send_headers(
                    event.stream_id,
                    [(b":status", b"%d" % event.status_code)]
                    + event.headers
                    + self.config.response_headers("h3"),
                )
                await self.send()
            elif isinstance(event, (Body, Data)):
                self.scheduler.unblock(event.stream_id)
                await self.has_data.set()
                await self.stream_buffers[event.stream_id].push(event.data)
            elif isinstance(event, (EndBody, EndData)):
                self.stream_buffers[event.stream_id].set_complete()
                self.scheduler.unblock(event.stream_id)
                await self.has_data.set()
                await self.stream_buffers[event.stream_id].drain()
            elif isinstance(event, Trailers):
                await self.stream_buffers[event.stream_id].drain()
                self.connection.send_headers(event.stream_id, event.headers)
                await self.send()
            elif isinstance(event, StreamClosed):
                self.streams.pop(event.stream_id, None)
                buffer = self.stream_buffers.get(event.stream_id)
                if buffer is None or not buffer.complete:
                    # Abnormal termination: nothing more will be sent, drop
                    # the stream immediately. A normal end (the complete
                    # flag set by EndBody) is left for the send task so it
                    # can emit the terminating FIN without racing us.
                    if buffer is not None:
                        await buffer.close()
                        del self.stream_buffers[event.stream_id]
                    self.scheduler.remove(event.stream_id)
            elif isinstance(event, Request):
                await self._create_server_push(event.stream_id, event.raw_path, event.headers)
        except (BufferCompleteError, KeyError):
            # Connection has closed whilst blocked on flow control or
            # connection has advanced ahead of the last emitted event.
            return

    async def _create_stream(self, request: HeadersReceived) -> None:
        priority_values = []
        for name, value in request.headers:
            if name == b":method":
                method = value.decode("ascii").upper()
            elif name == b":path":
                raw_path = value
            elif name == b"priority":
                priority_values.append(value)
        urgency, incremental = 3, False
        if priority_values:
            parsed = parse_priority_field(b", ".join(priority_values))
            if parsed is not None:
                urgency, incremental = parsed

        if method == "CONNECT":
            self.streams[request.stream_id] = WSStream(
                self.app,
                self.config,
                self.context,
                self.task_group,
                True,
                self.client,
                self.server,
                self.stream_send,
                request.stream_id,
            )
        else:
            self.streams[request.stream_id] = HTTPStream(
                self.app,
                self.config,
                self.context,
                self.task_group,
                True,
                self.client,
                self.server,
                self.stream_send,
                request.stream_id,
            )

        self.stream_buffers[request.stream_id] = StreamBuffer(self.context.event_class)
        push_id = None
        if request.stream_id & 0x3 == 0x3:
            # Server-initiated unidirectional streams carry pushes, the
            # push identifier is encoded in the stream identifier.
            push_id = (request.stream_id - 3) // 4
        # A PRIORITY_UPDATE received before the stream opened overrides
        # the initial Priority header field value.
        self.scheduler.add(
            request.stream_id, urgency, incremental, push_id=push_id
        )

        await self.streams[request.stream_id].handle(
            Request(
                stream_id=request.stream_id,
                headers=filter_pseudo_headers(request.headers),
                http_version="3",
                method=method,
                raw_path=raw_path,
                state=self.state,
            )
        )
        await self.context.mark_request()

    async def _create_server_push(
        self, stream_id: int, path: bytes, headers: list[tuple[bytes, bytes]]
    ) -> None:
        request_headers = [(b":method", b"GET"), (b":path", path)]
        request_headers.extend(headers)
        request_headers.extend(self.config.response_headers("h3"))
        try:
            push_stream_id = self.connection.send_push_promise(
                stream_id=stream_id, headers=request_headers
            )
        except NoAvailablePushIDError:
            # Client does not accept push promises or we are trying to
            # push on a push promises stream.
            pass
        else:
            event = HeadersReceived(
                stream_id=push_stream_id, stream_ended=True, headers=request_headers
            )
            await self._create_stream(event)
            await self.streams[event.stream_id].handle(EndBody(stream_id=event.stream_id))

    async def _close_stream(self, stream_id: int) -> None:
        stream = self.streams.pop(stream_id, None)
        if stream is not None:
            await stream.handle(StreamClosed(stream_id=stream_id))
        if stream_id in self.stream_buffers:
            await self.stream_buffers.pop(stream_id).close()
        self.scheduler.remove(stream_id)
        await self.has_data.set()

    async def close(self) -> None:
        self.closed = True
        for stream_id in list(self.streams.keys()):
            await self._close_stream(stream_id)
        await self.has_data.set()
