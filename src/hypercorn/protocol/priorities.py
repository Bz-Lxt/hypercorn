from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hyperframe.frame import Frame, FRAMES

from ..typing import Event as IOEvent

BUFFER_HIGH_WATER = 2 * 2**14  # Twice the default max frame size (two frames worth)
BUFFER_LOW_WATER = BUFFER_HIGH_WATER / 2

# RFC 9218
DEFAULT_URGENCY = 3
URGENCIES = range(8)
H2_PRIORITY_UPDATE_FRAME_TYPE = 0x10
H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE = 0xF0700
H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE = 0xF0701
_H3_CONTROL_STREAM_TYPE = 0x00
_PRIORITY_FIELD_LIMIT = 2048


class BufferCompleteError(Exception):
    pass


class StreamBuffer:
    def __init__(self, event_class: type[IOEvent]) -> None:
        self.buffer = bytearray()
        self._complete = False
        self._is_empty = event_class()
        self._paused = event_class()

    async def drain(self) -> None:
        await self._is_empty.wait()

    def set_complete(self) -> None:
        self._complete = True

    async def close(self) -> None:
        self._complete = True
        self.buffer = bytearray()
        await self._is_empty.set()
        await self._paused.set()

    @property
    def complete(self) -> bool:
        return self._complete and len(self.buffer) == 0

    async def push(self, data: bytes) -> None:
        if self._complete:
            raise BufferCompleteError()
        self.buffer.extend(data)
        await self._is_empty.clear()
        if len(self.buffer) >= BUFFER_HIGH_WATER:
            await self._paused.wait()
            await self._paused.clear()

    async def pop(self, max_length: int) -> bytes:
        length = min(len(self.buffer), max_length)
        data = bytes(self.buffer[:length])
        del self.buffer[:length]
        if len(data) < BUFFER_LOW_WATER:
            await self._paused.set()
        if len(self.buffer) == 0:
            await self._is_empty.set()
        return data


# ---------------------------------------------------------------------------
# RFC 9218 Priority field value (Structured Fields Dictionary) parsing
# ---------------------------------------------------------------------------


def _split_top_level(text: str, separator: str) -> list[str] | None:
    """Split *text* on *separator* characters outside of quoted strings."""
    parts: list[str] = []
    start = 0
    in_quotes = False
    escaped = False
    for index, char in enumerate(text):
        if in_quotes:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_quotes = False
        elif char == '"':
            in_quotes = True
        elif char == separator:
            parts.append(text[start:index])
            start = index + 1
    if in_quotes:
        return None
    parts.append(text[start:])
    return parts


def _strip_ows(text: str) -> str:
    return text.strip(" \t")


def parse_priority_field(value: bytes) -> tuple[int, bool] | None:
    """Parse an RFC 9218 Priority field value.

    The value is a Structured Fields Dictionary carrying the urgency
    (``u``) and incremental (``i``) parameters. Unknown parameters,
    out of range values and values of an unexpected type are ignored as
    required by RFC 9218.

    Returns the ``(urgency, incremental)`` tuple, or ``None`` if the
    value cannot be parsed as a Dictionary, in which case the signal
    ought to be ignored.
    """
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError:
        return None

    urgency = DEFAULT_URGENCY
    incremental = False
    seen: set[str] = set()

    # An empty Dictionary is valid and resets the parameters to their
    # default values.
    if text.strip() == "":
        return urgency, incremental

    members = _split_top_level(text, ",")
    if members is None:
        return None
    for member in members:
        # Parameters (the part after ';') are valid but of no interest.
        parameter_parts = _split_top_level(member, ";")
        if parameter_parts is None:
            return None
        item = _strip_ows(parameter_parts[0])
        if item == "":
            return None

        if "=" in item:
            key, raw_value = item.split("=", 1)
        else:
            key, raw_value = item, ""
        key = _strip_ows(key).lower()
        raw_value = _strip_ows(raw_value)
        if key == "":
            return None
        # Only the first instance of a key is meaningful.
        if key in seen:
            continue
        seen.add(key)

        if key == "u":
            # The urgency is an Integer in the range 0-7.
            digits = raw_value[1:] if raw_value[:1] == "-" else raw_value
            if digits.isdigit() and digits != "":
                parsed_urgency = int(raw_value)
                if 0 <= parsed_urgency <= 7:
                    urgency = parsed_urgency
        elif key == "i":
            # The incremental parameter is a Boolean; a bare key implies
            # a true value.
            if raw_value == "":
                incremental = True
            elif raw_value == "?1":
                incremental = True
            elif raw_value == "?0":
                incremental = False
    return urgency, incremental


# ---------------------------------------------------------------------------
# RFC 9218 response scheduler
# ---------------------------------------------------------------------------


@dataclass
class _StreamPriority:
    stream_id: int
    sequence: int
    urgency: int
    incremental: bool
    blocked: bool = True


class PriorityScheduler:
    """Schedule response data following the guidance of RFC 9218 section 10.

    * Streams are served in order of urgency (0 before 7).
    * Non-incremental streams of the same urgency are served one at a
      time, in the order the requests were made.
    * Incremental streams of the same urgency progress fairly, in a
      round-robin fashion.
    * A stream without data, with no flow-control window or that has
      been closed is skipped without hindering any other stream.
    """

    def __init__(self, pending_limit: int = 100) -> None:
        self._streams: dict[int, _StreamPriority] = {}
        self._buckets: dict[int, dict[int, _StreamPriority]] = {
            urgency: {} for urgency in URGENCIES
        }
        # Most recent PRIORITY_UPDATE received before the stream opens,
        # keyed by the stream identifier.
        self._pending: dict[int, tuple[int, bool]] = {}
        self._pending_limit = pending_limit
        # Round-robin cursor (sequence number) per urgency bucket.
        self._cursors: dict[int, int] = {}
        self._sequence = 0

    def add(
        self,
        stream_id: int,
        urgency: int = DEFAULT_URGENCY,
        incremental: bool = False,
        push_id: int | None = None,
    ) -> None:
        """Register a newly opened stream.

        A buffered PRIORITY_UPDATE for the stream (or push) overrides the
        supplied initial values, which typically come from the Priority
        header field.
        """
        if stream_id in self._streams:
            return
        pending_key = 4 * push_id + 3 if push_id is not None else stream_id
        pending = self._pending.pop(pending_key, None)
        if pending is not None:
            urgency, incremental = pending
        state = _StreamPriority(
            stream_id=stream_id,
            sequence=self._sequence,
            urgency=urgency,
            incremental=incremental,
        )
        self._sequence += 1
        self._streams[stream_id] = state
        self._buckets[urgency][stream_id] = state

    def set_priority(self, stream_id: int, urgency: int, incremental: bool) -> None:
        """Apply a PRIORITY_UPDATE, whether or not the stream is open.

        Updates of streams that are sending take immediate effect, and
        only the most recent update for a stream that has yet to open is
        retained (RFC 9218 section 7).
        """
        state = self._streams.get(stream_id)
        if state is None:
            if stream_id not in self._pending and len(self._pending) >= self._pending_limit:
                # Bound the resources spent remembering idle streams by
                # discarding the oldest remembered update.
                self._pending.pop(next(iter(self._pending)))
            self._pending[stream_id] = (urgency, incremental)
            return
        if state.urgency != urgency:
            self._buckets[state.urgency].pop(stream_id, None)
            state.urgency = urgency
            self._buckets[urgency][stream_id] = state
        state.incremental = incremental

    def block(self, stream_id: int) -> None:
        state = self._streams.get(stream_id)
        if state is not None:
            state.blocked = True

    def unblock(self, stream_id: int) -> None:
        state = self._streams.get(stream_id)
        if state is not None:
            state.blocked = False

    def remove(self, stream_id: int) -> None:
        state = self._streams.pop(stream_id, None)
        if state is not None:
            self._buckets[state.urgency].pop(stream_id, None)
        self._pending.pop(stream_id, None)

    def next_ready(self) -> int | None:
        """Return the stream that should send data next, or ``None``."""
        for urgency in URGENCIES:
            bucket = self._buckets[urgency]
            ready = [state for state in bucket.values() if not state.blocked]
            if not ready:
                continue

            # Non-incremental responses of the same urgency are served one
            # at a time in request order. Only when no such response can
            # make progress do the incremental responses get a turn.
            non_incremental = [state for state in ready if not state.incremental]
            if non_incremental:
                chosen = min(non_incremental, key=lambda state: state.sequence)
                return chosen.stream_id

            # Incremental responses share the bandwidth, progressing in a
            # round-robin order that follows request order.
            cursor = self._cursors.get(urgency)
            if cursor is None:
                chosen = min(ready, key=lambda state: state.sequence)
            else:
                ahead = sorted(
                    (state for state in ready if state.sequence > cursor),
                    key=lambda state: state.sequence,
                )
                chosen = ahead[0] if ahead else min(
                    ready, key=lambda state: state.sequence
                )
            self._cursors[urgency] = chosen.sequence
            return chosen.stream_id
        return None

    @property
    def pending(self) -> dict[int, tuple[int, bool]]:
        return self._pending


# ---------------------------------------------------------------------------
# HTTP/2 PRIORITY_UPDATE frame (RFC 9218 section 7.1)
# ---------------------------------------------------------------------------


class PriorityUpdateReceived:
    """Event emitted when an HTTP/2 PRIORITY_UPDATE frame is received."""

    def __init__(
        self,
        prioritized_stream_id: int,
        priority_field_value: bytes,
    ) -> None:
        self.prioritized_stream_id = prioritized_stream_id
        self.priority_field_value = priority_field_value


class _PriorityUpdateFrame(Frame):
    """hyperframe implementation of the HTTP/2 PRIORITY_UPDATE frame."""

    type = H2_PRIORITY_UPDATE_FRAME_TYPE
    defined_flags: list[Any] = []

    def __init__(
        self,
        stream_id: int = 0,
        prioritized_stream_id: int = 0,
        priority_field_value: bytes = b"",
        **kwargs: Any,
    ) -> None:
        super().__init__(stream_id, **kwargs)
        self.prioritized_stream_id = prioritized_stream_id
        self.priority_field_value = priority_field_value

    def parse_body(self, data: memoryview) -> None:
        raw = bytes(data)
        if len(raw) >= 4:
            self.prioritized_stream_id = int.from_bytes(raw[:4], "big") & 0x7FFFFFFF
            self.priority_field_value = raw[4:]
        self.body_len = len(raw)

    def serialize_body(self) -> bytes:
        return self.prioritized_stream_id.to_bytes(4, "big") + self.priority_field_value


if H2_PRIORITY_UPDATE_FRAME_TYPE not in FRAMES:
    FRAMES[H2_PRIORITY_UPDATE_FRAME_TYPE] = _PriorityUpdateFrame


def install_h2_priority_update_frame(connection: Any) -> None:
    """Teach an h2 connection to surface PRIORITY_UPDATE frames.

    The h2/hyperframe versions Hypercorn depends on do not know about the
    frame defined in RFC 9218, so it is registered with hyperframe and a
    matching dispatcher is added to the connection.
    """

    def _receive_priority_update(frame: _PriorityUpdateFrame) -> tuple[list, list]:
        events: list = []
        # The frame header Stream Identifier MUST be 0x0, and the
        # prioritized stream MUST NOT be 0x0 (RFC 9218 section 7.1).
        if frame.stream_id != 0 or frame.prioritized_stream_id == 0:
            import h2.exceptions

            raise h2.exceptions.ProtocolError("Invalid PRIORITY_UPDATE frame")
        events.append(
            PriorityUpdateReceived(
                frame.prioritized_stream_id,
                frame.priority_field_value,
            )
        )
        return [], events

    connection._frame_dispatch_table[_PriorityUpdateFrame] = _receive_priority_update


# ---------------------------------------------------------------------------
# HTTP/3 PRIORITY_UPDATE frames (RFC 9218 section 7.2)
# ---------------------------------------------------------------------------


def pull_uint_var(data: bytes, offset: int) -> tuple[int, int] | None:
    """Parse a QUIC variable-length integer starting at *offset*."""
    if offset >= len(data):
        return None
    first = data[offset]
    length = 1 << (first >> 6)
    if len(data) - offset < length:
        return None
    value = first & 0x3F
    for byte in data[offset + 1 : offset + length]:
        value = (value << 8) | byte
    return value, offset + length


@dataclass
class _UniStreamState:
    phase: str = "stream_type"
    carry: bytearray = field(default_factory=bytearray)
    frame_type: int = 0
    frame_size: int = 0
    frame_skipped: int = 0
    priority_payload: bytearray | None = None


class H3ControlStreamParser:
    """Find PRIORITY_UPDATE frames sent on the client HTTP/3 control stream.

    aioquic discards unknown frame types on control streams, so the raw
    data of client-initiated unidirectional streams is scanned here
    before being handed to aioquic. Only the latest value per stream is
    relevant, hence payloads of other (and overlong) frame types are
    streamed past without buffering.
    """

    def __init__(
        self,
        on_priority_update: Callable[[int, bytes, bool], None],
    ) -> None:
        self._on_priority_update = on_priority_update
        self._streams: dict[int, _UniStreamState] = {}

    def feed(self, stream_id: int, data: bytes) -> None:
        # Client-initiated unidirectional streams have an identifier with
        # the two least significant bits set to 0b10.
        if stream_id & 0x3 != 0x2:
            return
        state = self._streams.get(stream_id)
        if state is None:
            state = _UniStreamState()
            self._streams[stream_id] = state
        self._feed_state(state, data)

    def _feed_state(self, state: _UniStreamState, data: bytes) -> None:
        offset = 0
        if state.phase == "stream_type":
            state.carry.extend(data)
            parsed = pull_uint_var(bytes(state.carry), 0)
            if parsed is None:
                if len(state.carry) > 8:
                    state.phase = "ignore"
                return
            stream_type, consumed = parsed
            del state.carry[:consumed]
            if stream_type != _H3_CONTROL_STREAM_TYPE:
                state.phase = "ignore"
                return
            state.phase = "frame_header"
            data = bytes(state.carry)
            state.carry = bytearray()
            offset = 0

        if state.phase == "ignore":
            return

        while offset < len(data):
            if state.phase == "frame_header":
                state.carry.extend(data[offset:])
                offset = len(data)
                parsed_type = pull_uint_var(bytes(state.carry), 0)
                if parsed_type is None:
                    if len(state.carry) > 8:
                        # A single variable-length integer never exceeds
                        # 8 bytes; anything longer is malformed.
                        state.phase = "ignore"
                    return
                frame_type, type_end = parsed_type
                parsed_length = pull_uint_var(bytes(state.carry), type_end)
                if parsed_length is None:
                    if len(state.carry) - type_end > 8:
                        state.phase = "ignore"
                    return
                frame_size, length_end = parsed_length
                state.frame_type = frame_type
                state.frame_size = frame_size
                state.frame_skipped = 0
                state.priority_payload = (
                    bytearray() if frame_type in self._priority_frame_types() else None
                )
                del state.carry[:length_end]
                data = bytes(state.carry)
                state.carry = bytearray()
                offset = 0
                state.phase = "frame_payload"
                if state.frame_size == 0:
                    self._frame_complete(state)
                    state.phase = "frame_header"
                continue

            if state.phase == "frame_payload":
                remaining = state.frame_size - state.frame_skipped
                chunk = data[offset : offset + remaining]
                offset += len(chunk)
                if state.priority_payload is not None:
                    if (
                        len(state.priority_payload) + len(chunk)
                        <= _PRIORITY_FIELD_LIMIT
                    ):
                        state.priority_payload.extend(chunk)
                    else:
                        # The payload is unreasonably large; stop buffering.
                        state.priority_payload = None
                state.frame_skipped += len(chunk)
                if state.frame_skipped == state.frame_size:
                    self._frame_complete(state)
                    state.phase = "frame_header"
                continue

    @staticmethod
    def _priority_frame_types() -> frozenset[int]:
        return frozenset(
            {
                H3_PRIORITY_UPDATE_REQUEST_FRAME_TYPE,
                H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE,
            }
        )

    def _frame_complete(self, state: _UniStreamState) -> None:
        if state.priority_payload is None:
            return
        parsed = pull_uint_var(bytes(state.priority_payload), 0)
        if parsed is None:
            return
        element_id, value_start = parsed
        field_value = bytes(state.priority_payload[value_start:])
        is_push = state.frame_type == H3_PRIORITY_UPDATE_PUSH_FRAME_TYPE
        if is_push:
            element_id = 4 * element_id + 3
        self._on_priority_update(element_id, field_value, is_push)
