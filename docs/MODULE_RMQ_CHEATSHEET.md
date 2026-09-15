# Module RMQ Cheat Sheet

A module is a process that talks to other processes through an RMQ bus. There are three buses you might use, four call patterns you'll repeat, and two arguments (`n` and `timeout_s`) that don't behave the way you'd guess. Once you know these, you have everything you need.

This page is the source of truth. If you find yourself reaching for the `robotmq` source, something is missing here — please add it.

---

## The three buses

A coordinator-managed run has up to three RMQ servers running at known endpoints (configured in `modpack/modules/modpack_config.yaml` under `robotmq:`):

| Bus | What's on it | Reached at |
|-----|--------------|------------|
| **data** | Robot state, commands, neck pose, base pose — anything published every loop | `tcp://{host}:{port}` |
| **activation** | Lifecycle messages: "VP activated", "iPhone ready", episode start/stop | `tcp://{host}:{activation_port}` |
| **camera** | JPEG/raw frames from the head/wrist cameras | `tcp://{host}:{cam_port}` (== `cfg.camera_endpoint`) |

Your module receives a `cfg` (`ProcessRunConfig`) at startup. Read endpoints from `cfg.manager_cfg["robotmq"]` and `cfg.camera_endpoint` rather than hardcoding hosts/ports.

---

## Connecting

```python
from robotmq import RMQClient

rmq = cfg.manager_cfg["robotmq"]
data_client       = RMQClient("my_module_data",       f"tcp://{rmq['host']}:{rmq['port']}")
activation_client = RMQClient("my_module_activation", f"tcp://{rmq['host']}:{rmq['activation_port']}")
camera_client     = RMQClient("my_module_camera",     cfg.camera_endpoint)
```

`client_name` is for debug logging only — pick something that identifies your module. Only construct the clients you actually use.

---

## Two arguments to memorise

### `n` — how many messages

| `n` | Returns | Order |
|-----|---------|-------|
| `-1` | **single newest message** | length-1 list (or empty) |
| `0`  | **all messages currently held** | oldest first |
| `k>0` | **first `k` oldest** (clamped to what's available) | oldest first |

The result is always `(list_of_bytes, list_of_timestamps)` — both lists the same length.

**You almost always want `n=-1`.** That gives the latest state the publisher sent. Index it with `[0]` because the list is length 1:

```python
data, _ = client.peek_data(topic, n=-1, timeout_s=0.05)
if not data:
    return None
msg = deserialize(data[0])
```

Don't write `data[-1]` — it works (length-1 list) but reads as "newest of many," which is the wrong mental model.

### `timeout_s` — request budget, NOT a poll-wait

`timeout_s` is the time budget for the request itself to complete a network round-trip. It is **not** "wait this long for a message to appear." If the queue is empty, the call returns `([], [])` essentially instantly regardless of the timeout — the timeout governs how long the client will retry on transport failure before giving up.

| `timeout_s` | Behavior |
|-------------|----------|
| `0.0` | **Don't use this.** Means "zero budget for any reply" — the client retries forever. |
| small positive (e.g. `0.05`) | Standard. Plenty for a same-host call; aborts if the server is unreachable. |
| larger (e.g. `0.5`) | Use for cross-host calls or when transport hiccups are tolerable. |

Always pass a positive value. `timeout_s=0.05` is a fine default.

---

## The four call patterns

### 1. Read latest

```python
from robotmq.utils import deserialize

data, _ = client.peek_data(topic, n=-1, timeout_s=0.05)
if not data:
    return None
msg = deserialize(data[0])    # → dict
```

### 2. Drain a queue (consume, don't peek)

Use this for command queues where each message must be processed once and never re-read. Drain everything with `n=0` and act on the newest:

```python
data, _ = client.pop_data(topic, n=0, timeout_s=0.05)   # drain all queued messages
if data:
    cmd = deserialize(data[-1])                          # newest of what we drained
```

For most read paths you want `peek_data` (read latest, leave it in place). `pop_data` is for command/control queues.

### 3. Publish a message

```python
from robotmq.utils import serialize

payload = {"timestamp": time.time(), "value": 0.42}
client.put_data(topic, serialize(payload))
```

`serialize` is msgpack. The receiver calls `deserialize` to get the dict back.

### 4. Publish raw bytes (no serialization)

```python
client.put_data(topic, jpeg_bytes)   # publisher and consumer agree on format
```

For camera frames or anything already in a wire format. Don't double-serialize.

---

## Typed messages

For messages with a fixed schema (`NeckMessage`, `BaseProprioMessage`, etc.), use the helpers in `modpack/orchestration/message_formats.py`:

```python
from modpack.orchestration.message_formats import (
    NeckMessage,
    deserialize_neck_message,
    serialize_neck_message,
)

# read
data, _ = client.peek_data("neck_target_pose", n=-1, timeout_s=0.05)
if data:
    msg: NeckMessage = deserialize_neck_message(data[0])

# write
out = NeckMessage(timestamp=t, neck_pose=pose, data_valid=True)
client.put_data("neck_target_pose", serialize_neck_message(out))
```

Use typed messages whenever more than one module reads or writes the same topic — it forces both sides to agree on the schema.

---

## Topic registration

A module that **only reads** a topic doesn't need to register anything; it just calls `peek_data`.

A module that **publishes** a topic needs the topic to exist on the server. If the topic was registered by orchestration (via spec `setup_fn` calling `manager.register_data_topic(...)` or `manager.register_activation_topic(...)`), publish to it directly. Otherwise register it from your spec:

```python
# spec.py
def _setup(manager):
    manager.register_data_topic("my_topic", ttl_s=2.0)
    manager.register_activation_topic("my_status_topic", ttl_s=10.0)
```

`ttl_s` is how long the server keeps each message available to peekers. ~2s for high-rate state, ~10s for status messages, ~30s for lifecycle.

---

## Activation pattern

Most modules listen for an "activate"/"deactivate" command from the coordinator:

```python
from modpack.orchestration.message_formats import Topics

while running:
    data, _ = activation_client.peek_data(Topics.MY_MODULE_ACTIVATION, n=-1, timeout_s=0.05)
    if data:
        msg = deserialize(data[0])
        if msg["command"] == "activate":
            ...
        elif msg["command"] == "deactivate":
            ...
    time.sleep(0.05)
```

The coordinator publishes activation messages via the spec's `activation_fn` — see `modpack/modules/vision_pro/spec.py` for an example.

---

## Common bugs

**`timeout_s=0.0`.** Hangs the client in an infinite retry loop. Always pass a small positive value.

**Indexing with `[-1]` after `n=-1`.** Works (length-1 list) but masks intent. Use `[0]`.

**Mixed JSON and msgpack.** `serialize`/`deserialize` are msgpack. If a publisher does `json.dumps(...).encode()` and a consumer does `deserialize(...)`, decoding fails with confusing errors. Use `serialize` on both ends.

**Hardcoded ports.** Read from `cfg.manager_cfg["robotmq"]`. If a coworker changes the port in `modpack_config.yaml`, your module shouldn't need a code change.

**Calling `peek_data` on an unregistered topic.** Returns empty silently. Verify the publisher registered it.

**Decoding empty lists.** Always `if not data: return` before `data[0]` — `peek_data` returns `([], [])` when the queue is empty.

---

## Minimum viable module

```python
# runner.py
import sys
import time

from robotmq import RMQClient
from robotmq.utils import deserialize, serialize


def run_my_module(log_path, cfg):
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout

    rmq = cfg.manager_cfg["robotmq"]
    data = RMQClient("my_module", f"tcp://{rmq['host']}:{rmq['port']}")

    while True:
        raw, _ = data.peek_data("right_arm", n=-1, timeout_s=0.05)
        if raw:
            state = deserialize(raw[0])
            # ... do work ...
            out = {"timestamp": time.time(), "result": ...}
            data.put_data("my_output_topic", serialize(out))
        time.sleep(0.01)
```

That's the entire shape of an RMQ-using module. Lifecycle, activation gating, and multi-process spawn live in `spec.py` — see [ADD_A_MODULE.md](ADD_A_MODULE.md).
