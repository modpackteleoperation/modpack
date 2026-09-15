"""Client-side proxy to the Base RPC server.

The 250 Hz Vehicle control loop runs in a dedicated OS process (base_server.py,
spawned by BaseProcess.start_base). BaseProcess (which also hosts the
Flask/SocketIO WebXR server) talks to it through this proxy so the control loop
keeps its OWN GIL and is never starved by the web server. The proxy exposes the
same surface BaseProcess used on the old in-process Base: reset / activate /
deactivate / emergency_stop / execute_action / get_state / close.

Note: dev/teleop ran an equivalent BaseManager RPC but as an in-process
serve_forever thread (loopback) — that shared the Flask GIL and did NOT isolate
the control loop. Here the server is a real separate process, which is the point.
"""
import time
from multiprocessing.managers import BaseManager as MPBaseManager

from .constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY


class _BaseManagerClient(MPBaseManager):
    pass


_BaseManagerClient.register('Base')


def connect_base(host=BASE_RPC_HOST, port=BASE_RPC_PORT, authkey=RPC_AUTHKEY,
                 timeout=15.0):
    """Connect to the Base RPC server and return a Base proxy.

    Retries until the server process has bound its socket (it needs a moment to
    import phoenix6 and start), up to `timeout` seconds.
    """
    manager = _BaseManagerClient(address=(host, port), authkey=authkey)
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            manager.connect()
            return manager.Base()
        except Exception as e:  # ConnectionRefusedError until the server is up
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"Could not connect to Base RPC server at {host}:{port}: {last_err}")
