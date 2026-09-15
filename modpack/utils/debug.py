from __future__ import annotations

from typing import Any


class DebugPrinter:
    def __init__(self, enabled: bool, prefix: str = ""):
        self.enabled = enabled
        self.prefix = prefix

    def log(self, *args: Any, **kwargs: Any) -> None:
        if not self.enabled:
            return

        if len(args) == 1 and callable(args[0]) and not kwargs:
            msg = args[0]()
            if self.prefix:
                print(self.prefix, msg)
            else:
                print(msg)
            return

        if self.prefix:
            print(self.prefix, *args, **kwargs)
        else:
            print(*args, **kwargs)
