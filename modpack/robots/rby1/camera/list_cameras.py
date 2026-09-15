#!/usr/bin/env python3
from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _prepare_gi_import() -> None:
    """Expose system GI bindings/typelibs to compatible virtual environments."""

    dist_packages = Path("/usr/lib/python3/dist-packages")
    abi_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    gi_extension_matches = list(dist_packages.glob(f"gi/_gi.{abi_tag}-*.so"))
    if gi_extension_matches:
        site.addsitedir(str(dist_packages))

    typelib_dir = Path("/usr/local/lib/x86_64-linux-gnu/girepository-1.0")
    if typelib_dir.is_dir():
        existing = [p for p in os.environ.get("GI_TYPELIB_PATH", "").split(os.pathsep) if p]
        if str(typelib_dir) not in existing:
            os.environ["GI_TYPELIB_PATH"] = os.pathsep.join([str(typelib_dir), *existing])


_prepare_gi_import()

try:
    import gi

    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
except Exception as exc:
    raise SystemExit(
        "Failed to import Aravis via GI. "
        "This repo's venv expects system GI bindings in /usr/lib/python3/dist-packages "
        "and the Aravis typelib in /usr/local/lib/x86_64-linux-gnu/girepository-1.0. "
        f"Original error: {exc!r}"
    )

Aravis.update_device_list()
n_devices = Aravis.get_n_devices()

print(f"Found {n_devices} camera(s):")
for i in range(n_devices):
    device_id = Aravis.get_device_id(i)
    print(f"\nCamera {i}:")
    print(f"  Device ID: {device_id}")
    
    camera = Aravis.Camera.new(device_id)
    if camera:
        print(f"  Model: {camera.get_model_name()}")
        print(f"  Vendor: {camera.get_vendor_name()}")
        print(f"  Serial: {camera.get_device_serial_number()}")
        
        # Try to get MAC address
        try:
            mac = camera.get_string("GevMACAddress")
            print(f"  MAC Address: {mac}")
        except:
            pass
