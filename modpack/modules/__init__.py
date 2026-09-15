"""ModPack modules (publishers, loggers, vision, cameras)."""
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent
MODPACK_CONFIG_PATH = _MODULES_DIR / "modpack_config.yaml"
