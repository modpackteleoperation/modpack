# Data Tools

## Contents

| File | Role |
|------|------|
| `finalize_episode_video_spools.py` | encode deferred JPEG spools → `.mp4` |

## Usage

```bash
# single episode
python scripts/data_tools/finalize_episode_video_spools.py <episode_dir>
# whole run / parent dir (searched recursively)
python scripts/data_tools/finalize_episode_video_spools.py <run_dir>
```
