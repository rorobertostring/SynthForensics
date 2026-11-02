# Caption Coverage

Content coverage of the 1,363 human-validated structured captions, one per source video.

This analysis is not reported in the paper.

The script groups the free-text caption fields into categories with explicit, case-insensitive keyword rules, defined at the top of [`01_caption_coverage.py`](01_caption_coverage.py):

- **Single-label fields** — scene (from `background`), `shot_type` and `camera_movement`. Each value takes the first matching category in the listed order, and values matching no rule are reported as `other`.
- **Multi-label fields** — actions and the presence of more than one person, matched over `short_description` and `dense_description`.

## Requirements

- Python 3.10+
- `pandas`

## Usage

```bash
python 01_caption_coverage.py --captions-dir /path/to/SynthForensics/captions
```

The script writes `caption_coverage.csv` with one row per (`field`, `category`), giving the number of captions (`count`) and their percentage over all captions (`pct`).
