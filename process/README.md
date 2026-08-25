# process

Video ingestion for `v2r_pipeline`: probe the source video's format with
`ffprobe` and extract frames with `ffmpeg` into a per-clip workspace.

## Install

```bash
cd process
uv sync
```

## Usage

```bash
uv run python -m process.cli \
  --video-path inputs/a.mp4 \
  --extract.fps 16
```

For `inputs/a.mp4` this creates:

```
outputs/a/process/
├── video_info.json      # source container + stream metadata (from ffprobe)
├── frames.json          # frame manifest (index / frame_filename / timestamp_sec)
├── config.json          # effective run config + tool versions
└── frames/
    ├── 000000.png
    ├── 000001.png
    └── ...
```

## Options

- `--extract.fps <Hz>` — uniform extraction rate; omit to keep every source frame.
- `--extract.width` / `--extract.height` — output size; omit to keep the source size (the other axis is preserved automatically).
- `--extract.format png|jpg` — output image format (default `png`).
- `--extract.overwrite` — clear an existing `process` workspace and re-run (default keeps existing outputs untouched).
- `--output-root <dir>` — root under which `<stem>/process/` is created (default `outputs`).

Downstream stages read `frames.json` (frame paths + timestamps) and
`video_info.json` rather than guessing filenames.
