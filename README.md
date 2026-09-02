# Depth-Anything Video API

Converts a video into a **black-and-white depth video** — grayscale frames
where brightness encodes relative distance to the camera — using
[Depth-Anything V2](https://huggingface.co/depth-anything). This is the
format expected by depth-conditioned video generation models such as
**Seedance**.

It ships as:

- **`video_to_depth.py`** — a standalone CLI script, no server needed.
- **A FastAPI service** (`api/`) — upload a video, poll a job id, download
  the finished depth video from a URL when it's ready.
- **`setup.sh` / `autodl_start.sh`** — one-command setup and launch, tuned
  for rented GPU boxes on [AutoDL](https://www.autodl.com).

## Repo layout

```
core/depth_engine.py   Loads the Depth-Anything V2 model, runs batched inference
core/depth_video.py     Video -> depth-video pipeline (shared by CLI and API)
video_to_depth.py       Standalone CLI entry point
api/main.py             FastAPI app (upload / status / download endpoints)
api/jobs.py             In-process job queue (single GPU worker)
api/settings.py         Env-var driven configuration
setup.sh                 Installs dependencies
autodl_start.sh          AutoDL-specific setup + background launch
baidu_sync.sh             Push/pull files to/from Baidu Pan via rclone
Dockerfile                Container build
```

## How the conversion works

For each frame: run Depth-Anything V2, get a raw relative-depth map, resize
it back to the source resolution, then normalize it to 0-255 grayscale. The
min/max used for normalization is smoothed across frames with an EMA
(`--smoothing`, default `0.9`) instead of recomputed per frame, which
removes most of the flicker you'd otherwise get from independent per-frame
normalization. The encode uses libx264 at a high quality (CRF 16) since the
output is a conditioning input, not something meant to be visually
compressed.

By default **near = white, far = black**. Pass `--invert` /
`invert=true` if the model you're feeding expects the opposite convention —
check a few frames of Seedance's own depth-conditioning examples if you're
unsure, and toggle this flag accordingly.

## Quick start (local / any machine with a GPU)

```bash
git clone <this-repo> && cd Depth-anything
./setup.sh
```

CLI:

```bash
python video_to_depth.py input.mp4 output_depth.mp4
python video_to_depth.py input.mp4 output_depth.mp4 --model base --max-side 896 --invert
```

API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Quick start on AutoDL

1. Rent a GPU instance using a **PyTorch** base image (any recent
   CUDA/PyTorch image works — `setup.sh` detects the preinstalled torch and
   won't reinstall it).
2. Open a JupyterLab terminal or SSH in, then:

   ```bash
   git clone <this-repo> && cd Depth-anything
   ./autodl_start.sh
   ```

   This points model weights and job storage at the persistent
   `/root/autodl-tmp` data disk (survives stop/start cycles, unlike the
   system disk), optionally uses the `hf-mirror.com` mirror for faster
   model downloads, and starts the API in the background
   (`logs/api.log`, `logs/api.pid`).

3. Expose port `8000` to the outside world using AutoDL's **Custom
   Service** (自定义服务) port proxy for the instance, or tunnel it:

   ```bash
   ssh -L 8000:localhost:8000 <your-autodl-ssh-target>
   ```

## Transferring files via Baidu Pan (rclone)

Baidu Pan (百度网盘) has no rsync or SSH access, so plain `rsync` can't talk
to it. `baidu_sync.sh` wraps [`rclone`](https://rclone.org)'s `baidunetdisk`
backend instead, which gives the same delta-sync behaviour rsync would
(only changed files transfer, checksums, `--dry-run` support) over Baidu's
API. Handy for moving source videos onto a rented GPU box and finished
depth videos back off it.

```bash
./baidu_sync.sh install     # install rclone if missing
./baidu_sync.sh configure   # one-time: link the 'baidupan' rclone remote

./baidu_sync.sh pull inputs storage/uploads   # fetch source videos to process
./baidu_sync.sh push storage/outputs results  # upload finished depth videos
./baidu_sync.sh ls                            # list what's on Baidu Pan
```

Baidu's login is OAuth and needs a browser, which a headless rented box
usually doesn't have — `configure` walks through `rclone config`'s
"authorize on another machine, paste the token back" flow for that case.
`BAIDU_REMOTE` / `BAIDU_REMOTE_DIR` env vars override the remote name
(default `baidupan`) and the base remote path (default `depth-anything`).

## API usage

```bash
# Upload a video, get back a job id
curl -X POST "http://localhost:8000/videos?invert=false" -F "file=@input.mp4"
# {"id": "a1b2c3...", "status": "pending", ...}

# Poll status / progress
curl "http://localhost:8000/videos/a1b2c3..."
# {"status": "processing", "progress": 0.42, ...}

# Download once status == "completed"
curl -OJ "http://localhost:8000/videos/a1b2c3.../download"
```

Endpoints:

| Method | Path                     | Description                          |
|--------|--------------------------|---------------------------------------|
| POST   | `/videos`                | Upload a video, returns a job         |
| GET    | `/videos`                | List all jobs                         |
| GET    | `/videos/{id}`           | Job status / progress / download_url  |
| GET    | `/videos/{id}/download`  | Download the finished depth video     |
| DELETE | `/videos/{id}`           | Delete a job and its files            |
| GET    | `/health`                | Liveness, model id, device            |

Upload query params: `invert` (bool), `max_side` (int, downscale before
inference for speed), `smoothing` (float, EMA factor).

Jobs and their files are kept in memory / on local disk only (no
database), and are auto-deleted after `JOB_RETENTION_HOURS` (default 24).
This is intentionally simple — it's meant to run as a single process on a
single rented GPU, not as durable multi-tenant infrastructure.

## Configuration (environment variables)

| Variable             | Default                                              | Meaning                                    |
|-----------------------|-------------------------------------------------------|---------------------------------------------|
| `DEPTH_MODEL`          | `depth-anything/Depth-Anything-V2-Small-hf`           | HF model id (swap for `-Base-hf` / `-Large-hf`) |
| `STORAGE_DIR`          | `/root/autodl-tmp/depth-anything-storage` or `./storage` | Where uploads/outputs are stored        |
| `HF_HOME`              | `/root/autodl-tmp/hf-cache` (on AutoDL) or HF default | Hugging Face model cache dir                |
| `MAX_UPLOAD_MB`        | `500`                                                 | Max upload size                             |
| `BATCH_SIZE`           | `4`                                                   | Frames per inference batch                  |
| `MAX_SIDE`             | unset (no resize)                                     | Downscale longest side before inference     |
| `INVERT_DEFAULT`       | `false`                                               | Default value of the `invert` query param   |
| `SMOOTHING`            | `0.9`                                                 | Default temporal EMA smoothing factor       |
| `MAX_CONCURRENT_JOBS`  | `1`                                                    | Parallel GPU jobs (keep at 1 on a single GPU) |
| `JOB_RETENTION_HOURS`  | `24`                                                    | How long finished jobs' files are kept      |
| `HOST` / `PORT`        | `0.0.0.0` / `8000`                                     | Bind address for uvicorn                    |

## Docker

```bash
docker build -t depth-anything-api .
docker run --gpus all -p 8000:8000 -v $(pwd)/storage:/app/storage depth-anything-api
```
