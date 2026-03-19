# Not tested yet. Proceed at your own risk - 3/12/2026 - Tsubasa Kato - Inspire Search Corp.
import json
import os
import signal
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

app = Flask(__name__)

CRAWL_OUTPUTS_DIR = Path(__file__).parent / "crawl_outputs"
CRAWL_OUTPUTS_DIR.mkdir(exist_ok=True)

CRAWLER_SCRIPT = Path(__file__).parent / "quantum-decision-crawler4.py"

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration(job: dict) -> float | None:
    if not job.get("start_time"):
        return None
    # For terminal statuses, use end_time; for active statuses, use current time
    terminal = job.get("status") in ("completed", "failed", "stopped")
    end = job.get("end_time") or (None if terminal else _utcnow_iso())
    if not end:
        return None
    try:
        start = datetime.fromisoformat(job["start_time"])
        stop = datetime.fromisoformat(end)
        return round((stop - start).total_seconds(), 1)
    except Exception:
        return None


def _job_summary(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "paused": job.get("paused", False),
        "start_time": job.get("start_time"),
        "end_time": job.get("end_time"),
        "duration": _duration(job),
        "params": job.get("params", {}),
        "output_dir": str(job.get("output_dir", "")),
    }


def _get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404, description=f"Job {job_id!r} not found.")
    return job


def _parse_jsonl(path: str) -> list[dict]:
    results = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return results


def _to_iso_timestamp(ts_val) -> str | None:
    """
    Convert epoch seconds (float/int) to ISO-8601 (UTC). If conversion fails,
    return None so the UI can fall back to displaying '—'.
    """
    if ts_val is None:
        return None

    try:
        # Most crawler outputs use epoch seconds.
        if isinstance(ts_val, (int, float)):
            return datetime.fromtimestamp(float(ts_val), tz=timezone.utc).isoformat()

        if isinstance(ts_val, str):
            s = ts_val.strip()
            if not s:
                return None
            # Sometimes ts may be numeric-as-string.
            try:
                return datetime.fromtimestamp(float(s), tz=timezone.utc).isoformat()
            except Exception:
                # If it's already ISO-ish, keep it for the UI.
                if "T" in s and "-" in s:
                    return s
    except Exception:
        return None

    return None


def _normalise_explorer_row(row: dict) -> dict:
    """
    Results Explorer expects:
      - links_found / links_enqueued
      - fetch_time (seconds)
      - timestamp (ISO string)

    The crawlers may emit:
      - out_links_found / out_links_enqueued
      - fetch_seconds
      - ts (epoch seconds)
    """
    r = dict(row)

    if r.get("links_found") is None and "out_links_found" in r:
        r["links_found"] = r.get("out_links_found")
    if r.get("links_enqueued") is None and "out_links_enqueued" in r:
        r["links_enqueued"] = r.get("out_links_enqueued")

    if r.get("fetch_time") is None and "fetch_seconds" in r:
        r["fetch_time"] = r.get("fetch_seconds")
    if isinstance(r.get("fetch_time"), str):
        try:
            r["fetch_time"] = float(r["fetch_time"])
        except Exception:
            pass

    if r.get("timestamp") is None:
        # Crawlers typically store this as epoch seconds in `ts`.
        ts_val = r.get("ts")
        iso = _to_iso_timestamp(ts_val)
        if iso:
            r["timestamp"] = iso

    return r


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_crawler(job_id: str, cmd: list[str]) -> None:
    import subprocess

    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["start_time"] = _utcnow_iso()

    log_lines: list[str] = jobs[job_id]["log_lines"]
    log_lines.append(f"[QuantumCrawler] Starting job {job_id}")
    log_lines.append(f"[QuantumCrawler] Command: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with jobs_lock:
            jobs[job_id]["process"] = proc
        for line in proc.stdout:
            log_lines.append(line.rstrip("\n"))
        proc.wait()
        exit_code = proc.returncode
    except Exception as exc:
        log_lines.append(f"[QuantumCrawler] ERROR launching subprocess: {exc}")
        exit_code = 1

    with jobs_lock:
        # Don't overwrite status if the job was stopped by the user
        if jobs[job_id].get("status") == "stopped":
            log_lines.append(
                f"[QuantumCrawler] Job finished – exit code {exit_code}, status=stopped"
            )
            return
        jobs[job_id]["end_time"] = _utcnow_iso()
        jobs[job_id]["status"] = "completed" if exit_code == 0 else "failed"
        log_lines.append(
            f"[QuantumCrawler] Job finished – exit code {exit_code}, "
            f"status={jobs[job_id]['status']}"
        )


def _build_command(params: dict, output_dir: Path, seeds_file: str) -> list[str]:
    cmd = [sys.executable, str(CRAWLER_SCRIPT)]

    cmd += ["--seeds", seeds_file]
    cmd += ["--out", str(output_dir / "crawl.jsonl")]

    int_params = {
        "max_pages": "--max-pages",
        "max_depth": "--max-depth",
        "workers": "--workers",
        "shots": "--shots",
        "robots_ttl": "--robots-ttl",
        "sitemap_max_urls": "--sitemap-max-urls",
        "sitemap_index_depth": "--sitemap-index-depth",
        "retries": "--retries",
        "dead_end_threshold": "--dead-end-threshold",
        "dead_ends_before_boost": "--dead-ends-before-boost",
    }
    float_params = {
        "delay": "--delay",
        "connect_timeout": "--connect-timeout",
        "read_timeout": "--read-timeout",
        "total_timeout": "--total-timeout",
        "dns_timeout": "--dns-timeout",
        "backoff": "--backoff",
        "quantum_weight": "--quantum-weight",
        "heuristic_weight": "--heuristic-weight",
        "exploration_weight": "--exploration-weight",
        "exploration_temperature": "--exploration-temperature",
        "backtrack_boost": "--backtrack-boost",
        "compare_duration": "--compare-duration",
    }
    str_params = {
        "allow": "--allow",
        "deny": "--deny",
        "compare_csv": "--compare-csv",
        "compare_json": "--compare-json",
    }

    for key, flag in int_params.items():
        val = params.get(key)
        if val not in (None, "", "None"):
            cmd += [flag, str(int(val))]

    for key, flag in float_params.items():
        val = params.get(key)
        if val not in (None, "", "None"):
            cmd += [flag, str(float(val))]

    for key, flag in str_params.items():
        val = params.get(key)
        if val not in (None, "", "None", []):
            if isinstance(val, list):
                for item in val:
                    cmd += [flag, item]
            else:
                cmd += [flag, val]

    if params.get("debug"):
        cmd.append("--debug")
    if params.get("gpu_math"):
        cmd.append("--gpu-math")
    if params.get("no_aer_gpu"):
        cmd.append("--no-aer-gpu")
    if params.get("force_curl"):
        cmd.append("--force-curl")
    if params.get("use_base_crawler"):
        cmd.append("--use-base-crawler")
    if params.get("compare_algorithms"):
        cmd.append("--compare-algorithms")

    # robots / sitemaps (default True flags with --no- variants)
    respect_robots = params.get("respect_robots", True)
    if str(respect_robots).lower() in ("false", "0", "no"):
        cmd.append("--no-respect-robots")
    else:
        cmd.append("--respect-robots")

    use_sitemaps = params.get("use_sitemaps", True)
    if str(use_sitemaps).lower() in ("false", "0", "no"):
        cmd.append("--no-use-sitemaps")
    else:
        cmd.append("--use-sitemaps")

    # override compare output paths to job output dir
    if params.get("compare_algorithms"):
        csv_name = params.get("compare_csv") or "comparison.csv"
        json_name = params.get("compare_json") or "comparison.json"
        # already added via str_params above; we need to use job-scoped paths
        # Remove any existing --compare-csv / --compare-json and re-add
        for flag in ("--compare-csv", "--compare-json"):
            while flag in cmd:
                idx = cmd.index(flag)
                del cmd[idx : idx + 2]
        cmd += ["--compare-csv", str(output_dir / csv_name)]
        cmd += ["--compare-json", str(output_dir / json_name)]

    return cmd


def _start_job(params: dict, seeds_text: str | None = None, seeds_file_path: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    output_dir = CRAWL_OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve seeds file
    if seeds_file_path:
        seeds_file = seeds_file_path
    elif seeds_text:
        sf = output_dir / "seeds.txt"
        sf.write_text(seeds_text, encoding="utf-8")
        seeds_file = str(sf)
    else:
        seeds_file = str(Path(__file__).parent / "seeds.txt")

    out_jsonl = str(output_dir / "crawl.jsonl")
    compare_csv = str(output_dir / (params.get("compare_csv") or "comparison.csv"))
    compare_json = str(output_dir / (params.get("compare_json") or "comparison.json"))

    job: dict = {
        "id": job_id,
        "status": "pending",
        "paused": False,
        "start_time": None,
        "end_time": None,
        "log_lines": [],
        "params": params,
        "output_dir": str(output_dir),
        "seeds_file": seeds_file,
        "out_jsonl": out_jsonl,
        "compare_csv": compare_csv,
        "compare_json": compare_json,
        "process": None,
    }

    with jobs_lock:
        jobs[job_id] = job

    cmd = _build_command(params, output_dir, seeds_file)
    thread = threading.Thread(target=_run_crawler, args=(job_id, cmd), daemon=True)
    thread.start()

    return job_id


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    with jobs_lock:
        all_jobs = list(jobs.values())
    all_jobs.sort(key=lambda j: j.get("start_time") or "", reverse=True)
    return render_template("dashboard.html", jobs=all_jobs)


@app.route("/crawl/new", methods=["GET"])
def crawl_new():
    return render_template("crawl_new.html")


@app.route("/crawl/new", methods=["POST"])
def crawl_new_post():
    form = request.form
    files = request.files

    # Seeds
    seeds_text: str | None = None

    uploaded = files.get("seeds_file")
    if uploaded and uploaded.filename:
        # Read file content and pass as seeds_text to avoid orphaned temp dirs
        seeds_text = uploaded.read().decode("utf-8", errors="replace").strip() or None
    else:
        seeds_text = form.get("seeds_text", "").strip() or None

    crawl_mode = form.get("crawl_mode", "hybrid")
    use_base_crawler = crawl_mode == "base"
    compare_algorithms = crawl_mode == "compare"

    def _int(key, default=None):
        v = form.get(key, "").strip()
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    def _float(key, default=None):
        v = form.get(key, "").strip()
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _bool(key):
        return form.get(key) in ("on", "true", "1", "yes")

    params = {
        "max_pages": _int("max_pages", 500),
        "max_depth": _int("max_depth", 3),
        "workers": _int("workers", 16),
        "delay": _float("delay", 0.6),
        "shots": _int("shots", 128),
        "debug": _bool("debug"),
        "gpu_math": _bool("gpu_math"),
        "no_aer_gpu": _bool("no_aer_gpu"),
        "allow": form.get("allow", "").strip() or None,
        "deny": form.get("deny", "").strip() or None,
        "quantum_weight": _float("quantum_weight", 0.45),
        "heuristic_weight": _float("heuristic_weight", 0.35),
        "exploration_weight": _float("exploration_weight", 0.20),
        "exploration_temperature": _float("exploration_temperature", 0.4),
        "dead_end_threshold": _int("dead_end_threshold", 2),
        "backtrack_boost": _float("backtrack_boost", 0.25),
        "dead_ends_before_boost": _int("dead_ends_before_boost", 1),
        "connect_timeout": _float("connect_timeout", 5.0),
        "read_timeout": _float("read_timeout", 25.0),
        "total_timeout": _float("total_timeout", 12.0),
        "dns_timeout": _float("dns_timeout", 4.0),
        "retries": _int("retries", 1),
        "backoff": _float("backoff", 0.6),
        "force_curl": _bool("force_curl"),
        "respect_robots": not _bool("no_respect_robots"),
        "use_sitemaps": not _bool("no_use_sitemaps"),
        "robots_ttl": _int("robots_ttl", 21600),
        "sitemap_max_urls": _int("sitemap_max_urls", 5000),
        "sitemap_index_depth": _int("sitemap_index_depth", 2),
        "use_base_crawler": use_base_crawler,
        "compare_algorithms": compare_algorithms,
        "compare_duration": _float("compare_duration", 180.0),
        "compare_csv": form.get("compare_csv", "comparison.csv").strip() or "comparison.csv",
        "compare_json": form.get("compare_json", "comparison.json").strip() or "comparison.json",
        "crawl_mode": crawl_mode,
    }

    job_id = _start_job(params, seeds_text=seeds_text)
    return redirect(url_for("crawl_status", job_id=job_id))


@app.route("/crawl/<job_id>")
def crawl_status(job_id: str):
    job = _get_job(job_id)
    return render_template("crawl_status.html", job=job)


@app.route("/crawl/<job_id>/results")
def crawl_results(job_id: str):
    job = _get_job(job_id)
    return render_template("crawl_results.html", job=job)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    with jobs_lock:
        all_jobs = list(jobs.values())
    return jsonify([_job_summary(j) for j in all_jobs])


@app.route("/api/jobs", methods=["POST"])
def api_jobs_post():
    body = request.get_json(force=True, silent=True) or {}
    params = body.get("params", body)
    seeds_text = body.get("seeds_text")
    job_id = _start_job(params, seeds_text=seeds_text)
    return jsonify({"job_id": job_id}), 201


@app.route("/api/jobs/<job_id>", methods=["GET"])
def api_job_detail(job_id: str):
    job = _get_job(job_id)
    data = _job_summary(job)
    data["log_lines"] = job["log_lines"]
    return jsonify(data)


@app.route("/api/jobs/<job_id>/logs", methods=["GET"])
def api_job_logs(job_id: str):
    job = _get_job(job_id)
    return jsonify({"log": job["log_lines"], "status": job["status"], "paused": job.get("paused", False)})


@app.route("/api/jobs/<job_id>/pause", methods=["POST"])
def api_job_pause(job_id: str):
    job = _get_job(job_id)

    # SIGSTOP/SIGCONT are not available on Windows
    if not hasattr(signal, "SIGSTOP"):
        return jsonify({"error": "Pause/resume is not supported on this platform."}), 501

    with jobs_lock:
        if job["status"] not in ("running", "paused"):
            return jsonify({"error": "Job is not running or paused."}), 409

        proc = job.get("process")
        if not proc:
            return jsonify({"error": "Subprocess not available."}), 409

        log_lines: list[str] = job["log_lines"]

        if not job.get("paused", False):
            # Pause the process group
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGSTOP)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            job["paused"] = True
            job["status"] = "paused"
            log_lines.append("[QuantumCrawler] Job paused by user")
            new_status = "paused"
        else:
            # Resume the process group
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGCONT)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            job["paused"] = False
            job["status"] = "running"
            log_lines.append("[QuantumCrawler] Job resumed by user")
            new_status = "running"

    return jsonify({"status": new_status, "paused": job["paused"]})


@app.route("/api/jobs/<job_id>/stop", methods=["POST"])
def api_job_stop(job_id: str):
    job = _get_job(job_id)

    with jobs_lock:
        if job["status"] not in ("running", "paused"):
            return jsonify({"error": "Job is not running or paused."}), 409

        proc = job.get("process")
        log_lines: list[str] = job["log_lines"]

        # Mark stopped before terminating so _run_crawler thread won't overwrite status
        job["status"] = "stopped"
        job["end_time"] = _utcnow_iso()
        log_lines.append("[QuantumCrawler] Job stopped by user")

    if proc:
        # If paused, resume first so it can receive SIGTERM
        if job.get("paused") and hasattr(signal, "SIGCONT"):
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGCONT)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        with jobs_lock:
            job["paused"] = False

        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception as exc:
            app.logger.warning("[QuantumCrawler] terminate() failed for job %s: %s", job_id, exc)
            try:
                proc.kill()
            except Exception as kill_exc:
                app.logger.warning("[QuantumCrawler] kill() failed for job %s: %s", job_id, kill_exc)

    return jsonify({"status": "stopped"})


@app.route("/api/jobs/<job_id>/results", methods=["GET"])
def api_job_results(job_id: str):
    job = _get_job(job_id)
    results = _parse_jsonl(job["out_jsonl"])
    if results:
        return jsonify([_normalise_explorer_row(r) for r in results])

    # Comparison mode doesn't emit crawl.jsonl; fall back to comparison.json.
    compare_json_path = job.get("compare_json")
    if compare_json_path and os.path.exists(compare_json_path):
        try:
            with open(compare_json_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            records = payload.get("records") or []
            if isinstance(records, list) and records:
                return jsonify([_normalise_explorer_row(r) for r in records if isinstance(r, dict)])
        except Exception:
            pass

    return jsonify([])


@app.route("/api/jobs/<job_id>/download/<filetype>", methods=["GET"])
def api_job_download(job_id: str, filetype: str):
    job = _get_job(job_id)

    file_map = {
        "jsonl": (job["out_jsonl"], "crawl.jsonl", "application/jsonlines"),
        "csv": (job["compare_csv"], "comparison.csv", "text/csv"),
        "json": (job["compare_json"], "comparison.json", "application/json"),
    }

    if filetype not in file_map:
        abort(400, description=f"Unknown filetype {filetype!r}. Use jsonl, csv, or json.")

    path, filename, mimetype = file_map[filetype]
    if not os.path.exists(path):
        # Comparison mode doesn't emit crawl.jsonl; generate it on-demand so
        # "Download JSONL" works for already-completed jobs too.
        if filetype == "jsonl":
            compare_json_path = job.get("compare_json")
            if compare_json_path and os.path.exists(compare_json_path):
                try:
                    with open(compare_json_path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                    records = payload.get("records") or []
                    if isinstance(records, list):
                        with open(path, "w", encoding="utf-8") as out_f:
                            for r in records:
                                if isinstance(r, dict):
                                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        app.logger.info(
                            "[QuantumCrawler] Generated missing crawl.jsonl for job=%s (%d rows)",
                            job_id,
                            len(records),
                        )
                except Exception:
                    # Fall through to the normal 404 behaviour.
                    pass

        if not os.path.exists(path):
            abort(404, description=f"Output file not yet available for filetype={filetype!r}.")

    return send_file(path, mimetype=mimetype, as_attachment=True, download_name=filename)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": str(e)}), 404
    return render_template("dashboard.html", jobs=[], error=str(e)), 404


@app.errorhandler(400)
def bad_request(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": str(e)}), 400
    return render_template("dashboard.html", jobs=[], error=str(e)), 400


if __name__ == "__main__":
    _debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(debug=_debug, host="0.0.0.0", port=5000)
