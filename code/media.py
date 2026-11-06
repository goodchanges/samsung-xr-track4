import os
import re
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe

SIZE = 224


def ffmpeg(*args):
    return [get_ffmpeg_exe(), "-hide_banner", "-nostdin", *map(str, args)]


def probe(path):
    result = subprocess.run(ffmpeg("-i", path), capture_output=True)
    info = result.stderr.decode("utf-8", "replace")
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if not match or "Video:" not in info:
        raise ValueError(f"Cannot read a timed video stream in {path.name}.")
    hours, minutes, seconds = map(float, match.groups())
    duration = 3600 * hours + 60 * minutes + seconds
    if duration <= 0:
        raise ValueError("The video has zero duration.")
    return duration, "Audio:" in info


def frames(path, step):
    filters = (
        f"fps=1/{step}:start_time=0:round=up,"
        f"scale={SIZE}:{SIZE}:force_original_aspect_ratio=increase,crop={SIZE}:{SIZE}"
    )
    result = subprocess.run(
        ffmpeg("-loglevel", "error", "-i", path, "-map", "0:v:0",
               "-vf", filters, "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"),
        capture_output=True,
    )
    if result.returncode or not result.stdout:
        raise ValueError(f"Frame decoding failed: {result.stderr.decode(errors='replace')}")
    return result.stdout


def render(source, target, highlights, audio):
    if not highlights:
        raise ValueError("No highlight intervals to render.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.mp4")
    parts = []
    labels = []
    for i, clip in enumerate(highlights):
        bounds = f"start={clip['start']}:end={clip['end']}"
        parts.append(f"[0:v:0]trim={bounds},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
        if audio:
            parts.append(f"[0:a:0]atrim={bounds},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
    parts.append(
        "".join(labels) + f"concat=n={len(highlights)}:v=1:a={int(audio)}[joined]"
        + ("[audio]" if audio else "")
    )
    parts.append("[joined]pad=ceil(iw/2)*2:ceil(ih/2)*2[vout]")
    maps = ["-map", "[vout]"]
    if audio:
        maps += ["-map", "[audio]", "-c:a", "aac"]
    try:
        result = subprocess.run(
            ffmpeg("-loglevel", "error", "-y", "-i", source,
                   "-filter_complex", ";".join(parts), *maps,
                   "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", temporary),
            capture_output=True,
        )
        if result.returncode:
            raise ValueError(f"Highlight render failed: {result.stderr.decode(errors='replace')}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
