import os
from html import escape


def build_demo(target, source, highlight, result, duration):
    def url(path):
        try:
            from urllib.parse import quote

            return quote(os.path.relpath(path, target.parent).replace(os.sep, "/"))
        except ValueError:
            return path.as_uri()

    rows = []
    for action in result["actions"]:
        start = action["start"]
        label = escape(f"{action['verb']} {action['noun']}")
        rows.append(
            f'<li><button data-time="{start}">{start:.2f}s</button> '
            f'{label} <small>to {action["end"]:.2f}s</small></li>'
        )
    clips = result["highlights"]
    blocks = "".join(
        f'<button class="segment" data-time="{c["start"]}" '
        f'title="{c["start"]:.2f}s to {c["end"]:.2f}s" '
        f'aria-label="Jump to {c["start"]:.2f} seconds" '
        f'style="left:{100*c["start"]/duration:.5f}%;'
        f'width:{100*(c["end"]-c["start"])/duration:.5f}%"></button>'
        for c in clips
    )
    selected = sum(c["end"] - c["start"] for c in clips)
    target.write_text(f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Track 4 · Action highlights</title>
<style>
* {{box-sizing:border-box}} body {{font:16px system-ui;background:#111a29;color:#edf4ff;
max-width:1100px;margin:40px auto;padding:0 24px}}
small,p {{color:#a8bad0}} .videos {{display:grid;grid-template-columns:repeat(auto-fit,
minmax(min(100%,320px),1fr));gap:24px}} video {{width:100%;background:#080e18;
aspect-ratio:16/9;border-radius:10px}} button {{cursor:pointer;background:#81e4ce;
color:#122a26;border:0;border-radius:4px;padding:4px 8px}} li {{margin:8px 0}}
.track {{position:relative;height:28px;background:#283952;border-radius:4px;overflow:hidden}}
.segment {{position:absolute;height:100%;padding:0;border-radius:0;border-right:1px solid #283952}}
</style>
<h1>Action highlights</h1>
<p>{escape(result["video_id"])} · Predicted category: {escape(result["label"])} ·
{selected:.2f}s selected from {duration:.2f}s ({100*selected/duration:.1f}%)</p>
<div class="videos"><section><h2>Source</h2>
<video id="source" src="{escape(url(source))}" controls preload="metadata"></video></section>
<section><h2>Highlight reel</h2>
<video src="{escape(url(highlight))}" controls preload="metadata"></video></section></div>
<h2>Selected moments</h2><div class="track">{blocks}</div>
<h2>Predicted actions</h2><ul>{''.join(rows) or '<li>No confident action labels</li>'}</ul>
<script>
document.addEventListener('click', e => {{
  const button = e.target.closest('button[data-time]');
  if (button) document.getElementById('source').currentTime = Number(button.dataset.time);
}});
</script></html>""", encoding="utf-8")
