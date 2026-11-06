from math import isfinite


def clean_intervals(actions, highlights, duration):
    def bounded(rows):
        for row in rows:
            start, end = float(row["start"]), float(row["end"])
            if not (isfinite(start) and isfinite(end)):
                continue
            start, end = max(0.0, start), min(float(duration), end)
            if start < end:
                yield dict(row, start=start, end=end)

    clean_highlights = []
    previous_end = 0.0
    for row in sorted(bounded(highlights), key=lambda row: row["start"]):
        row["start"] = max(previous_end, row["start"])
        if row["start"] >= row["end"]:
            continue
        importance = float(row["importance"])
        row["importance"] = min(1.0, max(0.0, importance)) if isfinite(importance) else 0.0
        clean_highlights.append(row)
        previous_end = row["end"]
    return list(bounded(actions)), clean_highlights
