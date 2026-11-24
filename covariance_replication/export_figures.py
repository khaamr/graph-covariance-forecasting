"""
Export saved covariance figures into a portable Plotly gallery.
"""

from hashlib import sha256
from html import escape
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ("replication", "Covariance_Replication.ipynb"),
    ("shape extension", "Extended_Covariance.ipynb"),
    ("spectral extension", "Spectral_Covariance.ipynb"),
)


def export_figures():
    """
    Preserve the stored figure payloads without downloading data or fitting models.
    """
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    records, cards = [], []
    for group, filename in NOTEBOOKS:
        notebook_path = ROOT / "covariance_replication" / filename
        notebook = json.loads(notebook_path.read_text())
        notebook_hash = sha256(notebook_path.read_bytes()).hexdigest()
        number = 0
        for cell_index, cell in enumerate(notebook["cells"]):
            for output_index, output in enumerate(cell.get("outputs", [])):
                payload = output.get("data", {}).get("application/vnd.plotly.v1+json")
                if payload is None:
                    continue
                number += 1
                identifier = f"{group.replace(' ', '-')}-{number:02d}"
                title = payload.get("layout", {}).get("title", {}).get("text", identifier)
                display_title = title.replace(" " + chr(183) + " ", ", ")
                target = directory / f"{identifier}.json"
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                figure = go.Figure(data=payload["data"], layout=payload.get("layout", {}))
                figure.update_layout(title_text=display_title)
                chart = pio.to_html(
                    figure, full_html=False, include_plotlyjs=not cards,
                    config={"responsive": True}, default_width="100%",
                    default_height="520px", div_id=identifier, auto_play=False,
                )
                source = notebook_path.relative_to(ROOT).as_posix()
                records.append({
                    "id": identifier, "group": group, "title": title,
                    "file": target.relative_to(ROOT).as_posix(), "notebook": source,
                    "notebook_sha256": notebook_hash,
                    "cell_index": cell_index, "output_index": output_index,
                })
                cards.append(
                    f'<section id="section-{identifier}" class="card">'
                    f'<div class="eyebrow">{escape(group)}</div>'
                    f'<h2>{escape(display_title.lower())}</h2>'
                    f'<p class="source"><a href="https://github.com/khaamr/graph-covariance-forecasting/blob/main/{source}">source notebook</a>'
                    f', <a href="figures/{identifier}.json">figure json</a></p>'
                    f'{chart}</section>'
                )
    if len(records) != 9:
        raise ValueError(f"Expected nine saved covariance figures, found {len(records)}.")
    (directory / "index.json").write_text(json.dumps(records, indent=2) + "\n")
    links = "".join(
        f'<a href="#section-{record["id"]}">{index:02d}, {escape(record["group"])}</a>'
        for index, record in enumerate(records, 1)
    )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>graph covariance forecasting, research gallery</title>
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#eef3fb;color:#112345;font:16px system-ui,sans-serif}
header{background:linear-gradient(125deg,#050b24,#113b88);color:#f4f8ff;padding:64px max(24px,calc((100vw - 1180px)/2))}
header p{max-width:900px;line-height:1.65;color:#cedcf4}
h1{font-size:clamp(32px,4vw,54px);letter-spacing:-.035em;margin:12px 0}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:lowercase;color:#5b86ca}
header .eyebrow{color:#a7caff}nav{display:flex;flex-wrap:wrap;gap:9px;margin-top:24px}
nav a{color:#e8f0ff;border:1px solid #628ac0;border-radius:999px;padding:8px 12px;text-decoration:none;font-size:13px}
main{max-width:1228px;margin:32px auto;padding:0 24px}
.card{background:white;border:1px solid #d8e2f0;border-radius:18px;padding:24px;margin:24px 0;box-shadow:0 10px 36px #11234508;scroll-margin-top:20px}
h2{font-size:23px;letter-spacing:-.02em;margin:8px 0}.source{font-size:13px;color:#5b6d88}
a{color:#2458a8}.notice{border-left:4px solid #5d8bd0;padding-left:16px;line-height:1.65}
footer{max-width:1180px;margin:32px auto;padding:0 24px 40px;color:#5b6d88;font-size:14px}
</style></head><body>
<header><div class="eyebrow">saved research, interactive figures</div>
<h1>graph covariance forecasting</h1>
<p>nine figures from the executed covariance notebooks. explore the replication,
adaptive covariance shapes and geometric spectral extension without downloading
market data or rerunning the models.</p><nav><a href="index.html#learn">animated graph explanation ↗</a>""" + links + """</nav></header>
<main><p class="notice">these are stored experimental outputs. the newer extension
uses a reused 2025–2026 benchmark. its risk interval includes zero; a definite joint
accuracy and portfolio-risk improvement is not established.</p>""" + "".join(cards) + """
</main><footer>opens offline with its bundled Plotly library. source notebooks,
figure payloads and result tables remain in the project folder.
<a href="https://github.com/khaamr/graph-covariance-forecasting/blob/main/README.md">project overview</a></footer></body></html>
"""
    (ROOT / "research_dashboard.html").write_text(page)
    print(f"Exported {len(records)} figures into {ROOT / 'research_dashboard.html'}")


if __name__ == "__main__":
    export_figures()
