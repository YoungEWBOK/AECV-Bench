"""Generate an HTML report for QA cases judged incorrect.

The report is static and portable inside the output directory: referenced
images are copied into an assets folder next to the HTML file.
"""
import argparse
import csv
import html
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Map image stem to image path."""
    image_index = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[path.stem] = path
    return image_index


def read_wrong_cases(csv_path: Path) -> List[Dict[str, str]]:
    """Read rows with score/overall equal to zero."""
    wrong_cases = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = (row.get("overall") or row.get("score") or "").strip()
            try:
                is_wrong = float(score) == 0.0
            except ValueError:
                is_wrong = False
            if is_wrong:
                wrong_cases.append(row)
    return wrong_cases


def display_model_name(csv_path: Path) -> str:
    """Derive a readable model name from an evaluation CSV filename."""
    raw = csv_path.stem.replace("_evaluation_results", "").strip("_")
    known = {
        "openai_gpt_55": "OpenAI GPT-5.5",
        "api_gemini_31_pro": "Gemini 3.1 Pro",
        "gemini_31_pro": "Gemini 3.1 Pro",
    }
    return known.get(raw, raw.replace("_", " ").title())


def copy_case_image(
    image_id: str,
    image_index: Dict[str, Path],
    assets_dir: Path,
) -> Optional[str]:
    """Copy an image into the report assets folder and return a relative path."""
    source = image_index.get(image_id)
    if source is None:
        return None
    target = assets_dir / source.name
    if not target.exists():
        shutil.copy2(source, target)
    return f"assets/{target.name}"


def summarize_cases(cases: Iterable[Dict[str, str]]) -> Dict[str, Counter]:
    """Summarize cases by task and QA type."""
    task_counter = Counter()
    qa_type_counter = Counter()
    image_counter = Counter()
    for case in cases:
        task_counter[case.get("task", "unknown") or "unknown"] += 1
        qa_type_counter[case.get("qa_type", "unknown") or "unknown"] += 1
        image_counter[case.get("image_id", "unknown") or "unknown"] += 1
    return {
        "task": task_counter,
        "qa_type": qa_type_counter,
        "image": image_counter,
    }


def render_counter(counter: Counter) -> str:
    """Render a compact counter table."""
    if not counter:
        return "<p class=\"muted\">No cases.</p>"
    rows = []
    for key, value in counter.most_common():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(key))}</td>"
            f"<td>{value}</td>"
            "</tr>"
        )
    return "<table><tbody>" + "\n".join(rows) + "</tbody></table>"


def render_case_card(case: Dict[str, str], image_rel_path: Optional[str]) -> str:
    """Render one wrong-case card."""
    image_html = (
        f'<a href="{html.escape(image_rel_path)}" target="_blank">'
        f'<img src="{html.escape(image_rel_path)}" alt="{html.escape(case.get("image_id", ""))}">'
        "</a>"
        if image_rel_path
        else "<div class=\"missing-image\">Image not found</div>"
    )
    return f"""
<article class="case-card">
  <div class="case-image">{image_html}</div>
  <div class="case-body">
    <div class="meta">
      <span>{html.escape(case.get("image_id", ""))}</span>
      <span>{html.escape(case.get("qa_id", ""))}</span>
      <span>{html.escape(case.get("qa_type", ""))}</span>
      <span>{html.escape(case.get("task", ""))}</span>
    </div>
    <h3>{html.escape(case.get("question", ""))}</h3>
    <dl>
      <dt>Ground Truth</dt>
      <dd>{html.escape(case.get("ground_truth", ""))}</dd>
      <dt>Model Answer</dt>
      <dd>{html.escape(case.get("predicted", ""))}</dd>
    </dl>
  </div>
</article>
"""


def render_model_section(
    model_name: str,
    csv_path: Path,
    cases: List[Dict[str, str]],
    image_index: Dict[str, Path],
    assets_dir: Path,
) -> str:
    """Render one model's wrong-case section."""
    summaries = summarize_cases(cases)
    image_rel_paths = {
        case.get("image_id", ""): copy_case_image(case.get("image_id", ""), image_index, assets_dir)
        for case in cases
    }
    cards = [
        render_case_card(case, image_rel_paths.get(case.get("image_id", "")))
        for case in sorted(cases, key=lambda row: (row.get("image_id", ""), row.get("qa_id", "")))
    ]
    return f"""
<section class="model-section">
  <h2>{html.escape(model_name)}</h2>
  <p class="source">Source: {html.escape(str(csv_path))}</p>
  <div class="summary-grid">
    <div>
      <h3>Wrong Cases</h3>
      <p class="big-number">{len(cases)}</p>
    </div>
    <div>
      <h3>By QA Type</h3>
      {render_counter(summaries["qa_type"])}
    </div>
    <div>
      <h3>By Task</h3>
      {render_counter(summaries["task"])}
    </div>
    <div>
      <h3>By Image</h3>
      {render_counter(summaries["image"])}
    </div>
  </div>
  <div class="cases">
    {''.join(cards)}
  </div>
</section>
"""


def render_html(title: str, sections: List[str], total_cases: int) -> str:
    """Render the full HTML page."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #63717a;
      --line: #d9e0e4;
      --panel: #f7f9fa;
      --accent: #0f766e;
      --bad: #b42318;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.45;
    }}
    header {{
      padding: 32px 40px 22px;
      border-bottom: 1px solid var(--line);
      background: #f4f7f8;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 36px 0 6px;
      font-size: 24px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .muted, .source {{
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 0 28px 48px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: minmax(140px, 0.8fr) repeat(3, minmax(180px, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
      align-items: start;
    }}
    .summary-grid > div {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .big-number {{
      margin: 0;
      font-size: 36px;
      font-weight: 700;
      color: var(--bad);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    td {{
      border-top: 1px solid var(--line);
      padding: 6px 0;
      vertical-align: top;
    }}
    td:last-child {{
      text-align: right;
      font-weight: 700;
    }}
    .cases {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
      gap: 16px;
    }}
    .case-card {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-height: 210px;
    }}
    .case-image {{
      display: flex;
      align-items: flex-start;
      justify-content: center;
      background: #f1f4f5;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      min-height: 190px;
    }}
    .case-image img {{
      display: block;
      width: 100%;
      height: 210px;
      object-fit: contain;
      background: white;
    }}
    .missing-image {{
      padding: 24px;
      color: var(--muted);
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }}
    .meta span {{
      display: inline-block;
      padding: 3px 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--accent);
      font-size: 12px;
      white-space: nowrap;
    }}
    .case-body h3 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    dl {{
      margin: 0;
      font-size: 14px;
    }}
    dt {{
      margin-top: 10px;
      font-weight: 700;
    }}
    dd {{
      margin: 3px 0 0;
      color: #26343b;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 800px) {{
      header {{
        padding: 24px 20px 18px;
      }}
      main {{
        padding: 0 16px 36px;
      }}
      .summary-grid, .cases, .case-card {{
        grid-template-columns: 1fr;
      }}
      .case-image img {{
        height: 240px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="muted">Incorrect QA cases judged with overall=0.0. Total wrong cases across selected files: {total_cases}.</p>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA wrong-case HTML report.")
    parser.add_argument("--csv-files", nargs="+", required=True, help="Evaluation CSV files.")
    parser.add_argument(
        "--image-dir",
        default="data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images",
        help="Directory containing QA benchmark images.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/qa_error_reports",
        help="Directory for the HTML report and copied assets.",
    )
    parser.add_argument(
        "--title",
        default="QA Wrong Case Report",
        help="HTML report title.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(image_dir)
    sections = []
    total_cases = 0
    all_cases_by_image = defaultdict(list)

    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        cases = read_wrong_cases(csv_path)
        total_cases += len(cases)
        for case in cases:
            all_cases_by_image[case.get("image_id", "")].append((display_model_name(csv_path), case))
        sections.append(
            render_model_section(
                display_model_name(csv_path),
                csv_path,
                cases,
                image_index,
                assets_dir,
            )
        )

    html_text = render_html(args.title, sections, total_cases)
    report_path = output_dir / "qa_wrong_cases_report.html"
    report_path.write_text(html_text, encoding="utf-8")

    print(f"Report saved to: {report_path.resolve()}")
    print(f"Assets copied to: {assets_dir.resolve()}")
    print(f"Total wrong cases: {total_cases}")
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        print(f"  - {display_model_name(csv_path)}: {len(read_wrong_cases(csv_path))}")


if __name__ == "__main__":
    main()
