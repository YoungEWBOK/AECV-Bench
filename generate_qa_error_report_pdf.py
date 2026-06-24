"""Generate a PDF report for QA cases judged incorrect."""
import argparse
import csv
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


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


def wrap_text(value: str, width: int, max_lines: Optional[int] = None) -> str:
    """Wrap text for PDF rendering and optionally truncate it."""
    text = (value or "").replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=True)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip(". ") + " ..."
    return "\n".join(lines)


def draw_text_block(
    fig,
    x: float,
    y: float,
    title: str,
    body: str,
    width: int = 72,
    max_lines: Optional[int] = None,
    body_size: int = 9,
) -> float:
    """Draw a labeled wrapped text block and return the next y position."""
    fig.text(x, y, title, fontsize=10, fontweight="bold", color="#172026", va="top")
    wrapped = wrap_text(body, width=width, max_lines=max_lines)
    line_count = max(1, wrapped.count("\n") + 1)
    y_body = y - 0.026
    fig.text(
        x,
        y_body,
        wrapped,
        fontsize=body_size,
        color="#26343b",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fa", "edgecolor": "#d9e0e4"},
    )
    return y_body - line_count * 0.025 - 0.03


def add_title_page(pdf: PdfPages, title: str, model_summaries: List[Dict]) -> None:
    """Add the report title and summary page."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.88, title, fontsize=24, fontweight="bold", color="#172026")
    fig.text(
        0.06,
        0.82,
        "Incorrect QA cases where the judge score is overall=0.0.",
        fontsize=12,
        color="#63717a",
    )
    total = sum(item["wrong_count"] for item in model_summaries)
    fig.text(0.06, 0.73, f"Total wrong cases: {total}", fontsize=16, fontweight="bold", color="#b42318")

    y = 0.64
    for item in model_summaries:
        fig.text(0.06, y, item["model_name"], fontsize=14, fontweight="bold", color="#0f766e")
        fig.text(0.34, y, f"{item['wrong_count']} wrong cases", fontsize=14, color="#172026")
        y -= 0.06

    fig.text(
        0.06,
        0.12,
        "Each following page shows one case: source drawing, question, ground truth, and model answer.",
        fontsize=10,
        color="#63717a",
    )
    pdf.savefig(fig)
    plt.close(fig)


def add_model_summary_page(pdf: PdfPages, model_name: str, cases: List[Dict[str, str]]) -> None:
    """Add a compact summary page for one model."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.88, model_name, fontsize=22, fontweight="bold", color="#172026")
    fig.text(0.06, 0.82, f"Wrong cases: {len(cases)}", fontsize=16, fontweight="bold", color="#b42318")

    qa_counts = Counter(case.get("qa_type", "unknown") or "unknown" for case in cases)
    task_counts = Counter(case.get("task", "unknown") or "unknown" for case in cases)
    image_counts = Counter(case.get("image_id", "unknown") or "unknown" for case in cases)

    def draw_counter(x: float, y: float, title: str, counter: Counter, max_items: int = 14) -> None:
        fig.text(x, y, title, fontsize=13, fontweight="bold", color="#172026")
        y -= 0.045
        for key, value in counter.most_common(max_items):
            fig.text(x, y, str(key), fontsize=9, color="#26343b")
            fig.text(x + 0.25, y, str(value), fontsize=9, fontweight="bold", color="#26343b")
            y -= 0.033

    draw_counter(0.06, 0.72, "By QA Type", qa_counts)
    draw_counter(0.38, 0.72, "By Task", task_counts)
    draw_counter(0.70, 0.72, "By Image", image_counts)
    pdf.savefig(fig)
    plt.close(fig)


def add_case_page(
    pdf: PdfPages,
    model_name: str,
    case: Dict[str, str],
    image_path: Optional[Path],
    case_index: int,
    case_total: int,
) -> None:
    """Add a single wrong case page."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(
        0.05,
        0.94,
        f"{model_name} - wrong case {case_index}/{case_total}",
        fontsize=15,
        fontweight="bold",
        color="#172026",
    )
    meta = " | ".join(
        value
        for value in [
            case.get("image_id", ""),
            case.get("qa_id", ""),
            case.get("qa_type", ""),
            case.get("task", ""),
        ]
        if value
    )
    fig.text(0.05, 0.905, meta, fontsize=9, color="#0f766e")

    ax = fig.add_axes([0.05, 0.14, 0.45, 0.72])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f7f9fa")
    if image_path and image_path.exists():
        image = Image.open(image_path)
        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=9, color="#63717a", pad=8)
    else:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=12, color="#63717a")
    for spine in ax.spines.values():
        spine.set_color("#d9e0e4")

    x = 0.54
    y = 0.84
    y = draw_text_block(fig, x, y, "Question", case.get("question", ""), width=76, max_lines=5, body_size=10)
    y = draw_text_block(fig, x, y, "Ground Truth", case.get("ground_truth", ""), width=76, max_lines=5, body_size=10)
    draw_text_block(fig, x, y, "Model Answer", case.get("predicted", ""), width=76, max_lines=18, body_size=9)

    fig.text(0.05, 0.06, "QA wrong-case report", fontsize=8, color="#63717a")
    fig.text(0.88, 0.06, f"{case_index}/{case_total}", fontsize=8, color="#63717a")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report for QA wrong cases.")
    parser.add_argument("--csv-files", nargs="+", required=True, help="Evaluation CSV files.")
    parser.add_argument(
        "--image-dir",
        default="data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images",
        help="Directory containing QA benchmark images.",
    )
    parser.add_argument(
        "--output-pdf",
        default="results/qa_error_reports/qa_wrong_cases_report.pdf",
        help="Path to write the PDF report.",
    )
    parser.add_argument(
        "--title",
        default="QA Wrong Case Report",
        help="PDF report title.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_index = build_image_index(image_dir)
    output_pdf = Path(args.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    model_data = []
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        cases = read_wrong_cases(csv_path)
        model_data.append(
            {
                "csv_path": csv_path,
                "model_name": display_model_name(csv_path),
                "cases": sorted(cases, key=lambda row: (row.get("image_id", ""), row.get("qa_id", ""))),
                "wrong_count": len(cases),
            }
        )

    with PdfPages(output_pdf) as pdf:
        add_title_page(pdf, args.title, model_data)
        for item in model_data:
            add_model_summary_page(pdf, item["model_name"], item["cases"])
            total = len(item["cases"])
            for idx, case in enumerate(item["cases"], 1):
                image_path = image_index.get(case.get("image_id", ""))
                add_case_page(pdf, item["model_name"], case, image_path, idx, total)

    print(f"PDF saved to: {output_pdf.resolve()}")
    print(f"Pages: {1 + len(model_data) + sum(item['wrong_count'] for item in model_data)}")
    for item in model_data:
        print(f"  - {item['model_name']}: {item['wrong_count']}")


if __name__ == "__main__":
    main()
