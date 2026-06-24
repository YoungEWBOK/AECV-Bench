"""Generate HTML and PDF analysis for prompt-strategy benchmark results."""
import argparse
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


STRATEGIES = ("one_shot", "step_by_step", "self_refine", "two_pass_reflection", "skill_guided")
STRATEGY_SUFFIXES = ("step_by_step", "self_refine", "two_pass_reflection", "skill_guided")
COUNT_FIELDS = ("Door", "Window", "Space", "Bedroom", "Toilet")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

MODEL_NAMES = {
    "api_gemini_31_pro": "Gemini 3.1 Pro",
    "gemini_31_pro": "Gemini 3.1 Pro",
    "openai_gpt_55": "OpenAI GPT-5.5",
    "openai_gpt_54": "OpenAI GPT-5.4",
    "openai_gpt_53": "OpenAI GPT-5.3",
    "openai_gpt_52": "OpenAI GPT-5.2",
    "qwen_35_plus": "Qwen 3.5 Plus",
    "qwen37_plus": "Qwen3.7 Plus",
    "qwen_37_plus": "Qwen3.7 Plus",
    "glm_46v": "GLM-4.6V",
}


def split_strategy(raw_name: str) -> Tuple[str, str]:
    """Split raw filename model id into base id and prompt strategy."""
    raw_name = raw_name.strip("_").lower()
    for suffix in sorted(STRATEGY_SUFFIXES, key=len, reverse=True):
        token = f"_{suffix}"
        if raw_name.endswith(token):
            return raw_name[:-len(token)], suffix
    return raw_name, "one_shot"


def display_name(base: str, strategy: str) -> str:
    """Readable display name."""
    model = MODEL_NAMES.get(base, base.replace("_", " ").title())
    return model if strategy == "one_shot" else f"{model} ({strategy})"


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into dict rows."""
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    """Map image_id/stem to image path."""
    if not image_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def copy_image_asset(image_id: str, image_index: Dict[str, Path], assets_dir: Path) -> Optional[str]:
    """Copy a case image into report assets and return relative asset path."""
    source = image_index.get(image_id)
    if source is None:
        return None
    target = assets_dir / source.name
    if not target.exists():
        shutil.copy2(source, target)
    return f"assets/{target.name}"


def safe_float(value, default=0.0) -> float:
    """Parse float with a default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_qa_evaluations(eval_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """Load QA judge result CSVs grouped by base model and strategy."""
    evaluations = {}
    for path in sorted(eval_dir.glob("*_evaluation_results.csv")):
        if path.name in {"complete_evaluation_results.csv", "detailed_evaluation_results.csv"}:
            continue
        raw = path.stem.replace("_evaluation_results", "").strip("_")
        base, strategy = split_strategy(raw)
        rows = read_csv(path)
        if not rows:
            continue
        scores = [safe_float(row.get("overall")) for row in rows if row.get("overall") not in (None, "")]
        if not scores:
            continue

        qa_type_scores = defaultdict(list)
        task_scores = defaultdict(list)
        for row in rows:
            score = safe_float(row.get("overall"))
            qa_type_scores[row.get("qa_type", "unknown") or "unknown"].append(score)
            task_scores[row.get("task", "unknown") or "unknown"].append(score)

        evaluations[(base, strategy)] = {
            "base": base,
            "strategy": strategy,
            "raw": raw,
            "path": path,
            "rows": rows,
            "n": len(scores),
            "mean": sum(scores) / len(scores),
            "correct": sum(1 for score in scores if score >= 0.5),
            "wrong": sum(1 for score in scores if score < 0.5),
            "qa_type": {
                key: sum(vals) / len(vals)
                for key, vals in sorted(qa_type_scores.items())
                if vals
            },
            "task": {
                key: sum(vals) / len(vals)
                for key, vals in sorted(task_scores.items())
                if vals
            },
        }
    return evaluations


def result_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    """Stable QA row key."""
    return (row.get("image_id", ""), row.get("qa_id", ""), row.get("qa_type", ""))


def compare_to_baseline(baseline: Dict, variant: Dict) -> Dict:
    """Compare a strategy variant against its one-shot baseline."""
    base_scores = {result_key(row): safe_float(row.get("overall")) for row in baseline["rows"]}
    var_scores = {result_key(row): safe_float(row.get("overall")) for row in variant["rows"]}
    shared = sorted(set(base_scores) & set(var_scores))
    fixed = [key for key in shared if base_scores[key] < 0.5 and var_scores[key] >= 0.5]
    regressed = [key for key in shared if base_scores[key] >= 0.5 and var_scores[key] < 0.5]
    both_wrong = [key for key in shared if base_scores[key] < 0.5 and var_scores[key] < 0.5]
    both_right = [key for key in shared if base_scores[key] >= 0.5 and var_scores[key] >= 0.5]
    return {
        "shared": len(shared),
        "fixed": len(fixed),
        "regressed": len(regressed),
        "both_wrong": len(both_wrong),
        "both_right": len(both_right),
        "delta": variant["mean"] - baseline["mean"],
        "fixed_keys": fixed,
        "regressed_keys": regressed,
        "both_wrong_keys": both_wrong,
        "both_right_keys": both_right,
    }


def qa_row_map(evaluation: Dict) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Map QA rows by stable key."""
    return {result_key(row): row for row in evaluation["rows"]}


def build_improvement_records(evaluations: Dict[Tuple[str, str], Dict]) -> List[Dict]:
    """Build per-strategy improvement/regression records against one-shot."""
    grouped = defaultdict(dict)
    for (base, strategy), data in evaluations.items():
        grouped[base][strategy] = data

    records = []
    for base, strategies in sorted(grouped.items()):
        baseline = strategies.get("one_shot")
        if not baseline:
            continue
        base_rows = qa_row_map(baseline)
        for strategy in STRATEGIES:
            if strategy == "one_shot" or strategy not in strategies:
                continue
            variant = strategies[strategy]
            variant_rows = qa_row_map(variant)
            comparison = compare_to_baseline(baseline, variant)

            fixed_counter = Counter()
            regressed_counter = Counter()
            for key in comparison["fixed_keys"]:
                fixed_counter[variant_rows[key].get("qa_type", "unknown") or "unknown"] += 1
            for key in comparison["regressed_keys"]:
                regressed_counter[variant_rows[key].get("qa_type", "unknown") or "unknown"] += 1

            records.append(
                {
                    "base": base,
                    "strategy": strategy,
                    "baseline": baseline,
                    "variant": variant,
                    "base_rows": base_rows,
                    "variant_rows": variant_rows,
                    "comparison": comparison,
                    "fixed_by_type": fixed_counter,
                    "regressed_by_type": regressed_counter,
                }
            )
    return records


def parse_json_cell(value: str) -> Optional[Dict[str, float]]:
    """Parse object-counting JSON cell."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    result = {}
    for field in COUNT_FIELDS:
        if field not in parsed:
            return None
        result[field] = safe_float(parsed[field], math.nan)
    return result


def load_object_results(object_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """Load object-counting CSVs grouped by base model and strategy."""
    results = {}
    for path in sorted(object_dir.glob("*.csv")):
        raw = path.stem.strip("_").lower()
        base, strategy = split_strategy(raw)
        rows = read_csv(path)
        valid = []
        for row in rows:
            original = parse_json_cell(row.get("original", ""))
            extracted = parse_json_cell(row.get("extracted", ""))
            if original is None or extracted is None:
                continue
            valid.append((row.get("name", ""), original, extracted))
        if not valid:
            continue

        exact_rows = 0
        field_abs_errors = {field: [] for field in COUNT_FIELDS}
        field_exact = {field: 0 for field in COUNT_FIELDS}
        for _, original, extracted in valid:
            row_exact = True
            for field in COUNT_FIELDS:
                diff = abs(extracted[field] - original[field])
                field_abs_errors[field].append(diff)
                if diff == 0:
                    field_exact[field] += 1
                else:
                    row_exact = False
            if row_exact:
                exact_rows += 1

        results[(base, strategy)] = {
            "base": base,
            "strategy": strategy,
            "path": path,
            "n": len(valid),
            "row_exact": exact_rows / len(valid),
            "mean_mae": sum(
                sum(vals) / len(vals)
                for vals in field_abs_errors.values()
                if vals
            ) / len(COUNT_FIELDS),
            "field_mae": {
                field: sum(vals) / len(vals)
                for field, vals in field_abs_errors.items()
                if vals
            },
            "field_accuracy": {
                field: field_exact[field] / len(valid)
                for field in COUNT_FIELDS
            },
        }
    return results


def ensure_font() -> None:
    """Use a Chinese-capable font when available."""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    for font in candidates:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            return
        except Exception:
            continue


def plot_qa_overall(evaluations: Dict[Tuple[str, str], Dict], chart_dir: Path) -> Optional[Path]:
    """Plot QA overall scores for bases that have strategy variants."""
    grouped = defaultdict(dict)
    for (base, strategy), data in evaluations.items():
        grouped[base][strategy] = data
    rows = []
    for base, strategies in grouped.items():
        if any(strategy != "one_shot" for strategy in strategies):
            for strategy in STRATEGIES:
                if strategy in strategies:
                    rows.append((display_name(base, strategy), strategies[strategy]["mean"], strategy))
    if not rows:
        return None

    chart_path = chart_dir / "qa_overall_scores.png"
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = ["#5b8def" if row[2] == "one_shot" else "#18a999" for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("QA Prompt Strategy Overall Accuracy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def plot_qa_type_breakdown(evaluations: Dict[Tuple[str, str], Dict], chart_dir: Path) -> Optional[Path]:
    """Plot QA type breakdown for strategy variants."""
    grouped = defaultdict(dict)
    for (base, strategy), data in evaluations.items():
        grouped[base][strategy] = data
    selected_base = None
    for base, strategies in grouped.items():
        if any(strategy != "one_shot" for strategy in strategies):
            selected_base = base
            break
    if selected_base is None:
        return None

    strategies = [strategy for strategy in STRATEGIES if strategy in grouped[selected_base]]
    qa_types = sorted(set().union(*(grouped[selected_base][s]["qa_type"].keys() for s in strategies)))
    if not qa_types:
        return None

    chart_path = chart_dir / "qa_type_breakdown_strategy.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(strategies))
    x = list(range(len(qa_types)))
    palette = {
        "one_shot": "#5b8def",
        "step_by_step": "#18a999",
        "self_refine": "#f2a65a",
        "two_pass_reflection": "#9b5de5",
        "skill_guided": "#3a86ff",
    }
    for i, strategy in enumerate(strategies):
        vals = [grouped[selected_base][strategy]["qa_type"].get(qt, 0.0) for qt in qa_types]
        offset = (i - (len(strategies) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=strategy, color=palette.get(strategy, "#777"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"QA Type Breakdown - {display_name(selected_base, 'one_shot').replace(' (one_shot)', '')}")
    ax.set_xticks(x)
    ax.set_xticklabels(qa_types, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def plot_fixed_regressed(records: List[Dict], chart_dir: Path) -> Optional[Path]:
    """Plot fixed and regressed counts for each strategy variant."""
    if not records:
        return None
    chart_path = chart_dir / "qa_fixed_regressed_counts.png"
    labels = [display_name(record["base"], record["strategy"]) for record in records]
    fixed = [record["comparison"]["fixed"] for record in records]
    regressed = [record["comparison"]["regressed"] for record in records]

    x = list(range(len(records)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars_fixed = ax.bar([xi - width / 2 for xi in x], fixed, width=width, color="#18a999", label="Fixed")
    bars_regressed = ax.bar([xi + width / 2 for xi in x], regressed, width=width, color="#d95d39", label="Regressed")
    ax.set_title("QA Cases Fixed vs Regressed Compared with one_shot")
    ax.set_ylabel("Number of shared QA cases")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (bars_fixed, bars_regressed):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{int(height)}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def plot_object_metrics(object_results: Dict[Tuple[str, str], Dict], chart_dir: Path) -> Optional[Path]:
    """Plot object-counting metrics when strategy variants exist."""
    grouped = defaultdict(dict)
    for (base, strategy), data in object_results.items():
        grouped[base][strategy] = data
    selected = []
    for base, strategies in grouped.items():
        if any(strategy != "one_shot" for strategy in strategies):
            for strategy in STRATEGIES:
                if strategy in strategies:
                    selected.append((base, strategy, strategies[strategy]))
    if not selected:
        return None

    chart_path = chart_dir / "object_counting_strategy_metrics.png"
    labels = [display_name(base, strategy) for base, strategy, _ in selected]
    mae_vals = [data["mean_mae"] for _, _, data in selected]
    exact_vals = [data["row_exact"] for _, _, data in selected]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    x = range(len(selected))
    ax1.bar(x, exact_vals, color="#18a999", label="Exact row accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Exact Row Accuracy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(list(x), mae_vals, marker="o", color="#b42318", label="Mean MAE")
    ax2.set_ylabel("Mean MAE")
    ax1.set_title("Object Counting Strategy Metrics")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def qa_summary_table(evaluations: Dict[Tuple[str, str], Dict]) -> str:
    """Render QA summary HTML table."""
    grouped = defaultdict(dict)
    for (base, strategy), data in evaluations.items():
        grouped[base][strategy] = data
    rows = []
    for base in sorted(grouped):
        if not any(strategy != "one_shot" for strategy in grouped[base]):
            continue
        baseline = grouped[base].get("one_shot")
        for strategy in STRATEGIES:
            data = grouped[base].get(strategy)
            if not data:
                continue
            delta = ""
            fixed = regressed = shared = ""
            if baseline and strategy != "one_shot":
                cmp = compare_to_baseline(baseline, data)
                delta = f"{cmp['delta']:+.4f}"
                fixed = str(cmp["fixed"])
                regressed = str(cmp["regressed"])
                shared = str(cmp["shared"])
            rows.append(
                "<tr>"
                f"<td>{html.escape(display_name(base, strategy))}</td>"
                f"<td>{data['n']}</td>"
                f"<td>{data['mean']:.4f}</td>"
                f"<td>{delta}</td>"
                f"<td>{fixed}</td>"
                f"<td>{regressed}</td>"
                f"<td>{shared}</td>"
                f"<td>{html.escape(str(data['path']))}</td>"
                "</tr>"
            )
    if not rows:
        return "<p class=\"muted\">未检测到 QA prompt strategy 变体结果。</p>"
    return (
        "<table><thead><tr>"
        "<th>Model / Strategy</th><th>N</th><th>Accuracy</th><th>Δ vs one_shot</th>"
        "<th>Fixed</th><th>Regressed</th><th>Shared</th><th>Source CSV</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def qa_type_table(evaluations: Dict[Tuple[str, str], Dict]) -> str:
    """Render QA type breakdown table."""
    grouped = defaultdict(dict)
    for (base, strategy), data in evaluations.items():
        grouped[base][strategy] = data
    rows = []
    for base in sorted(grouped):
        if not any(strategy != "one_shot" for strategy in grouped[base]):
            continue
        for strategy in STRATEGIES:
            data = grouped[base].get(strategy)
            if not data:
                continue
            cells = "".join(
                f"<td>{score:.4f}</td>"
                for _, score in sorted(data["qa_type"].items())
            )
            headers = "".join(f"<th>{html.escape(qt)}</th>" for qt in sorted(data["qa_type"]))
            if not rows:
                rows.append("<table><thead><tr><th>Model / Strategy</th>" + headers + "</tr></thead><tbody>")
            rows.append(f"<tr><td>{html.escape(display_name(base, strategy))}</td>{cells}</tr>")
    if not rows:
        return "<p class=\"muted\">暂无 QA type breakdown。</p>"
    rows.append("</tbody></table>")
    return "\n".join(rows)


def object_summary_table(object_results: Dict[Tuple[str, str], Dict]) -> str:
    """Render object-counting summary HTML table."""
    grouped = defaultdict(dict)
    for (base, strategy), data in object_results.items():
        grouped[base][strategy] = data
    rows = []
    for base in sorted(grouped):
        has_variant = any(strategy != "one_shot" for strategy in grouped[base])
        if not has_variant:
            continue
        baseline = grouped[base].get("one_shot")
        for strategy in STRATEGIES:
            data = grouped[base].get(strategy)
            if not data:
                continue
            delta_mae = ""
            if baseline and strategy != "one_shot":
                delta_mae = f"{data['mean_mae'] - baseline['mean_mae']:+.4f}"
            rows.append(
                "<tr>"
                f"<td>{html.escape(display_name(base, strategy))}</td>"
                f"<td>{data['n']}</td>"
                f"<td>{data['row_exact']:.4f}</td>"
                f"<td>{data['mean_mae']:.4f}</td>"
                f"<td>{delta_mae}</td>"
                f"<td>{html.escape(str(data['path']))}</td>"
                "</tr>"
            )
    if not rows:
        available = [
            f"{display_name(base, strategy)} ({data['n']} rows)"
            for (base, strategy), data in sorted(object_results.items())
        ]
        note = "<br>".join(html.escape(item) for item in available[:20])
        return (
            "<p class=\"muted\">未检测到带 prompt strategy 后缀的 object counting 结果；"
            "当前只汇总已有普通 object counting CSV。</p>"
            + (f"<p class=\"muted\">可用文件：<br>{note}</p>" if available else "")
        )
    return (
        "<table><thead><tr>"
        "<th>Model / Strategy</th><th>N</th><th>Exact Row Acc.</th><th>Mean MAE</th><th>Δ MAE vs one_shot</th><th>Source CSV</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_counter(counter: Counter) -> str:
    """Render a compact counter summary."""
    if not counter:
        return "<span class=\"muted\">none</span>"
    return ", ".join(f"{html.escape(str(key))}: {value}" for key, value in counter.most_common())


def improvement_location_table(records: List[Dict]) -> str:
    """Render where strategy variants improved/regressed."""
    if not records:
        return "<p class=\"muted\">没有检测到可与 one_shot 对齐的策略变体。</p>"
    rows = []
    for record in records:
        cmp = record["comparison"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(display_name(record['base'], record['strategy']))}</td>"
            f"<td>{cmp['shared']}</td>"
            f"<td>{cmp['fixed']}</td>"
            f"<td>{cmp['regressed']}</td>"
            f"<td>{cmp['fixed'] - cmp['regressed']:+d}</td>"
            f"<td>{render_counter(record['fixed_by_type'])}</td>"
            f"<td>{render_counter(record['regressed_by_type'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Strategy</th><th>Shared</th><th>Fixed</th><th>Regressed</th><th>Net</th>"
        "<th>Fixed By QA Type</th><th>Regressed By QA Type</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_case_card(
    record: Dict,
    key: Tuple[str, str, str],
    case_type: str,
    image_index: Dict[str, Path],
    assets_dir: Path,
) -> str:
    """Render one fixed/regressed QA case card."""
    base_row = record["base_rows"][key]
    variant_row = record["variant_rows"][key]
    image_id = variant_row.get("image_id", "")
    image_rel = copy_image_asset(image_id, image_index, assets_dir)
    if image_rel:
        image_html = f'<a href="{html.escape(image_rel)}" target="_blank"><img src="{html.escape(image_rel)}" alt="{html.escape(image_id)}"></a>'
    else:
        image_html = '<div class="missing-image">Image not found</div>'
    badge = "Fixed" if case_type == "fixed" else "Regressed"
    badge_class = "fixed" if case_type == "fixed" else "regressed"
    return f"""
<article class="case-card">
  <div class="case-image">{image_html}</div>
  <div class="case-body">
    <div class="meta">
      <span class="{badge_class}">{badge}</span>
      <span>{html.escape(display_name(record['base'], record['strategy']))}</span>
      <span>{html.escape(image_id)}</span>
      <span>{html.escape(variant_row.get('qa_id', ''))}</span>
      <span>{html.escape(variant_row.get('qa_type', ''))}</span>
    </div>
    <h3>{html.escape(variant_row.get('question', ''))}</h3>
    <dl>
      <dt>Ground Truth</dt>
      <dd>{html.escape(variant_row.get('ground_truth', ''))}</dd>
      <dt>one_shot Answer</dt>
      <dd>{html.escape(base_row.get('predicted', ''))}</dd>
      <dt>{html.escape(record['strategy'])} Answer</dt>
      <dd>{html.escape(variant_row.get('predicted', ''))}</dd>
    </dl>
  </div>
</article>
"""


def improvement_examples_html(
    records: List[Dict],
    image_index: Dict[str, Path],
    assets_dir: Path,
    max_examples: int,
) -> str:
    """Render fixed and regressed example cards."""
    if not records:
        return "<p class=\"muted\">暂无可对齐的策略案例。</p>"

    sections = []
    for record in records:
        cmp = record["comparison"]
        fixed_cards = [
            render_case_card(record, key, "fixed", image_index, assets_dir)
            for key in cmp["fixed_keys"][:max_examples]
        ]
        regressed_cards = [
            render_case_card(record, key, "regressed", image_index, assets_dir)
            for key in cmp["regressed_keys"][:max_examples]
        ]
        sections.append(
            f"""
<section class="example-section">
  <h3>{html.escape(display_name(record['base'], record['strategy']))}</h3>
  <p class="muted">Fixed examples: {cmp['fixed']} total; Regressed examples: {cmp['regressed']} total. Showing up to {max_examples} each.</p>
  <h4>Fixed Examples</h4>
  <div class="cases">{''.join(fixed_cards) if fixed_cards else '<p class="muted">No fixed examples.</p>'}</div>
  <h4>Regressed Examples</h4>
  <div class="cases">{''.join(regressed_cards) if regressed_cards else '<p class="muted">No regressed examples.</p>'}</div>
</section>
"""
        )
    return "\n".join(sections)


def render_html(
    output_path: Path,
    chart_paths: List[Optional[Path]],
    qa_evaluations: Dict[Tuple[str, str], Dict],
    object_results: Dict[Tuple[str, str], Dict],
    improvement_records: List[Dict],
    image_index: Dict[str, Path],
    assets_dir: Path,
    max_examples: int,
) -> None:
    """Write HTML report."""
    chart_html = []
    for chart in chart_paths:
        if chart:
            rel = chart.relative_to(output_path.parent).as_posix()
            chart_html.append(f'<figure><img src="{html.escape(rel)}" alt="{html.escape(chart.stem)}"></figure>')

    qa_variants = sum(1 for (_, strategy) in qa_evaluations if strategy != "one_shot")
    object_variants = sum(1 for (_, strategy) in object_results if strategy != "one_shot")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AECV-Bench Prompt Strategy Analysis</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #172026; background: #fff; }}
    header {{ padding: 30px 42px 22px; background: #f4f8f9; border-bottom: 1px solid #d9e0e4; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px 32px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 12px; font-size: 22px; }}
    .muted {{ color: #63717a; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 20px 0; }}
    .card {{ border: 1px solid #d9e0e4; background: #f7f9fa; border-radius: 8px; padding: 16px; }}
    .big {{ font-size: 30px; font-weight: 700; color: #0f766e; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0 22px; }}
    th, td {{ border-top: 1px solid #d9e0e4; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f8f9; font-weight: 700; }}
    figure {{ margin: 18px 0 26px; border: 1px solid #d9e0e4; border-radius: 8px; padding: 12px; background: #fff; }}
    img {{ max-width: 100%; display: block; margin: 0 auto; }}
    code {{ background: #eef3f4; padding: 2px 5px; border-radius: 4px; }}
    .cases {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 16px; margin: 10px 0 24px; }}
    .case-card {{ display: grid; grid-template-columns: 200px 1fr; gap: 14px; border: 1px solid #d9e0e4; border-radius: 8px; padding: 12px; background: #fff; }}
    .case-image {{ display: flex; align-items: flex-start; justify-content: center; background: #f7f9fa; border: 1px solid #d9e0e4; border-radius: 6px; min-height: 180px; overflow: hidden; }}
    .case-image img {{ width: 100%; height: 200px; object-fit: contain; background: white; }}
    .case-body h3 {{ margin: 8px 0 12px; font-size: 16px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .meta span {{ border: 1px solid #d9e0e4; border-radius: 999px; padding: 3px 7px; font-size: 12px; color: #0f766e; }}
    .meta .fixed {{ background: #e7f7f1; color: #087443; border-color: #99dec2; }}
    .meta .regressed {{ background: #fff1ed; color: #b42318; border-color: #f2b8a8; }}
    dl {{ margin: 0; font-size: 13px; }}
    dt {{ margin-top: 8px; font-weight: 700; }}
    dd {{ margin: 2px 0 0; overflow-wrap: anywhere; }}
    .missing-image {{ padding: 28px; color: #63717a; }}
    @media (max-width: 760px) {{ .case-card, .cases {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>AECV-Bench Prompt Strategy Analysis</h1>
    <p class="muted">自动扫描本地 QA judge 与 object counting 结果，比较 one_shot、step_by_step、self_refine、two_pass_reflection、skill_guided 等策略。</p>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="big">{len(qa_evaluations)}</div><div>QA evaluation files loaded</div></div>
      <div class="card"><div class="big">{qa_variants}</div><div>QA strategy variants detected</div></div>
      <div class="card"><div class="big">{len(object_results)}</div><div>Object counting CSVs loaded</div></div>
      <div class="card"><div class="big">{object_variants}</div><div>Object strategy variants detected</div></div>
    </section>

    <h2>Key Visualizations</h2>
    {''.join(chart_html) if chart_html else '<p class="muted">没有足够的策略变体生成图表。</p>'}

    <h2>QA Strategy Summary</h2>
    {qa_summary_table(qa_evaluations)}

    <h2>Where Did Strategies Improve?</h2>
    {improvement_location_table(improvement_records)}

    <h2>QA Type Breakdown</h2>
    {qa_type_table(qa_evaluations)}

    <h2>Representative Fixed / Regressed Examples</h2>
    {improvement_examples_html(improvement_records, image_index, assets_dir, max_examples)}

    <h2>Object Counting Strategy Summary</h2>
    {object_summary_table(object_results)}

    <h2>Reading Notes</h2>
    <p>Fixed 表示 one_shot 错、策略答对的共享题目数量；Regressed 表示 one_shot 对、策略答错的共享题目数量。若某策略样本数少于 baseline，报告只在 shared 样本上计算 fixed/regressed。</p>
    <p>Object Counting 的 Mean MAE 越低越好；Exact Row Accuracy 表示 Door、Window、Space、Bedroom、Toilet 五个字段全部正确的样本比例。</p>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def add_pdf_page_title(fig, title: str, subtitle: str = "") -> None:
    """Draw PDF title."""
    fig.text(0.06, 0.91, title, fontsize=22, fontweight="bold", color="#172026")
    if subtitle:
        fig.text(0.06, 0.86, subtitle, fontsize=11, color="#63717a")


def truncate_text(text: str, limit: int = 380) -> str:
    """Truncate long text for PDF pages."""
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def add_pdf_case_page(
    pdf: PdfPages,
    title: str,
    record: Dict,
    key: Tuple[str, str, str],
    case_type: str,
    image_index: Dict[str, Path],
) -> None:
    """Add one fixed/regressed example page to the PDF."""
    base_row = record["base_rows"][key]
    variant_row = record["variant_rows"][key]
    image_id = variant_row.get("image_id", "")
    image_path = image_index.get(image_id)

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    add_pdf_page_title(
        fig,
        title,
        f"{case_type.upper()} | {display_name(record['base'], record['strategy'])} | {image_id} | {variant_row.get('qa_id', '')} | {variant_row.get('qa_type', '')}",
    )

    ax = fig.add_axes([0.06, 0.16, 0.40, 0.64])
    ax.axis("off")
    if image_path and image_path.exists():
        ax.imshow(plt.imread(image_path))
    else:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center")

    x = 0.50
    fig.text(x, 0.78, "Question", fontsize=10, fontweight="bold")
    fig.text(x, 0.73, truncate_text(variant_row.get("question", ""), 360), fontsize=9, wrap=True)
    fig.text(x, 0.60, "Ground Truth", fontsize=10, fontweight="bold")
    fig.text(x, 0.55, truncate_text(variant_row.get("ground_truth", ""), 260), fontsize=9, wrap=True)
    fig.text(x, 0.44, "one_shot Answer", fontsize=10, fontweight="bold")
    fig.text(x, 0.39, truncate_text(base_row.get("predicted", ""), 420), fontsize=8, wrap=True)
    fig.text(x, 0.23, f"{record['strategy']} Answer", fontsize=10, fontweight="bold")
    fig.text(x, 0.18, truncate_text(variant_row.get("predicted", ""), 420), fontsize=8, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def render_pdf(
    pdf_path: Path,
    chart_paths: List[Optional[Path]],
    qa_evaluations: Dict[Tuple[str, str], Dict],
    object_results: Dict[Tuple[str, str], Dict],
    improvement_records: List[Dict],
    image_index: Dict[str, Path],
    max_examples: int,
) -> None:
    """Write a compact PDF report."""
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        add_pdf_page_title(fig, "AECV-Bench Prompt Strategy Analysis", "Local benchmark result summary")
        fig.text(0.08, 0.73, f"QA evaluation files: {len(qa_evaluations)}", fontsize=15)
        fig.text(0.08, 0.67, f"QA strategy variants: {sum(1 for (_, s) in qa_evaluations if s != 'one_shot')}", fontsize=15)
        fig.text(0.08, 0.61, f"Object counting CSVs: {len(object_results)}", fontsize=15)
        fig.text(0.08, 0.55, f"Object strategy variants: {sum(1 for (_, s) in object_results if s != 'one_shot')}", fontsize=15)
        fig.text(0.08, 0.24, "Fixed = one_shot wrong but strategy correct. Regressed = one_shot correct but strategy wrong.", fontsize=10, color="#63717a")
        pdf.savefig(fig)
        plt.close(fig)

        for chart in chart_paths:
            if not chart or not chart.exists():
                continue
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("white")
            image = plt.imread(chart)
            ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # Summary table page for QA strategy variants.
        rows = []
        grouped = defaultdict(dict)
        for (base, strategy), data in qa_evaluations.items():
            grouped[base][strategy] = data
        for base in sorted(grouped):
            if not any(strategy != "one_shot" for strategy in grouped[base]):
                continue
            baseline = grouped[base].get("one_shot")
            for strategy in STRATEGIES:
                data = grouped[base].get(strategy)
                if not data:
                    continue
                delta = ""
                fixed = ""
                regressed = ""
                if baseline and strategy != "one_shot":
                    cmp = compare_to_baseline(baseline, data)
                    delta = f"{cmp['delta']:+.4f}"
                    fixed = str(cmp["fixed"])
                    regressed = str(cmp["regressed"])
                rows.append([display_name(base, strategy), str(data["n"]), f"{data['mean']:.4f}", delta, fixed, regressed])
        if rows:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            add_pdf_page_title(fig, "QA Strategy Table")
            table = ax.table(
                cellText=rows,
                colLabels=["Model / Strategy", "N", "Accuracy", "Delta", "Fixed", "Regressed"],
                cellLoc="left",
                colLoc="left",
                bbox=[0.04, 0.05, 0.92, 0.78],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            pdf.savefig(fig)
            plt.close(fig)

        # Improvement location table.
        improvement_rows = []
        for record in improvement_records:
            cmp = record["comparison"]
            improvement_rows.append(
                [
                    display_name(record["base"], record["strategy"]),
                    str(cmp["shared"]),
                    str(cmp["fixed"]),
                    str(cmp["regressed"]),
                    f"{cmp['fixed'] - cmp['regressed']:+d}",
                    ", ".join(f"{k}:{v}" for k, v in record["fixed_by_type"].most_common()),
                ]
            )
        if improvement_rows:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            add_pdf_page_title(fig, "Where Did Strategies Improve?")
            table = ax.table(
                cellText=improvement_rows,
                colLabels=["Strategy", "Shared", "Fixed", "Regressed", "Net", "Fixed By Type"],
                cellLoc="left",
                colLoc="left",
                bbox=[0.04, 0.08, 0.92, 0.72],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            pdf.savefig(fig)
            plt.close(fig)

        for record in improvement_records:
            cmp = record["comparison"]
            for key in cmp["fixed_keys"][:max_examples]:
                add_pdf_case_page(pdf, "Representative Fixed Example", record, key, "fixed", image_index)
            for key in cmp["regressed_keys"][:max_examples]:
                add_pdf_case_page(pdf, "Representative Regressed Example", record, key, "regressed", image_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt-strategy analysis HTML and PDF.")
    parser.add_argument("--qa-eval-dir", default="results/qa_llm_judge_results")
    parser.add_argument("--object-dir", default="benchmark_result_object_counting")
    parser.add_argument(
        "--image-dir",
        default="data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images",
        help="Directory containing QA benchmark images for example cards.",
    )
    parser.add_argument("--output-dir", default="results/strategy_analysis")
    parser.add_argument("--max-examples", type=int, default=4, help="Max fixed/regressed examples per strategy.")
    args = parser.parse_args()

    ensure_font()
    output_dir = Path(args.output_dir)
    chart_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    qa_evaluations = load_qa_evaluations(Path(args.qa_eval_dir))
    object_results = load_object_results(Path(args.object_dir))
    improvement_records = build_improvement_records(qa_evaluations)
    image_index = build_image_index(Path(args.image_dir))

    chart_paths = [
        plot_qa_overall(qa_evaluations, chart_dir),
        plot_qa_type_breakdown(qa_evaluations, chart_dir),
        plot_fixed_regressed(improvement_records, chart_dir),
        plot_object_metrics(object_results, chart_dir),
    ]

    html_path = output_dir / "strategy_analysis_report.html"
    pdf_path = output_dir / "strategy_analysis_report.pdf"
    render_html(
        html_path,
        chart_paths,
        qa_evaluations,
        object_results,
        improvement_records,
        image_index,
        chart_dir,
        args.max_examples,
    )
    render_pdf(
        pdf_path,
        chart_paths,
        qa_evaluations,
        object_results,
        improvement_records,
        image_index,
        args.max_examples,
    )

    print(f"HTML report saved to: {html_path.resolve()}")
    print(f"PDF report saved to: {pdf_path.resolve()}")
    print(f"Charts saved to: {chart_dir.resolve()}")
    print(f"QA evaluations loaded: {len(qa_evaluations)}")
    print(f"Object counting results loaded: {len(object_results)}")
    print(f"Improvement comparisons built: {len(improvement_records)}")


if __name__ == "__main__":
    main()
