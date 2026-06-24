"""
Benchmark processing and evaluation utilities.
"""
from .processor import process_benchmark_floorplans
from .evaluator import compute_recalls_and_completeness


def plot_benchmark_results(*args, **kwargs):
    from .visualizer import plot_benchmark_results as _plot_benchmark_results

    return _plot_benchmark_results(*args, **kwargs)


def plot_all_models_comparison(*args, **kwargs):
    from .visualizer import plot_all_models_comparison as _plot_all_models_comparison

    return _plot_all_models_comparison(*args, **kwargs)

__all__ = [
    'process_benchmark_floorplans',
    'compute_recalls_and_completeness',
    'plot_benchmark_results',
    'plot_all_models_comparison',
]

