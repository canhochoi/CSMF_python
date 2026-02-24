"""
Utility modules for NMF algorithms.
"""

from .stopping_criteria import get_stop_criterion
from .hungarian import hungarian
from .evaluation import (
    compute_reconstruction_error,
    sparsity,
    matrix_similarity,
    compute_accuracy,
    normalize_factors,
)
from .rank_selection import (
    amari_distance,
    stability_between_solutions,
    compute_pairwise_stability,
    rank_stability_score,
    nenmf_rank_sweep_per_dataset,
    analyze_stability_per_dataset,
    find_optimal_ranks,
    select_best_factorization,
    learn_common_specific_ranks_from_correlations,
    rank_selection_pipeline,
)

__all__ = [
    'get_stop_criterion',
    'hungarian',
    'compute_reconstruction_error',
    'sparsity',
    'matrix_similarity',
    'compute_accuracy',
    'normalize_factors',
    # Rank selection (MATLAB-matched implementation)
    'amari_distance',
    'stability_between_solutions',
    'compute_pairwise_stability',
    'rank_stability_score',
    'nenmf_rank_sweep_per_dataset',
    'analyze_stability_per_dataset',
    'find_optimal_ranks',
    'select_best_factorization',
    'learn_common_specific_ranks_from_correlations',
    'rank_selection_pipeline',
]
