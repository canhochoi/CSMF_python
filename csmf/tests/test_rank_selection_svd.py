import numpy as np
import pytest

from csmf.utils.rank_selection_svd import rank_selection_svd_pipeline


def _diagonal_dataset(values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(values), len(values)), dtype=float)
    np.fill_diagonal(matrix, values)
    return matrix


def test_stack_svd_methods_agree() -> None:
    diag_1 = _diagonal_dataset(np.array([5.0, 3.0, 1.0, 0.2]))
    diag_2 = _diagonal_dataset(np.array([4.0, 2.5, 0.8, 0.1]))

    vec_full, analysis_full = rank_selection_svd_pipeline(
        [diag_1, diag_2],
        verbose=0,
        stack_svd_method='full',
    )
    vec_gram, analysis_gram = rank_selection_svd_pipeline(
        [diag_1, diag_2],
        verbose=0,
        stack_svd_method='gram',
    )

    assert vec_full == vec_gram
    assert analysis_full['r_common'] == analysis_gram['r_common']


def test_variance_threshold_limits_rank() -> None:
    diag = _diagonal_dataset(np.array([5.0, 3.0, 1.0, 0.2, 0.1]))
    _, analysis = rank_selection_svd_pipeline(
        [diag],
        verbose=0,
        variance_threshold=0.9,
        stack_svd_method='gram',
    )

    assert analysis['optimal_ranks'][0] == 2


def test_randomized_stack_svd_matches_gram() -> None:
    pytest.importorskip('sklearn')
    diag_1 = _diagonal_dataset(np.array([5.0, 3.0, 1.0, 0.2]))
    diag_2 = _diagonal_dataset(np.array([4.0, 2.5, 0.8, 0.1]))

    vec_gram, _ = rank_selection_svd_pipeline(
        [diag_1, diag_2],
        verbose=0,
        stack_svd_method='gram',
    )
    vec_randomized, _ = rank_selection_svd_pipeline(
        [diag_1, diag_2],
        verbose=0,
        stack_svd_method='randomized',
        stack_svd_components=3,
        stack_svd_random_state=0,
    )

    assert vec_randomized == vec_gram
