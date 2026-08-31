import numpy as np

from depthrecall.surface import evaluate_surface


def test_bidirectional_surface_metrics_distinguish_accuracy_and_completeness():
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    predicted = np.array([[0.0, 0.0, 0.0]])
    result = evaluate_surface(predicted, reference, np.array([0.1, 2.1]))
    assert result.accuracy == 0.0
    assert result.completeness == 1.0
    np.testing.assert_allclose(result.precision, [1.0, 1.0])
    np.testing.assert_allclose(result.recall, [0.5, 1.0])
    np.testing.assert_allclose(result.fscore, [2 / 3, 1.0])
