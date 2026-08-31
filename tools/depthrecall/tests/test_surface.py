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


def test_chunked_surface_metrics_match_single_chunk_evaluation():
    rng = np.random.default_rng(2)
    predicted = rng.normal(size=(101, 3))
    reference = rng.normal(size=(137, 3))
    taus = np.array([0.1, 0.5, 1.0])

    single_chunk = evaluate_surface(predicted, reference, taus, query_chunk_size=1_000, workers=1)
    chunks = evaluate_surface(predicted, reference, taus, query_chunk_size=17, workers=1)

    np.testing.assert_allclose(chunks.accuracy, single_chunk.accuracy)
    np.testing.assert_allclose(chunks.completeness, single_chunk.completeness)
    np.testing.assert_allclose(chunks.precision, single_chunk.precision)
    np.testing.assert_allclose(chunks.recall, single_chunk.recall)
    np.testing.assert_allclose(chunks.fscore, single_chunk.fscore)
