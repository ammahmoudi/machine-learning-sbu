"""Checks of the mathematics, preprocessing boundaries, and saved experiment."""

import json
import math
import unittest

import numpy as np
from regression import (HERE, DEFAULT_DATA, choose_centers, design_matrix, fit_least_squares,
                        fit_scaler, gaussian_basis, load_data, rmse, shuffled_indices, sse, transform)


class RegressionTests(unittest.TestCase):
    def test_known_basis_and_intercept(self):
        x = np.array([[0.0], [2.0]])
        center = np.array([[0.0]])
        actual = gaussian_basis(x, center, 1.0)
        self.assertAlmostEqual(actual[0, 0], 1.0)
        self.assertAlmostEqual(actual[1, 0], math.exp(-2))
        np.testing.assert_allclose(design_matrix(x, center, 1.0)[:, 0], 1.0)
        with self.assertRaises(ValueError):
            gaussian_basis(x, center, 0)

    def test_error_equations(self):
        self.assertEqual(sse(np.array([1., 3.]), np.array([2., 1.])), 5.0)
        self.assertAlmostEqual(rmse(np.array([1., 3.]), np.array([2., 1.])), math.sqrt(2.5))
        with self.assertRaises(ValueError):
            sse(np.array([1.]), np.array([[1.]]))

    def test_recover_synthetic_weights(self):
        x = np.array([[i / 10] for i in range(30)])
        centers = x[[3, 15, 27]]
        matrix = design_matrix(x, centers, 0.6)
        expected = np.array([2., -1., 3., 0.5])
        target = matrix @ expected
        actual, rank, _ = fit_least_squares(matrix, target)
        self.assertEqual(rank, 4)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_scaler_uses_supplied_training_only(self):
        training = np.array([[1., 2., 7., 0.], [3., 4., 7., 1.]])
        means, scales = fit_scaler(training)
        np.testing.assert_allclose(means, [2., 3., 7., 0.])
        np.testing.assert_allclose(scales, [1., 1., 1., 1.])
        held_out = np.array([[100., 200., 7., 1.]])
        np.testing.assert_allclose(transform(held_out, means, scales), [[98., 197., 0., 1.]])

    def test_split_and_centers_are_reproducible(self):
        order = shuffled_indices(731, 42)
        self.assertEqual(len(set(order.tolist())), 731)
        self.assertFalse(set(order[:584]) & set(order[584:]))
        np.testing.assert_array_equal(order, shuffled_indices(731, 42))
        centers = choose_centers(584)
        self.assertEqual(len(set(centers.tolist())), 64)
        self.assertTrue(all(0 <= index < 584 for index in centers))
        np.testing.assert_array_equal(centers, choose_centers(584, count=128)[:64])

    @unittest.skipUnless(DEFAULT_DATA.exists(), "Download the UCI dataset first")
    def test_dataset_schema_and_no_target_inputs(self):
        rows, x, y, names = load_data(DEFAULT_DATA)
        self.assertEqual((len(rows), x.shape, y.shape), (731, (731, 21), (731,)))
        self.assertFalse({"total_rentals", "casual_rentals", "registered_rentals", "record_id"} & set(names))

    @unittest.skipUnless((HERE / "results/metadata.json").exists(), "Run the experiment first")
    def test_saved_model_and_least_squares_stationarity(self):
        _, x, y, _ = load_data(DEFAULT_DATA)
        metadata = json.loads((HERE / "results/metadata.json").read_text())
        train, test = metadata["train_indices"], metadata["test_indices"]
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(metadata["inner_fit_indices"]) | set(metadata["validation_indices"]), set(train))
        self.assertFalse(set(metadata["inner_fit_indices"]) & set(metadata["validation_indices"]))
        self.assertTrue(set(metadata["center_indices"]) <= set(train))
        with np.load(HERE / "results/model.npz") as model:
            matrix = design_matrix(transform(x[train], model["means"], model["scales"]),
                                   model["centers"], float(model["width"]))
            residual = matrix @ model["weights"] - y[train]
            gradient = matrix.T @ residual
            self.assertLess(float(max(abs(gradient))) / max(1., float(max(abs(matrix.T @ y[train])))), 1e-9)

    @unittest.skipUnless((HERE / "results/center_count_model.npz").exists(), "Run the experiment first")
    def test_center_count_selection_and_saved_predictions(self):
        _, x, y, _ = load_data(DEFAULT_DATA)
        metadata = json.loads((HERE / "results/metadata.json").read_text())
        study = metadata["center_count_study"]
        selected = min(study["results"], key=lambda row: row["validation_rmse"])
        self.assertEqual(study["selected_count"], selected["center_count"])
        train_errors = [row["train_rmse"] for row in study["results"]]
        self.assertTrue(all(b <= a + 1e-7 for a, b in zip(train_errors, train_errors[1:])))
        self.assertTrue(all(count + 1 < metadata["inner_fit_count"] for count in study["counts"]))
        with np.load(HERE / "results/center_count_model.npz") as model:
            np.testing.assert_array_equal(model["train_indices"], metadata["train_indices"])
            np.testing.assert_array_equal(model["test_indices"], metadata["test_indices"])
            z_train = transform(x[model["train_indices"]], model["means"], model["scales"])
            expected_centers = z_train[choose_centers(len(z_train), count=study["selected_count"])]
            np.testing.assert_array_equal(model["centers"], expected_centers)
            test = model["test_indices"]
            prediction = design_matrix(transform(x[test], model["means"], model["scales"]),
                                       model["centers"], float(model["width"])) @ model["weights"]
            self.assertAlmostEqual(rmse(y[test], prediction), selected["test_rmse"], places=7)

    @unittest.skipUnless((HERE / "results/grid_model.npz").exists(), "Run the experiment first")
    def test_full_grid_coverage_selection_and_saved_model(self):
        _, x, y, _ = load_data(DEFAULT_DATA)
        metadata = json.loads((HERE / "results/metadata.json").read_text())
        grid = metadata["grid_search"]
        settings = {(r["center_count"], r["width"]): r for r in grid["results"]}
        expected = {(count, width) for count in grid["counts"] for width in grid["widths"]}
        self.assertEqual(set(settings), expected)
        self.assertEqual(len(settings), len(grid["results"]))
        best = min(grid["results"], key=lambda r: r["validation_rmse"])
        self.assertEqual((grid["selected_count"], grid["selected_width"]),
                         (best["center_count"], best["width"]))
        # The earlier center-count sweep must be an exact slice of this grid.
        for row in metadata["center_count_study"]["results"]:
            actual = settings[row["center_count"], row["width"]]
            for metric in ("train_rmse", "test_rmse", "validation_rmse"):
                self.assertAlmostEqual(actual[metric], row[metric], places=7)
        with np.load(HERE / "results/grid_model.npz") as model:
            np.testing.assert_array_equal(model["train_indices"], metadata["train_indices"])
            np.testing.assert_array_equal(model["test_indices"], metadata["test_indices"])
            self.assertEqual(len(model["centers"]), best["center_count"])
            self.assertEqual(float(model["width"]), best["width"])
            z_train = transform(x[model["train_indices"]], model["means"], model["scales"])
            np.testing.assert_array_equal(model["centers"],
                z_train[choose_centers(len(z_train), count=best["center_count"])])
            for indices, metric in [(model["train_indices"], "train_rmse"), (model["test_indices"], "test_rmse")]:
                prediction = design_matrix(transform(x[indices], model["means"], model["scales"]),
                                           model["centers"], float(model["width"])) @ model["weights"]
                self.assertAlmostEqual(rmse(y[indices], prediction), best[metric], places=7)


if __name__ == "__main__":
    unittest.main()
