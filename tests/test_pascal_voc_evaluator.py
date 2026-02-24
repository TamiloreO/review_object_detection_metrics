"""Unit tests for pascal_voc_evaluator helper functions."""

from math import isclose

import numpy as np
import pandas as pd
import pytest
from src.bounding_box import BoundingBox
from src.evaluators.pascal_voc_evaluator import (
    _group_boxes_by_class,
    _match_detections_to_ground_truth,
    _compute_precision_recall,
    _compute_average_precision,
    _build_results_table,
    _evaluate_class,
    _compute_map,
    calculate_ap_every_point,
    calculate_ap_11_point_interp,
    get_pascalvoc_metrics,
)
from src.utils.enumerators import BBFormat, BBType, MethodAveragePrecision


def create_gt_box(image_name, class_id, coords):
    """Helper to create a ground truth bounding box."""
    return BoundingBox(
        image_name=image_name,
        class_id=class_id,
        coordinates=coords,
        bb_type=BBType.GROUND_TRUTH,
        format=BBFormat.XYWH
    )


def create_det_box(image_name, class_id, coords, confidence):
    """Helper to create a detection bounding box."""
    return BoundingBox(
        image_name=image_name,
        class_id=class_id,
        coordinates=coords,
        bb_type=BBType.DETECTED,
        confidence=confidence,
        format=BBFormat.XYWH
    )


class TestGroupBoxesByClass:
    """Tests for _group_boxes_by_class function."""

    def test_single_class(self):
        gt_boxes = [
            create_gt_box('img1', 'cat', (10, 10, 50, 50)),
            create_gt_box('img2', 'cat', (20, 20, 60, 60)),
        ]
        det_boxes = [
            create_det_box('img1', 'cat', (12, 12, 48, 48), 0.9),
        ]

        classes_bbs, gt_classes = _group_boxes_by_class(gt_boxes, det_boxes)

        assert 'cat' in classes_bbs
        assert gt_classes == ['cat']
        assert len(classes_bbs['cat']['gt']) == 2
        assert len(classes_bbs['cat']['det']) == 1

    def test_multiple_classes(self):
        gt_boxes = [
            create_gt_box('img1', 'cat', (10, 10, 50, 50)),
            create_gt_box('img1', 'dog', (100, 100, 50, 50)),
            create_gt_box('img2', 'cat', (20, 20, 60, 60)),
        ]
        det_boxes = [
            create_det_box('img1', 'cat', (12, 12, 48, 48), 0.9),
            create_det_box('img1', 'dog', (102, 102, 48, 48), 0.8),
        ]

        classes_bbs, gt_classes = _group_boxes_by_class(gt_boxes, det_boxes)

        assert 'cat' in classes_bbs
        assert 'dog' in classes_bbs
        assert set(gt_classes) == {'cat', 'dog'}
        assert len(classes_bbs['cat']['gt']) == 2
        assert len(classes_bbs['cat']['det']) == 1
        assert len(classes_bbs['dog']['gt']) == 1
        assert len(classes_bbs['dog']['det']) == 1

    def test_detection_only_class(self):
        gt_boxes = [
            create_gt_box('img1', 'cat', (10, 10, 50, 50)),
        ]
        det_boxes = [
            create_det_box('img1', 'cat', (12, 12, 48, 48), 0.9),
            create_det_box('img1', 'bird', (200, 200, 30, 30), 0.7),
        ]

        classes_bbs, gt_classes = _group_boxes_by_class(gt_boxes, det_boxes)

        assert 'cat' in classes_bbs
        assert 'bird' in classes_bbs
        assert gt_classes == ['cat']
        assert len(classes_bbs['bird']['gt']) == 0
        assert len(classes_bbs['bird']['det']) == 1

    def test_empty_inputs(self):
        classes_bbs, gt_classes = _group_boxes_by_class([], [])

        assert classes_bbs == {}
        assert gt_classes == []


class TestMatchDetectionsToGroundTruth:
    """Tests for _match_detections_to_ground_truth function."""

    def test_perfect_match(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        assert TP[0] == 1
        assert FP[0] == 0

    def test_no_overlap(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (200, 200, 50, 50), 0.9)]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        assert TP[0] == 0
        assert FP[0] == 1

    def test_partial_overlap_below_threshold(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (40, 40, 50, 50), 0.9)]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        assert TP[0] == 0
        assert FP[0] == 1

    def test_partial_overlap_above_threshold(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (15, 15, 50, 50), 0.9)]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        assert TP[0] == 1
        assert FP[0] == 0

    def test_duplicate_detection(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9),
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.8),
        ]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        assert TP[0] == 1
        assert FP[0] == 0
        assert TP[1] == 0
        assert FP[1] == 1

    def test_confidence_ordering(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.5),
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9),
        ]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        # Higher confidence should be processed first and get the TP
        assert TP[0] == 1  # 0.9 confidence
        assert TP[1] == 0  # 0.5 confidence
        assert FP[0] == 0
        assert FP[1] == 1

    def test_multiple_gt_multiple_det(self):
        gt_boxes = [
            create_gt_box('img1', 'cat', (10, 10, 50, 50)),
            create_gt_box('img1', 'cat', (100, 100, 50, 50)),
        ]
        det_boxes = [
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9),
            create_det_box('img1', 'cat', (100, 100, 50, 50), 0.8),
        ]

        TP, FP, _ = _match_detections_to_ground_truth(det_boxes, gt_boxes, 0.5)

        assert np.sum(TP) == 2
        assert np.sum(FP) == 0

    def test_generate_table(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        TP, FP, dict_table = _match_detections_to_ground_truth(
            det_boxes, gt_boxes, 0.5, generate_table=True
        )

        assert dict_table is not None
        assert 'image' in dict_table
        assert 'confidence' in dict_table
        assert 'TP' in dict_table
        assert 'FP' in dict_table
        assert dict_table['image'] == ['img1']
        assert dict_table['confidence'] == ['90.00%']
        assert dict_table['TP'] == [1]
        assert dict_table['FP'] == [0]

    def test_empty_detections(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]

        TP, FP, _ = _match_detections_to_ground_truth([], gt_boxes, 0.5)

        assert len(TP) == 0
        assert len(FP) == 0


class TestComputePrecisionRecall:
    """Tests for _compute_precision_recall function."""

    def test_all_true_positives(self):
        TP = np.array([1, 1, 1])
        FP = np.array([0, 0, 0])
        npos = 3

        prec, rec, acc_TP, acc_FP = _compute_precision_recall(TP, FP, npos)

        np.testing.assert_array_almost_equal(prec, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(rec, [1/3, 2/3, 1.0])
        np.testing.assert_array_equal(acc_TP, [1, 2, 3])
        np.testing.assert_array_equal(acc_FP, [0, 0, 0])

    def test_all_false_positives(self):
        TP = np.array([0, 0, 0])
        FP = np.array([1, 1, 1])
        npos = 3

        prec, rec, acc_TP, acc_FP = _compute_precision_recall(TP, FP, npos)

        np.testing.assert_array_almost_equal(prec, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rec, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(acc_TP, [0, 0, 0])
        np.testing.assert_array_equal(acc_FP, [1, 2, 3])

    def test_mixed_results(self):
        TP = np.array([1, 0, 1, 0])
        FP = np.array([0, 1, 0, 1])
        npos = 4

        prec, rec, acc_TP, acc_FP = _compute_precision_recall(TP, FP, npos)

        # acc_TP: [1, 1, 2, 2]
        # acc_FP: [0, 1, 1, 2]
        # prec: [1/1, 1/2, 2/3, 2/4]
        # rec: [1/4, 1/4, 2/4, 2/4]
        np.testing.assert_array_almost_equal(prec, [1.0, 0.5, 2/3, 0.5])
        np.testing.assert_array_almost_equal(rec, [0.25, 0.25, 0.5, 0.5])


class TestComputeAveragePrecision:
    """Tests for _compute_average_precision function."""

    def test_every_point_interpolation(self):
        rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prec = np.array([1.0, 0.8, 0.7, 0.6, 0.5])

        ap, mpre, mrec, _ = _compute_average_precision(
            rec, prec, MethodAveragePrecision.EVERY_POINT_INTERPOLATION
        )

        assert 0 <= ap <= 1

    def test_eleven_point_interpolation(self):
        rec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prec = np.array([1.0, 0.8, 0.7, 0.6, 0.5])

        ap, mpre, mrec, _ = _compute_average_precision(
            rec, prec, MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
        )

        assert 0 <= ap <= 1

    def test_perfect_precision_recall(self):
        rec = np.array([0.25, 0.5, 0.75, 1.0])
        prec = np.array([1.0, 1.0, 1.0, 1.0])

        ap, _, _, _ = _compute_average_precision(
            rec, prec, MethodAveragePrecision.EVERY_POINT_INTERPOLATION
        )

        assert isclose(ap, 1.0)

    def test_invalid_method(self):
        rec = np.array([0.1, 0.2])
        prec = np.array([1.0, 0.8])

        with pytest.raises(ValueError):
            _compute_average_precision(rec, prec, "invalid_method")


class TestBuildResultsTable:
    """Tests for _build_results_table function."""

    def test_basic_table(self):
        dict_table = {
            'image': ['img1', 'img2'],
            'confidence': ['90.00%', '80.00%'],
            'TP': [1, 0],
            'FP': [0, 1],
            'acc TP': [],
            'acc FP': [],
            'precision': [],
            'recall': []
        }
        acc_TP = np.array([1, 1])
        acc_FP = np.array([0, 1])
        prec = np.array([1.0, 0.5])
        rec = np.array([0.5, 0.5])

        table = _build_results_table(dict_table, acc_TP, acc_FP, prec, rec)

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert list(table.columns) == ['image', 'confidence', 'TP', 'FP', 'acc TP', 'acc FP', 'precision', 'recall']
        assert list(table['acc TP']) == [1, 1]
        assert list(table['acc FP']) == [0, 1]


class TestEvaluateClass:
    """Tests for _evaluate_class function."""

    def test_perfect_detection(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        result = _evaluate_class(
            gt_boxes, det_boxes, 0.5, MethodAveragePrecision.EVERY_POINT_INTERPOLATION
        )

        assert result['AP'] == 1.0
        assert result['total positives'] == 1
        assert result['total TP'] == 1
        assert result['total FP'] == 0
        assert result['iou'] == 0.5
        assert result['method'] == MethodAveragePrecision.EVERY_POINT_INTERPOLATION

    def test_no_detection(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = []

        result = _evaluate_class(
            gt_boxes, det_boxes, 0.5, MethodAveragePrecision.EVERY_POINT_INTERPOLATION
        )

        assert result['AP'] == 0.0
        assert result['total positives'] == 1
        assert result['total TP'] == 0
        assert result['total FP'] == 0

    def test_false_positive_only(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (200, 200, 50, 50), 0.9)]

        result = _evaluate_class(
            gt_boxes, det_boxes, 0.5, MethodAveragePrecision.EVERY_POINT_INTERPOLATION
        )

        assert result['AP'] == 0.0
        assert result['total TP'] == 0
        assert result['total FP'] == 1

    def test_with_table_generation(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        result = _evaluate_class(
            gt_boxes, det_boxes, 0.5, MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=True
        )

        assert result['table'] is not None
        assert isinstance(result['table'], pd.DataFrame)

    def test_without_table_generation(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        result = _evaluate_class(
            gt_boxes, det_boxes, 0.5, MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
            generate_table=False
        )

        assert result['table'] is None


class TestComputeMap:
    """Tests for _compute_map function."""

    def test_single_class(self):
        per_class_results = {'cat': {'AP': 0.8}}
        gt_classes = ['cat']

        mAP = _compute_map(per_class_results, gt_classes)

        assert mAP == 0.8

    def test_multiple_classes(self):
        per_class_results = {
            'cat': {'AP': 0.8},
            'dog': {'AP': 0.6},
            'bird': {'AP': 0.4}
        }
        gt_classes = ['cat', 'dog', 'bird']

        mAP = _compute_map(per_class_results, gt_classes)

        assert isclose(mAP, 0.6)

    def test_ignores_non_gt_classes(self):
        per_class_results = {
            'cat': {'AP': 0.8},
            'dog': {'AP': 0.6},
            'unknown': {'AP': 0.0}
        }
        gt_classes = ['cat', 'dog']

        mAP = _compute_map(per_class_results, gt_classes)

        assert isclose(mAP, 0.7)


class TestCalculateAPEveryPoint:
    """Tests for calculate_ap_every_point function."""

    def test_perfect_curve(self):
        rec = np.array([0.25, 0.5, 0.75, 1.0])
        prec = np.array([1.0, 1.0, 1.0, 1.0])

        ap, mpre, mrec, ii = calculate_ap_every_point(rec, prec)

        assert isclose(ap, 1.0)

    def test_empty_arrays(self):
        rec = np.array([])
        prec = np.array([])

        ap, mpre, mrec, ii = calculate_ap_every_point(rec, prec)

        assert ap == 0


class TestCalculateAP11PointInterp:
    """Tests for calculate_ap_11_point_interp function."""

    def test_perfect_curve(self):
        rec = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        prec = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ap, rhoInterp, recallValues, _ = calculate_ap_11_point_interp(rec, prec)

        assert isclose(ap, 1.0)

    def test_zero_precision(self):
        rec = np.array([0.1, 0.2, 0.3])
        prec = np.array([0.0, 0.0, 0.0])

        ap, _, _, _ = calculate_ap_11_point_interp(rec, prec)

        assert ap == 0.0


class TestGetPascalvocMetricsIntegration:
    """Integration tests for get_pascalvoc_metrics function."""

    def test_basic_evaluation(self):
        gt_boxes = [
            create_gt_box('img1', 'cat', (10, 10, 50, 50)),
            create_gt_box('img2', 'cat', (20, 20, 60, 60)),
        ]
        det_boxes = [
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9),
            create_det_box('img2', 'cat', (20, 20, 60, 60), 0.8),
        ]

        results = get_pascalvoc_metrics(gt_boxes, det_boxes, iou_threshold=0.5)

        assert 'per_class' in results
        assert 'mAP' in results
        assert 'cat' in results['per_class']
        assert results['mAP'] == 1.0

    def test_multi_class_evaluation(self):
        gt_boxes = [
            create_gt_box('img1', 'cat', (10, 10, 50, 50)),
            create_gt_box('img1', 'dog', (100, 100, 50, 50)),
        ]
        det_boxes = [
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9),
            create_det_box('img1', 'dog', (100, 100, 50, 50), 0.8),
        ]

        results = get_pascalvoc_metrics(gt_boxes, det_boxes, iou_threshold=0.5)

        assert 'cat' in results['per_class']
        assert 'dog' in results['per_class']
        assert results['per_class']['cat']['AP'] == 1.0
        assert results['per_class']['dog']['AP'] == 1.0
        assert results['mAP'] == 1.0

    def test_detection_only_class_ignored(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [
            create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9),
            create_det_box('img1', 'bird', (200, 200, 30, 30), 0.7),
        ]

        results = get_pascalvoc_metrics(gt_boxes, det_boxes, iou_threshold=0.5)

        assert 'cat' in results['per_class']
        assert 'bird' not in results['per_class']

    def test_with_table_generation(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        results = get_pascalvoc_metrics(
            gt_boxes, det_boxes, iou_threshold=0.5, generate_table=True
        )

        assert results['per_class']['cat']['table'] is not None

    def test_different_iou_thresholds(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (15, 15, 50, 50), 0.9)]

        results_low = get_pascalvoc_metrics(gt_boxes, det_boxes, iou_threshold=0.3)
        results_high = get_pascalvoc_metrics(gt_boxes, det_boxes, iou_threshold=0.9)

        assert results_low['per_class']['cat']['AP'] >= results_high['per_class']['cat']['AP']

    def test_eleven_point_interpolation(self):
        gt_boxes = [create_gt_box('img1', 'cat', (10, 10, 50, 50))]
        det_boxes = [create_det_box('img1', 'cat', (10, 10, 50, 50), 0.9)]

        results = get_pascalvoc_metrics(
            gt_boxes, det_boxes,
            method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
        )

        assert results['per_class']['cat']['method'] == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
