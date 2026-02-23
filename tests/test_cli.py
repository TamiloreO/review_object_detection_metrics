"""
Unit tests for the CLI module.

Tests follow the AAA pattern (Arrange, Act, Assert) and cover:
- Argument parsing
- Configuration validation
- Annotation loading
- Metrics evaluation
- Results formatting
- End-to-end integration
"""

import json
import os
import sys
import tempfile
from math import isclose
from unittest.mock import MagicMock, patch

import pytest

from cli import (
    APInterpolationMethod,
    AnnotationLoadError,
    CLIApplication,
    CLIConfig,
    ConfigurationError,
    DetectionFormat,
    DetectionLoader,
    GroundTruthFormat,
    GroundTruthLoader,
    MetricsEvaluator,
    MetricType,
    ResultsFormatter,
    create_argument_parser,
    main,
    parse_arguments,
)


class TestArgumentParsing:
    """Test suite for command line argument parsing."""

    def test_parse_required_arguments_only(self):
        # Arrange
        args = [
            "--gt-dir", "/path/to/gts",
            "--det-dir", "/path/to/dets",
            "--gt-format", "pascalvoc",
            "--det-format", "xywh_abs",
        ]

        # Act
        config = parse_arguments(args)

        # Assert
        assert config.gt_dir == "/path/to/gts"
        assert config.det_dir == "/path/to/dets"
        assert config.gt_format == GroundTruthFormat.PASCALVOC
        assert config.det_format == DetectionFormat.XYWH_ABS

    def test_parse_all_optional_arguments(self):
        # Arrange
        args = [
            "--gt-dir", "/path/to/gts",
            "--det-dir", "/path/to/dets",
            "--gt-format", "coco",
            "--det-format", "coco",
            "--output-dir", "/path/to/output",
            "--images-dir", "/path/to/images",
            "--gt-classes-file", "/path/to/classes.txt",
            "--det-classes-file", "/path/to/det_classes.txt",
            "--metric", "pascal",
            "--iou", "0.75",
            "--ap-method", "eleven_point",
            "--save-plots",
            "--output-format", "table",
            "--quiet",
        ]

        # Act
        config = parse_arguments(args)

        # Assert
        assert config.output_dir == "/path/to/output"
        assert config.images_dir == "/path/to/images"
        assert config.gt_classes_file == "/path/to/classes.txt"
        assert config.det_classes_file == "/path/to/det_classes.txt"
        assert config.metric == MetricType.PASCAL
        assert config.iou_threshold == 0.75
        assert config.ap_method == APInterpolationMethod.ELEVEN_POINT
        assert config.save_plots is True
        assert config.output_format == "table"
        assert config.quiet is True

    def test_parse_default_values(self):
        # Arrange
        args = [
            "--gt-dir", "/path/to/gts",
            "--det-dir", "/path/to/dets",
            "--gt-format", "pascalvoc",
            "--det-format", "xywh_abs",
        ]

        # Act
        config = parse_arguments(args)

        # Assert
        assert config.metric == MetricType.ALL
        assert config.iou_threshold == 0.5
        assert config.ap_method == APInterpolationMethod.EVERY_POINT
        assert config.save_plots is False
        assert config.output_format == "json"
        assert config.quiet is False

    def test_parse_missing_required_argument_raises_error(self):
        # Arrange
        args = [
            "--gt-dir", "/path/to/gts",
            "--gt-format", "pascalvoc",
            "--det-format", "xywh_abs",
        ]

        # Act & Assert
        with pytest.raises(SystemExit):
            parse_arguments(args)

    def test_parse_invalid_gt_format_raises_error(self):
        # Arrange
        args = [
            "--gt-dir", "/path/to/gts",
            "--det-dir", "/path/to/dets",
            "--gt-format", "invalid_format",
            "--det-format", "xywh_abs",
        ]

        # Act & Assert
        with pytest.raises(SystemExit):
            parse_arguments(args)

    def test_parse_all_gt_formats(self):
        # Arrange
        gt_formats = ["coco", "cvat", "openimage", "labelme", "pascalvoc", "imagenet", "yolo", "text_abs"]

        for fmt in gt_formats:
            args = [
                "--gt-dir", "/path/to/gts",
                "--det-dir", "/path/to/dets",
                "--gt-format", fmt,
                "--det-format", "xywh_abs",
            ]

            # Act
            config = parse_arguments(args)

            # Assert
            assert config.gt_format == GroundTruthFormat(fmt)

    def test_parse_all_det_formats(self):
        # Arrange
        det_formats = ["coco", "xywh_abs", "xyx2y2_abs", "yolo_rel"]

        for fmt in det_formats:
            args = [
                "--gt-dir", "/path/to/gts",
                "--det-dir", "/path/to/dets",
                "--gt-format", "pascalvoc",
                "--det-format", fmt,
            ]

            # Act
            config = parse_arguments(args)

            # Assert
            assert config.det_format == DetectionFormat(fmt)


class TestCLIConfig:
    """Test suite for CLIConfig dataclass."""

    def test_config_creation_with_defaults(self):
        # Arrange & Act
        config = CLIConfig(
            gt_dir="/path/to/gts",
            det_dir="/path/to/dets",
            gt_format=GroundTruthFormat.PASCALVOC,
            det_format=DetectionFormat.XYWH_ABS,
        )

        # Assert
        assert config.output_dir is None
        assert config.images_dir is None
        assert config.metric == MetricType.ALL
        assert config.iou_threshold == 0.5

    def test_config_creation_with_all_parameters(self):
        # Arrange & Act
        config = CLIConfig(
            gt_dir="/path/to/gts",
            det_dir="/path/to/dets",
            gt_format=GroundTruthFormat.COCO,
            det_format=DetectionFormat.COCO,
            output_dir="/path/to/output",
            images_dir="/path/to/images",
            gt_classes_file="/path/to/classes.txt",
            det_classes_file="/path/to/det_classes.txt",
            metric=MetricType.COCO,
            iou_threshold=0.75,
            ap_method=APInterpolationMethod.ELEVEN_POINT,
            save_plots=True,
            quiet=True,
            output_format="table",
        )

        # Assert
        assert config.gt_dir == "/path/to/gts"
        assert config.metric == MetricType.COCO
        assert config.save_plots is True


class TestResultsFormatter:
    """Test suite for ResultsFormatter."""

    def test_format_json_output(self):
        # Arrange
        results = {
            "pascal": {
                "mAP": 0.75,
                "per_class": {
                    "cat": {"AP": 0.8, "total_TP": 10, "total_FP": 2, "total_positives": 12}
                },
                "iou_threshold": 0.5,
                "method": "every_point",
            }
        }

        # Act
        output = ResultsFormatter.format(results, "json")

        # Assert
        parsed = json.loads(output)
        assert parsed["pascal"]["mAP"] == 0.75
        assert parsed["pascal"]["per_class"]["cat"]["AP"] == 0.8

    def test_format_table_output_pascal(self):
        # Arrange
        results = {
            "pascal": {
                "mAP": 0.75,
                "per_class": {
                    "cat": {"AP": 0.8, "total_TP": 10, "total_FP": 2, "total_positives": 12}
                },
                "iou_threshold": 0.5,
                "method": "every_point",
            }
        }

        # Act
        output = ResultsFormatter.format(results, "table")

        # Assert
        assert "PASCAL VOC Metrics" in output
        assert "mAP @ IoU=0.5: 0.7500" in output
        assert "cat" in output

    def test_format_table_output_coco(self):
        # Arrange
        results = {
            "coco": {
                "AP": 0.5,
                "AP50": 0.7,
                "AP75": 0.55,
                "APsmall": 0.4,
                "APmedium": 0.5,
                "APlarge": 0.6,
                "AR1": 0.3,
                "AR10": 0.5,
                "AR100": 0.55,
                "ARsmall": 0.35,
                "ARmedium": 0.5,
                "ARlarge": 0.6,
            }
        }

        # Act
        output = ResultsFormatter.format(results, "table")

        # Assert
        assert "COCO Metrics" in output
        assert "AP (IoU=0.50:0.95): 0.5000" in output

    def test_format_unsupported_format_raises_error(self):
        # Arrange
        results = {"test": "data"}

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported output format"):
            ResultsFormatter.format(results, "xml")


class TestGroundTruthLoader:
    """Test suite for GroundTruthLoader."""

    def test_load_pascalvoc_format(self):
        # Arrange
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
        )
        loader = GroundTruthLoader(config)

        # Act
        annotations = loader.load()

        # Assert
        assert len(annotations) > 0
        from src.utils.enumerators import BBType
        assert all(bb.get_bb_type() == BBType.GROUND_TRUTH for bb in annotations)

    def test_load_yolo_without_images_dir_raises_error(self):
        # Arrange
        config = CLIConfig(
            gt_dir="/path/to/gts",
            det_dir="/path/to/dets",
            gt_format=GroundTruthFormat.YOLO,
            det_format=DetectionFormat.XYWH_ABS,
        )
        loader = GroundTruthLoader(config)

        # Act & Assert
        with pytest.raises(ConfigurationError, match="--images-dir"):
            loader.load()

    def test_load_yolo_without_classes_file_raises_error(self):
        # Arrange
        config = CLIConfig(
            gt_dir="/path/to/gts",
            det_dir="/path/to/dets",
            gt_format=GroundTruthFormat.YOLO,
            det_format=DetectionFormat.XYWH_ABS,
            images_dir="/path/to/images",
        )
        loader = GroundTruthLoader(config)

        # Act & Assert
        with pytest.raises(ConfigurationError, match="--gt-classes-file"):
            loader.load()

    def test_load_openimage_without_images_dir_raises_error(self):
        # Arrange
        config = CLIConfig(
            gt_dir="/path/to/gts",
            det_dir="/path/to/dets",
            gt_format=GroundTruthFormat.OPENIMAGE,
            det_format=DetectionFormat.XYWH_ABS,
        )
        loader = GroundTruthLoader(config)

        # Act & Assert
        with pytest.raises(ConfigurationError, match="--images-dir"):
            loader.load()


class TestDetectionLoader:
    """Test suite for DetectionLoader."""

    def test_load_xywh_abs_format(self):
        # Arrange
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
        )
        loader = DetectionLoader(config)

        # Act
        annotations = loader.load()

        # Assert
        assert len(annotations) > 0

    def test_load_yolo_rel_without_images_dir_raises_error(self):
        # Arrange
        config = CLIConfig(
            gt_dir="/path/to/gts",
            det_dir="/path/to/dets",
            gt_format=GroundTruthFormat.PASCALVOC,
            det_format=DetectionFormat.YOLO_REL,
        )
        loader = DetectionLoader(config)

        # Act & Assert
        with pytest.raises(ConfigurationError, match="--images-dir"):
            loader.load()


class TestMetricsEvaluator:
    """Test suite for MetricsEvaluator."""

    @pytest.fixture
    def sample_annotations(self):
        """Load sample annotations for testing."""
        import src.utils.converter as converter
        from src.utils.enumerators import BBType

        gts = converter.text2bb("tests/test_case_1/gts", BBType.GROUND_TRUTH)
        dets = converter.text2bb("tests/test_case_1/dets", BBType.DETECTED)
        return gts, dets

    def test_evaluate_pascal_metrics(self, sample_annotations):
        # Arrange
        gts, dets = sample_annotations
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            metric=MetricType.PASCAL,
            iou_threshold=0.5,
        )
        evaluator = MetricsEvaluator(config)

        # Act
        results = evaluator.evaluate(gts, dets)

        # Assert
        assert "pascal" in results
        assert "mAP" in results["pascal"]
        assert "per_class" in results["pascal"]
        assert results["pascal"]["iou_threshold"] == 0.5

    def test_evaluate_coco_metrics(self, sample_annotations):
        # Arrange
        gts, dets = sample_annotations
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            metric=MetricType.COCO,
        )
        evaluator = MetricsEvaluator(config)

        # Act
        results = evaluator.evaluate(gts, dets)

        # Assert
        assert "coco" in results
        assert "AP" in results["coco"]
        assert "AP50" in results["coco"]
        assert "AP75" in results["coco"]

    def test_evaluate_all_metrics(self, sample_annotations):
        # Arrange
        gts, dets = sample_annotations
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            metric=MetricType.ALL,
        )
        evaluator = MetricsEvaluator(config)

        # Act
        results = evaluator.evaluate(gts, dets)

        # Assert
        assert "pascal" in results
        assert "coco" in results

    def test_evaluate_with_eleven_point_interpolation(self, sample_annotations):
        # Arrange
        gts, dets = sample_annotations
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            metric=MetricType.PASCAL,
            ap_method=APInterpolationMethod.ELEVEN_POINT,
        )
        evaluator = MetricsEvaluator(config)

        # Act
        results = evaluator.evaluate(gts, dets)

        # Assert
        assert results["pascal"]["method"] == "eleven_point"


class TestCLIApplication:
    """Test suite for CLIApplication end-to-end tests."""

    def test_run_with_valid_config(self):
        # Arrange
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            metric=MetricType.PASCAL,
            quiet=True,
        )
        app = CLIApplication(config)

        # Act
        with patch("builtins.print"):
            exit_code = app.run()

        # Assert
        assert exit_code == 0

    def test_run_with_invalid_gt_dir_returns_error(self):
        # Arrange
        config = CLIConfig(
            gt_dir="/nonexistent/path",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            quiet=True,
        )
        app = CLIApplication(config)

        # Act
        with patch("builtins.print"):
            exit_code = app.run()

        # Assert
        assert exit_code == 1

    def test_run_with_invalid_det_dir_returns_error(self):
        # Arrange
        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="/nonexistent/path",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            quiet=True,
        )
        app = CLIApplication(config)

        # Act
        with patch("builtins.print"):
            exit_code = app.run()

        # Assert
        assert exit_code == 1

    def test_run_saves_results_to_output_dir(self):
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CLIConfig(
                gt_dir="tests/test_case_1/gts",
                det_dir="tests/test_case_1/dets",
                gt_format=GroundTruthFormat.TEXT_ABS,
                det_format=DetectionFormat.XYWH_ABS,
                output_dir=tmpdir,
                metric=MetricType.PASCAL,
                quiet=True,
            )
            app = CLIApplication(config)

            # Act
            with patch("builtins.print"):
                exit_code = app.run()

            # Assert
            assert exit_code == 0
            assert os.path.exists(os.path.join(tmpdir, "results.json"))


class TestMainFunction:
    """Test suite for the main entry point."""

    def test_main_with_valid_args(self):
        # Arrange
        args = [
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--metric", "pascal",
            "--quiet",
        ]

        # Act
        with patch("builtins.print"):
            exit_code = main(args)

        # Assert
        assert exit_code == 0

    def test_main_with_table_output(self):
        # Arrange
        args = [
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--output-format", "table",
            "--quiet",
        ]

        # Act
        with patch("builtins.print") as mock_print:
            exit_code = main(args)

        # Assert
        assert exit_code == 0


class TestIntegrationWithExistingTests:
    """Integration tests verifying CLI produces same results as existing evaluators."""

    def test_pascal_voc_metrics_match_existing_test_case_1(self):
        """Verify CLI produces same results as test_eval_pascal.py::test_case_1."""
        # Arrange
        import src.utils.converter as converter
        from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
        from src.utils.enumerators import BBType, MethodAveragePrecision

        gts = converter.text2bb("tests/test_case_1/gts", BBType.GROUND_TRUTH)
        dets = converter.text2bb("tests/test_case_1/dets", BBType.DETECTED)

        config = CLIConfig(
            gt_dir="tests/test_case_1/gts",
            det_dir="tests/test_case_1/dets",
            gt_format=GroundTruthFormat.TEXT_ABS,
            det_format=DetectionFormat.XYWH_ABS,
            metric=MetricType.PASCAL,
            iou_threshold=0.5,
            ap_method=APInterpolationMethod.ELEVEN_POINT,
        )
        evaluator = MetricsEvaluator(config)

        # Act
        cli_results = evaluator.evaluate(gts, dets)

        direct_results = get_pascalvoc_metrics(
            gts, dets,
            iou_threshold=0.5,
            method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
        )

        # Assert
        assert isclose(
            cli_results["pascal"]["mAP"],
            direct_results["mAP"],
            rel_tol=1e-6
        )

    def test_coco_metrics_match_existing_test(self):
        """Verify CLI produces same results as test_eval_coco.py."""
        # Arrange
        from src.utils.converter import coco2bb
        from src.evaluators.coco_evaluator import get_coco_summary
        from src.utils.enumerators import BBType

        gts = coco2bb("tests/test_coco_eval/gts", BBType.GROUND_TRUTH)
        dts = coco2bb("tests/test_coco_eval/dets", BBType.DETECTED)

        config = CLIConfig(
            gt_dir="tests/test_coco_eval/gts",
            det_dir="tests/test_coco_eval/dets",
            gt_format=GroundTruthFormat.COCO,
            det_format=DetectionFormat.COCO,
            metric=MetricType.COCO,
        )
        evaluator = MetricsEvaluator(config)

        # Act
        cli_results = evaluator.evaluate(gts, dts)
        direct_results = get_coco_summary(gts, dts)

        # Assert
        tol = 1e-6
        assert abs(cli_results["coco"]["AP"] - direct_results["AP"]) < tol
        assert abs(cli_results["coco"]["AP50"] - direct_results["AP50"]) < tol
        assert abs(cli_results["coco"]["AP75"] - direct_results["AP75"]) < tol


class TestEndToEndCLIWorkflow:
    """End-to-end integration tests for the complete CLI workflow."""

    def test_e2e_pascal_voc_evaluation_json_output(self):
        """Test complete workflow: Pascal VOC XML ground truth with text detections, JSON output."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--gt-dir", "toyexample/gts_vocpascal_format",
                "--det-dir", "toyexample/dets_classname_abs_xywh",
                "--gt-format", "pascalvoc",
                "--det-format", "xywh_abs",
                "--metric", "pascal",
                "--iou", "0.5",
                "--output-dir", tmpdir,
                "--output-format", "json",
                "--quiet",
            ]

            # Act
            exit_code = main(args)

            # Assert
            assert exit_code == 0
            results_file = os.path.join(tmpdir, "results.json")
            assert os.path.exists(results_file)

            with open(results_file) as f:
                results = json.load(f)

            assert "pascal" in results
            assert "mAP" in results["pascal"]
            assert results["pascal"]["mAP"] > 0
            assert "cat" in results["pascal"]["per_class"]

    def test_e2e_coco_evaluation_with_all_metrics(self):
        """Test complete workflow: COCO format with all metrics computed."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--gt-dir", "tests/test_coco_eval/gts",
                "--det-dir", "tests/test_coco_eval/dets",
                "--gt-format", "coco",
                "--det-format", "coco",
                "--metric", "all",
                "--output-dir", tmpdir,
                "--quiet",
            ]

            # Act
            exit_code = main(args)

            # Assert
            assert exit_code == 0
            results_file = os.path.join(tmpdir, "results.json")
            assert os.path.exists(results_file)

            with open(results_file) as f:
                results = json.load(f)

            assert "pascal" in results
            assert "coco" in results
            assert abs(results["coco"]["AP"] - 0.503647) < 1e-4
            assert abs(results["coco"]["AP50"] - 0.696973) < 1e-4

    def test_e2e_pascal_evaluation_with_plots(self):
        """Test complete workflow: Pascal VOC evaluation with precision-recall plots saved."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--gt-dir", "toyexample/gts_vocpascal_format",
                "--det-dir", "toyexample/dets_classname_abs_xywh",
                "--gt-format", "pascalvoc",
                "--det-format", "xywh_abs",
                "--metric", "pascal",
                "--output-dir", tmpdir,
                "--save-plots",
                "--quiet",
            ]

            # Act
            exit_code = main(args)

            # Assert
            assert exit_code == 0
            assert os.path.exists(os.path.join(tmpdir, "results.json"))
            assert os.path.exists(os.path.join(tmpdir, "all_classes.png"))
            assert os.path.exists(os.path.join(tmpdir, "cat.png"))

    def test_e2e_text_format_ground_truth_and_detections(self):
        """Test complete workflow: Text format for both ground truth and detections."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--gt-dir", "tests/test_case_1/gts",
                "--det-dir", "tests/test_case_1/dets",
                "--gt-format", "text_abs",
                "--det-format", "xywh_abs",
                "--metric", "pascal",
                "--iou", "0.5",
                "--ap-method", "eleven_point",
                "--output-dir", tmpdir,
                "--quiet",
            ]

            # Act
            exit_code = main(args)

            # Assert
            assert exit_code == 0
            results_file = os.path.join(tmpdir, "results.json")
            with open(results_file) as f:
                results = json.load(f)

            expected_ap = 0.0303030303
            assert isclose(results["pascal"]["per_class"]["object"]["AP"], expected_ap, rel_tol=1e-4)

    def test_e2e_table_output_format(self):
        """Test complete workflow with table output format to stdout."""
        # Arrange
        args = [
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--metric", "all",
            "--output-format", "table",
            "--quiet",
        ]

        # Act
        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            exit_code = main(args)

        output = captured_output.getvalue()

        # Assert
        assert exit_code == 0
        assert "PASCAL VOC Metrics" in output
        assert "COCO Metrics" in output
        assert "mAP @" in output
        assert "object" in output

    def test_e2e_multiple_iou_thresholds(self):
        """Test evaluation at different IoU thresholds produces expected results."""
        # Arrange
        iou_values = [0.1, 0.3, 0.5, 0.75]
        expected_aps = {
            0.1: 0.3333333333,
            0.3: 0.2683982683,
            0.5: 0.0303030303,
            0.75: 0.0
        }

        for iou in iou_values:
            with tempfile.TemporaryDirectory() as tmpdir:
                args = [
                    "--gt-dir", "tests/test_case_1/gts",
                    "--det-dir", "tests/test_case_1/dets",
                    "--gt-format", "text_abs",
                    "--det-format", "xywh_abs",
                    "--metric", "pascal",
                    "--iou", str(iou),
                    "--ap-method", "eleven_point",
                    "--output-dir", tmpdir,
                    "--quiet",
                ]

                # Act
                exit_code = main(args)

                # Assert
                assert exit_code == 0
                with open(os.path.join(tmpdir, "results.json")) as f:
                    results = json.load(f)

                actual_ap = results["pascal"]["per_class"]["object"]["AP"]
                assert isclose(actual_ap, expected_aps[iou], rel_tol=1e-6), \
                    f"IoU {iou}: expected {expected_aps[iou]}, got {actual_ap}"

    def test_e2e_invalid_directory_returns_error(self):
        """Test that invalid input directories return proper error code."""
        # Arrange
        args = [
            "--gt-dir", "/nonexistent/path/to/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--quiet",
        ]

        # Act
        import io
        from contextlib import redirect_stderr

        captured_stderr = io.StringIO()
        with redirect_stderr(captured_stderr):
            exit_code = main(args)

        # Assert
        assert exit_code == 1
        assert "not found" in captured_stderr.getvalue().lower()

    def test_e2e_empty_annotations_returns_error(self):
        """Test that empty annotation directories return proper error code."""
        # Arrange
        with tempfile.TemporaryDirectory() as empty_dir:
            args = [
                "--gt-dir", empty_dir,
                "--det-dir", "tests/test_case_1/dets",
                "--gt-format", "text_abs",
                "--det-format", "xywh_abs",
                "--quiet",
            ]

            # Act
            import io
            from contextlib import redirect_stderr

            captured_stderr = io.StringIO()
            with redirect_stderr(captured_stderr):
                exit_code = main(args)

            # Assert
            assert exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
