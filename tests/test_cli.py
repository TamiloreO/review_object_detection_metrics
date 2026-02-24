"""
Unit tests for the CLI module.

Tests follow the AAA pattern (Arrange, Act, Assert) and cover:
- Argument parsing
- Configuration validation
- Annotation loading
- Metrics evaluation
- Results formatting
- End-to-end integration (using subprocess)
"""

import json
import os
import subprocess
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


class TestEndToEndCLI:
    """End-to-end integration tests using subprocess to simulate real CLI usage."""

    @staticmethod
    def run_cli(args: list) -> subprocess.CompletedProcess:
        """Helper to run CLI as subprocess."""
        return subprocess.run(
            [sys.executable, "cli.py"] + args,
            capture_output=True,
            text=True,
        )

    def test_e2e_pascal_voc_text_format_json_output(self):
        """Full workflow: text annotations -> Pascal VOC metrics -> JSON output."""
        # Arrange
        args = [
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--metric", "pascal",
            "--iou", "0.5",
            "--output-format", "json",
            "--quiet",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        output_json = json.loads(result.stdout)
        assert "pascal" in output_json
        assert "mAP" in output_json["pascal"]
        assert "per_class" in output_json["pascal"]
        assert output_json["pascal"]["iou_threshold"] == 0.5
        assert 0.0 <= output_json["pascal"]["mAP"] <= 1.0

    def test_e2e_coco_format_full_metrics(self):
        """Full workflow: COCO annotations -> COCO metrics -> JSON output."""
        # Arrange
        args = [
            "--gt-dir", "tests/test_coco_eval/gts",
            "--det-dir", "tests/test_coco_eval/dets",
            "--gt-format", "coco",
            "--det-format", "coco",
            "--metric", "coco",
            "--output-format", "json",
            "--quiet",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        output_json = json.loads(result.stdout)
        assert "coco" in output_json
        coco_metrics = output_json["coco"]
        expected_keys = ["AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
                         "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge"]
        for key in expected_keys:
            assert key in coco_metrics

    def test_e2e_all_metrics_table_output(self):
        """Full workflow: compute all metrics with table output format."""
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
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        assert "PASCAL VOC Metrics" in result.stdout
        assert "COCO Metrics" in result.stdout
        assert "mAP @" in result.stdout
        assert "AP (IoU=0.50:0.95)" in result.stdout

    def test_e2e_with_output_directory_saves_files(self):
        """Full workflow: evaluation results saved to output directory."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--gt-dir", "tests/test_case_1/gts",
                "--det-dir", "tests/test_case_1/dets",
                "--gt-format", "text_abs",
                "--det-format", "xywh_abs",
                "--output-dir", tmpdir,
                "--metric", "pascal",
                "--quiet",
            ]

            # Act
            result = self.run_cli(args)

            # Assert
            assert result.returncode == 0
            results_file = os.path.join(tmpdir, "results.json")
            assert os.path.exists(results_file)

            with open(results_file) as f:
                saved_results = json.load(f)
            assert "pascal" in saved_results
            assert "mAP" in saved_results["pascal"]

    def test_e2e_with_plots_generation(self):
        """Full workflow: evaluation with precision-recall plots saved."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--gt-dir", "toyexample/gts_vocpascal_format",
                "--det-dir", "toyexample/dets_classname_abs_xywh",
                "--gt-format", "pascalvoc",
                "--det-format", "xywh_abs",
                "--output-dir", tmpdir,
                "--metric", "pascal",
                "--save-plots",
                "--quiet",
            ]

            # Act
            result = self.run_cli(args)

            # Assert
            assert result.returncode == 0
            assert os.path.exists(os.path.join(tmpdir, "results.json"))
            assert os.path.exists(os.path.join(tmpdir, "all_classes.png"))
            assert os.path.exists(os.path.join(tmpdir, "cat.png"))

    def test_e2e_different_iou_thresholds(self):
        """Full workflow: verify different IoU thresholds produce different results."""
        # Arrange
        results_by_iou = {}

        for iou in ["0.1", "0.5", "0.75"]:
            args = [
                "--gt-dir", "tests/test_case_1/gts",
                "--det-dir", "tests/test_case_1/dets",
                "--gt-format", "text_abs",
                "--det-format", "xywh_abs",
                "--metric", "pascal",
                "--iou", iou,
                "--quiet",
            ]

            # Act
            result = self.run_cli(args)

            # Assert
            assert result.returncode == 0
            results_by_iou[iou] = json.loads(result.stdout)["pascal"]["mAP"]

        # Assert: mAP should generally decrease as IoU threshold increases
        assert results_by_iou["0.1"] >= results_by_iou["0.5"]
        assert results_by_iou["0.5"] >= results_by_iou["0.75"]

    def test_e2e_eleven_point_vs_every_point_interpolation(self):
        """Full workflow: compare 11-point and every-point AP interpolation methods."""
        # Arrange
        results_by_method = {}

        for method in ["every_point", "eleven_point"]:
            args = [
                "--gt-dir", "tests/test_case_1/gts",
                "--det-dir", "tests/test_case_1/dets",
                "--gt-format", "text_abs",
                "--det-format", "xywh_abs",
                "--metric", "pascal",
                "--ap-method", method,
                "--quiet",
            ]

            # Act
            result = self.run_cli(args)

            # Assert
            assert result.returncode == 0
            results_by_method[method] = json.loads(result.stdout)

        # Assert: both methods should produce valid results
        assert results_by_method["every_point"]["pascal"]["method"] == "every_point"
        assert results_by_method["eleven_point"]["pascal"]["method"] == "eleven_point"
        assert 0.0 <= results_by_method["every_point"]["pascal"]["mAP"] <= 1.0
        assert 0.0 <= results_by_method["eleven_point"]["pascal"]["mAP"] <= 1.0

    def test_e2e_pascalvoc_xml_ground_truth(self):
        """Full workflow: Pascal VOC XML ground truth format."""
        # Arrange
        args = [
            "--gt-dir", "toyexample/gts_vocpascal_format",
            "--det-dir", "toyexample/dets_classname_abs_xywh",
            "--gt-format", "pascalvoc",
            "--det-format", "xywh_abs",
            "--metric", "pascal",
            "--quiet",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        output_json = json.loads(result.stdout)
        assert output_json["pascal"]["mAP"] > 0.8

    def test_e2e_invalid_ground_truth_directory_fails(self):
        """Full workflow: graceful failure on invalid ground truth directory."""
        # Arrange
        args = [
            "--gt-dir", "/nonexistent/path/to/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--quiet",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_e2e_invalid_detection_directory_fails(self):
        """Full workflow: graceful failure on invalid detection directory."""
        # Arrange
        args = [
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "/nonexistent/path/to/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--quiet",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_e2e_progress_messages_shown_without_quiet(self):
        """Full workflow: progress messages displayed when not in quiet mode."""
        # Arrange
        args = [
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_abs",
            "--det-format", "xywh_abs",
            "--metric", "pascal",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        assert "Loading ground truth" in result.stdout
        assert "Loading detection" in result.stdout
        assert "Evaluating" in result.stdout

    def test_e2e_help_displays_usage(self):
        """Full workflow: --help displays usage information."""
        # Arrange
        args = ["--help"]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        assert "Object Detection Metrics Evaluation CLI" in result.stdout
        assert "--gt-dir" in result.stdout
        assert "--det-dir" in result.stdout
        assert "Examples:" in result.stdout

    def test_e2e_missing_required_args_shows_error(self):
        """Full workflow: missing required arguments shows error."""
        # Arrange
        args = ["--gt-dir", "tests/test_case_1/gts"]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

    def test_e2e_coco_metrics_values_match_expected(self):
        """Full workflow: COCO metrics match expected values from test_eval_coco.py."""
        # Arrange
        args = [
            "--gt-dir", "tests/test_coco_eval/gts",
            "--det-dir", "tests/test_coco_eval/dets",
            "--gt-format", "coco",
            "--det-format", "coco",
            "--metric", "coco",
            "--quiet",
        ]

        # Act
        result = self.run_cli(args)

        # Assert
        assert result.returncode == 0
        output_json = json.loads(result.stdout)
        coco = output_json["coco"]

        tol = 1e-5
        assert abs(coco["AP"] - 0.503647) < tol
        assert abs(coco["AP50"] - 0.696973) < tol
        assert abs(coco["AP75"] - 0.571667) < tol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
