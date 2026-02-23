"""Tests for the command-line interface."""

import json
import os
import subprocess
import sys
import tempfile
from math import isclose


def run_cli(*args):
    """Run the CLI and return the result."""
    cmd = [sys.executable, "cli.py"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def test_cli_help():
    """Test that help is displayed correctly."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "Evaluate object detection metrics" in result.stdout
    assert "--gt-dir" in result.stdout
    assert "--det-dir" in result.stdout


def test_cli_missing_required_args():
    """Test error handling for missing required arguments."""
    result = run_cli()
    assert result.returncode != 0
    assert "required" in result.stderr.lower()


def test_cli_invalid_gt_dir():
    """Test error handling for non-existent ground truth directory."""
    result = run_cli("--gt-dir", "nonexistent", "--det-dir", "tests/test_case_1/dets")
    assert result.returncode != 0
    assert "Ground truth directory not found" in result.stderr


def test_cli_invalid_det_dir():
    """Test error handling for non-existent detection directory."""
    result = run_cli("--gt-dir", "tests/test_case_1/gts", "--det-dir", "nonexistent")
    assert result.returncode != 0
    assert "Detection directory not found" in result.stderr


def test_cli_pascalvoc_every_point():
    """Test Pascal VOC metrics with every-point interpolation."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "text_xywh",
        "--det-format", "text_xywh",
        "--metric", "pascalvoc",
        "--ap-method", "every_point",
        "--iou-threshold", "0.5",
        "-q"
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isclose(data["mAP"], 0.0222222222, rel_tol=1e-6)
    assert isclose(data["per_class"]["object"]["AP"], 0.0222222222, rel_tol=1e-6)


def test_cli_pascalvoc_eleven_point():
    """Test Pascal VOC metrics with eleven-point interpolation."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "text_xywh",
        "--det-format", "text_xywh",
        "--metric", "pascalvoc",
        "--ap-method", "eleven_point",
        "--iou-threshold", "0.5",
        "-q"
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isclose(data["mAP"], 0.0303030303, rel_tol=1e-6)


def test_cli_pascalvoc_different_iou():
    """Test Pascal VOC metrics with different IOU thresholds."""
    expected_aps = {0.1: 0.3371980676, 0.3: 0.2456866804, 0.75: 0.0}

    for iou, expected_ap in expected_aps.items():
        result = run_cli(
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_xywh",
            "--det-format", "text_xywh",
            "--metric", "pascalvoc",
            "--iou-threshold", str(iou),
            "-q"
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isclose(data["mAP"], expected_ap, rel_tol=1e-6), \
            f"Failed for IOU {iou}: expected {expected_ap}, got {data['mAP']}"


def test_cli_coco_metrics():
    """Test COCO metrics computation."""
    result = run_cli(
        "--gt-dir", "tests/test_coco_eval/gts",
        "--det-dir", "tests/test_coco_eval/dets",
        "--gt-format", "coco",
        "--det-format", "coco",
        "--metric", "coco",
        "-q"
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)

    # Compare to expected values from test_eval_coco.py
    tol = 1e-6
    assert abs(data["AP"] - 0.503647) < tol
    assert abs(data["AP50"] - 0.696973) < tol
    assert abs(data["AP75"] - 0.571667) < tol
    assert abs(data["APsmall"] - 0.593252) < tol
    assert abs(data["APmedium"] - 0.557991) < tol
    assert abs(data["APlarge"] - 0.489363) < tol
    assert abs(data["AR1"] - 0.386813) < tol
    assert abs(data["AR10"] - 0.593680) < tol
    assert abs(data["AR100"] - 0.595353) < tol


def test_cli_coco_selected_metrics():
    """Test COCO metrics with specific metrics selected."""
    result = run_cli(
        "--gt-dir", "tests/test_coco_eval/gts",
        "--det-dir", "tests/test_coco_eval/dets",
        "--gt-format", "coco",
        "--det-format", "coco",
        "--metric", "coco",
        "--coco-metrics", "AP", "AP50",
        "-q"
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)

    # Only AP and AP50 should be present
    assert "AP" in data
    assert "AP50" in data
    assert "AP75" not in data
    assert "AR1" not in data


def test_cli_output_file():
    """Test JSON output to file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        result = run_cli(
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_xywh",
            "--det-format", "text_xywh",
            "--metric", "pascalvoc",
            "-o", output_path
        )
        assert result.returncode == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = json.load(f)
        assert "mAP" in data
        assert "per_class" in data
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_cli_save_plots():
    """Test saving precision-recall plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cli(
            "--gt-dir", "tests/test_case_1/gts",
            "--det-dir", "tests/test_case_1/dets",
            "--gt-format", "text_xywh",
            "--det-format", "text_xywh",
            "--metric", "pascalvoc",
            "--output-dir", tmpdir,
            "--save-plots"
        )
        assert result.returncode == 0
        assert os.path.exists(os.path.join(tmpdir, "all_classes.png"))
        assert os.path.exists(os.path.join(tmpdir, "object.png"))


def test_cli_save_plots_requires_output_dir():
    """Test that --save-plots requires --output-dir."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "text_xywh",
        "--det-format", "text_xywh",
        "--metric", "pascalvoc",
        "--save-plots"
    )
    assert result.returncode != 0
    assert "--output-dir is required when using --save-plots" in result.stderr


def test_cli_quiet_mode():
    """Test quiet mode suppresses progress output."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "text_xywh",
        "--det-format", "text_xywh",
        "--metric", "pascalvoc",
        "-q"
    )
    assert result.returncode == 0
    # In quiet mode, stderr should be empty
    assert "Loading" not in result.stderr
    # stdout should contain valid JSON
    json.loads(result.stdout)


def test_cli_verbose_mode():
    """Test verbose mode shows formatted output."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "text_xywh",
        "--det-format", "text_xywh",
        "--metric", "pascalvoc"
    )
    assert result.returncode == 0
    # stderr should contain progress messages
    assert "Loading ground truth" in result.stderr
    assert "Loading detection" in result.stderr
    # stdout should contain formatted text, not JSON
    assert "Pascal VOC Metrics:" in result.stdout


def test_cli_yolo_format_requires_images_dir():
    """Test that YOLO format requires images directory."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "yolo",
        "--det-format", "text_xywh"
    )
    assert result.returncode != 0
    assert "--images-dir is required for yolo format" in result.stderr


def test_cli_yolo_format_requires_classes_file():
    """Test that YOLO format requires classes file."""
    result = run_cli(
        "--gt-dir", "tests/test_case_1/gts",
        "--det-dir", "tests/test_case_1/dets",
        "--gt-format", "yolo",
        "--det-format", "text_xywh",
        "--images-dir", "toyexample/images"
    )
    assert result.returncode != 0
    assert "--gt-classes-file is required for yolo format" in result.stderr


def test_cli_pascalvoc_format():
    """Test Pascal VOC XML format for ground truth."""
    result = run_cli(
        "--gt-dir", "toyexample/gts_vocpascal_format",
        "--det-dir", "toyexample/dets_classname_abs_xywh",
        "--gt-format", "pascalvoc",
        "--det-format", "text_xywh",
        "--metric", "pascalvoc",
        "-q"
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "cat" in data["per_class"]
    assert data["per_class"]["cat"]["AP"] > 0.8  # Should have high AP
