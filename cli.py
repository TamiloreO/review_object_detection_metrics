#!/usr/bin/env python
"""
Command Line Interface for Object Detection Metrics Evaluation.

This module provides a CLI alternative to the GUI for running object detection
evaluations in automated scripts and pipelines.

Usage Examples:
    # Pascal VOC evaluation with default settings
    python cli.py --gt-dir ./gts --det-dir ./dets --gt-format pascalvoc --det-format xywh_abs

    # COCO evaluation
    python cli.py --gt-dir ./gts --det-dir ./dets --gt-format coco --det-format coco --metric coco

    # Full evaluation with output
    python cli.py --gt-dir ./gts --det-dir ./dets --gt-format pascalvoc --det-format xywh_abs \
                  --output-dir ./results --metric pascal --iou 0.5 --save-plots
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.evaluators.coco_evaluator import get_coco_metrics, get_coco_summary
from src.evaluators.pascal_voc_evaluator import (
    get_pascalvoc_metrics,
    plot_precision_recall_curve,
    plot_precision_recall_curves,
)
from src.utils.enumerators import (
    BBFormat,
    BBType,
    CoordinatesType,
    MethodAveragePrecision,
)


class GroundTruthFormat(Enum):
    """Supported ground truth annotation formats."""
    COCO = "coco"
    CVAT = "cvat"
    OPENIMAGE = "openimage"
    LABELME = "labelme"
    PASCALVOC = "pascalvoc"
    IMAGENET = "imagenet"
    YOLO = "yolo"
    TEXT_ABS = "text_abs"


class DetectionFormat(Enum):
    """Supported detection annotation formats."""
    COCO = "coco"
    XYWH_ABS = "xywh_abs"
    XYX2Y2_ABS = "xyx2y2_abs"
    YOLO_REL = "yolo_rel"


class MetricType(Enum):
    """Supported evaluation metric types."""
    PASCAL = "pascal"
    COCO = "coco"
    ALL = "all"


class APInterpolationMethod(Enum):
    """Average Precision interpolation methods."""
    EVERY_POINT = "every_point"
    ELEVEN_POINT = "eleven_point"


@dataclass
class CLIConfig:
    """Configuration dataclass for CLI parameters."""
    gt_dir: str
    det_dir: str
    gt_format: GroundTruthFormat
    det_format: DetectionFormat
    output_dir: Optional[str] = None
    images_dir: Optional[str] = None
    gt_classes_file: Optional[str] = None
    det_classes_file: Optional[str] = None
    metric: MetricType = MetricType.ALL
    iou_threshold: float = 0.5
    ap_method: APInterpolationMethod = APInterpolationMethod.EVERY_POINT
    save_plots: bool = False
    quiet: bool = False
    output_format: str = "json"


class AnnotationLoadError(Exception):
    """Raised when annotations cannot be loaded."""
    pass


class ConfigurationError(Exception):
    """Raised when CLI configuration is invalid."""
    pass


class GroundTruthLoader:
    """Strategy pattern implementation for loading ground truth annotations."""

    def __init__(self, config: CLIConfig):
        self._config = config
        self._loaders = {
            GroundTruthFormat.COCO: self._load_coco,
            GroundTruthFormat.CVAT: self._load_cvat,
            GroundTruthFormat.OPENIMAGE: self._load_openimage,
            GroundTruthFormat.LABELME: self._load_labelme,
            GroundTruthFormat.PASCALVOC: self._load_pascalvoc,
            GroundTruthFormat.IMAGENET: self._load_imagenet,
            GroundTruthFormat.YOLO: self._load_yolo,
            GroundTruthFormat.TEXT_ABS: self._load_text_abs,
        }

    def load(self) -> List:
        loader = self._loaders.get(self._config.gt_format)
        if loader is None:
            raise AnnotationLoadError(
                f"Unsupported ground truth format: {self._config.gt_format}"
            )
        annotations = loader()
        for bb in annotations:
            bb.set_bb_type(BBType.GROUND_TRUTH)
        return annotations

    def _load_coco(self) -> List:
        return converter.coco2bb(self._config.gt_dir)

    def _load_cvat(self) -> List:
        return converter.cvat2bb(self._config.gt_dir)

    def _load_openimage(self) -> List:
        if not self._config.images_dir:
            raise ConfigurationError(
                "OpenImage format requires --images-dir parameter"
            )
        return converter.openimage2bb(
            self._config.gt_dir,
            self._config.images_dir,
            BBType.GROUND_TRUTH,
        )

    def _load_labelme(self) -> List:
        return converter.labelme2bb(self._config.gt_dir)

    def _load_pascalvoc(self) -> List:
        return converter.vocpascal2bb(self._config.gt_dir)

    def _load_imagenet(self) -> List:
        return converter.imagenet2bb(self._config.gt_dir)

    def _load_yolo(self) -> List:
        if not self._config.images_dir:
            raise ConfigurationError("YOLO format requires --images-dir parameter")
        if not self._config.gt_classes_file:
            raise ConfigurationError(
                "YOLO format requires --gt-classes-file parameter"
            )
        return converter.yolo2bb(
            self._config.gt_dir,
            self._config.images_dir,
            self._config.gt_classes_file,
            bb_type=BBType.GROUND_TRUTH,
        )

    def _load_text_abs(self) -> List:
        return converter.text2bb(
            self._config.gt_dir,
            bb_type=BBType.GROUND_TRUTH,
        )


class DetectionLoader:
    """Strategy pattern implementation for loading detection annotations."""

    def __init__(self, config: CLIConfig):
        self._config = config
        self._loaders = {
            DetectionFormat.COCO: self._load_coco,
            DetectionFormat.XYWH_ABS: self._load_xywh_abs,
            DetectionFormat.XYX2Y2_ABS: self._load_xyx2y2_abs,
            DetectionFormat.YOLO_REL: self._load_yolo_rel,
        }

    def load(self) -> List:
        loader = self._loaders.get(self._config.det_format)
        if loader is None:
            raise AnnotationLoadError(
                f"Unsupported detection format: {self._config.det_format}"
            )
        annotations = loader()
        if self._config.det_classes_file:
            annotations = general_utils.replace_id_with_classes(
                annotations, self._config.det_classes_file
            )
        return annotations

    def _load_coco(self) -> List:
        return converter.coco2bb(self._config.det_dir, bb_type=BBType.DETECTED)

    def _load_xywh_abs(self) -> List:
        return converter.text2bb(
            self._config.det_dir,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYWH,
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_dir=self._config.images_dir,
        )

    def _load_xyx2y2_abs(self) -> List:
        return converter.text2bb(
            self._config.det_dir,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYX2Y2,
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_dir=self._config.images_dir,
        )

    def _load_yolo_rel(self) -> List:
        if not self._config.images_dir:
            raise ConfigurationError(
                "YOLO relative format requires --images-dir parameter"
            )
        return converter.text2bb(
            self._config.det_dir,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.YOLO,
            type_coordinates=CoordinatesType.RELATIVE,
            img_dir=self._config.images_dir,
        )


class MetricsEvaluator:
    """Facade for computing evaluation metrics."""

    def __init__(self, config: CLIConfig):
        self._config = config

    def evaluate(
        self, gt_annotations: List, det_annotations: List
    ) -> Dict[str, Any]:
        results = {}

        if self._config.metric in (MetricType.PASCAL, MetricType.ALL):
            results["pascal"] = self._evaluate_pascal(gt_annotations, det_annotations)

        if self._config.metric in (MetricType.COCO, MetricType.ALL):
            results["coco"] = self._evaluate_coco(gt_annotations, det_annotations)

        return results

    def _evaluate_pascal(
        self, gt_annotations: List, det_annotations: List
    ) -> Dict[str, Any]:
        method = (
            MethodAveragePrecision.EVERY_POINT_INTERPOLATION
            if self._config.ap_method == APInterpolationMethod.EVERY_POINT
            else MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
        )

        results = get_pascalvoc_metrics(
            gt_annotations,
            det_annotations,
            iou_threshold=self._config.iou_threshold,
            method=method,
            generate_table=True,
        )

        per_class = {}
        for class_id, class_results in results["per_class"].items():
            per_class[class_id] = {
                "AP": float(class_results["AP"]),
                "total_positives": int(class_results["total positives"]),
                "total_TP": int(class_results["total TP"]),
                "total_FP": int(class_results["total FP"]),
                "iou_threshold": float(class_results["iou"]),
            }

        return {
            "mAP": float(results["mAP"]),
            "per_class": per_class,
            "iou_threshold": self._config.iou_threshold,
            "method": self._config.ap_method.value,
        }

    def _evaluate_coco(
        self, gt_annotations: List, det_annotations: List
    ) -> Dict[str, Any]:
        summary = get_coco_summary(gt_annotations, det_annotations)
        return {k: float(v) if v is not None else None for k, v in summary.items()}


class ResultsFormatter:
    """Strategy pattern for formatting output results."""

    @staticmethod
    def format(results: Dict[str, Any], output_format: str) -> str:
        if output_format == "json":
            return ResultsFormatter._format_json(results)
        elif output_format == "table":
            return ResultsFormatter._format_table(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    @staticmethod
    def _format_json(results: Dict[str, Any]) -> str:
        return json.dumps(results, indent=2)

    @staticmethod
    def _format_table(results: Dict[str, Any]) -> str:
        lines = []

        if "pascal" in results:
            lines.append("=" * 60)
            lines.append("PASCAL VOC Metrics")
            lines.append("=" * 60)
            pascal = results["pascal"]
            lines.append(f"mAP @ IoU={pascal['iou_threshold']}: {pascal['mAP']:.4f}")
            lines.append(f"Interpolation Method: {pascal['method']}")
            lines.append("-" * 60)
            lines.append(f"{'Class':<20} {'AP':<10} {'TP':<8} {'FP':<8} {'GT':<8}")
            lines.append("-" * 60)
            for class_id, metrics in pascal["per_class"].items():
                lines.append(
                    f"{class_id:<20} {metrics['AP']:<10.4f} "
                    f"{metrics['total_TP']:<8} {metrics['total_FP']:<8} "
                    f"{metrics['total_positives']:<8}"
                )
            lines.append("")

        if "coco" in results:
            lines.append("=" * 60)
            lines.append("COCO Metrics")
            lines.append("=" * 60)
            coco = results["coco"]
            lines.append(f"AP (IoU=0.50:0.95): {coco['AP']:.4f}")
            lines.append(f"AP50 (IoU=0.50):    {coco['AP50']:.4f}")
            lines.append(f"AP75 (IoU=0.75):    {coco['AP75']:.4f}")
            lines.append(f"AP (small):         {coco['APsmall']:.4f}")
            lines.append(f"AP (medium):        {coco['APmedium']:.4f}")
            lines.append(f"AP (large):         {coco['APlarge']:.4f}")
            lines.append("-" * 60)
            lines.append(f"AR (max=1):         {coco['AR1']:.4f}")
            lines.append(f"AR (max=10):        {coco['AR10']:.4f}")
            lines.append(f"AR (max=100):       {coco['AR100']:.4f}")
            lines.append(f"AR (small):         {coco['ARsmall']:.4f}")
            lines.append(f"AR (medium):        {coco['ARmedium']:.4f}")
            lines.append(f"AR (large):         {coco['ARlarge']:.4f}")

        return "\n".join(lines)


class PlotGenerator:
    """Generates and saves precision-recall plots."""

    def __init__(self, config: CLIConfig):
        self._config = config

    def generate(
        self, gt_annotations: List, det_annotations: List, results: Dict[str, Any]
    ) -> None:
        if not self._config.save_plots or not self._config.output_dir:
            return

        if "pascal" not in results:
            return

        method = (
            MethodAveragePrecision.EVERY_POINT_INTERPOLATION
            if self._config.ap_method == APInterpolationMethod.EVERY_POINT
            else MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
        )

        pascal_results = get_pascalvoc_metrics(
            gt_annotations,
            det_annotations,
            iou_threshold=self._config.iou_threshold,
            method=method,
            generate_table=False,
        )

        plot_precision_recall_curve(
            pascal_results["per_class"],
            mAP=pascal_results["mAP"],
            savePath=self._config.output_dir,
            showGraphic=False,
        )

        plot_precision_recall_curves(
            pascal_results["per_class"],
            showAP=True,
            savePath=self._config.output_dir,
            showGraphic=False,
        )


class CLIApplication:
    """Main CLI application orchestrator."""

    def __init__(self, config: CLIConfig):
        self._config = config
        self._gt_loader = GroundTruthLoader(config)
        self._det_loader = DetectionLoader(config)
        self._evaluator = MetricsEvaluator(config)
        self._plot_generator = PlotGenerator(config)

    def run(self) -> int:
        try:
            self._validate_config()
            gt_annotations = self._load_ground_truth()
            det_annotations = self._load_detections()
            results = self._evaluate(gt_annotations, det_annotations)
            self._generate_plots(gt_annotations, det_annotations, results)
            self._output_results(results)
            return 0
        except (AnnotationLoadError, ConfigurationError) as e:
            self._print_error(str(e))
            return 1
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
            return 1

    def _validate_config(self) -> None:
        if not os.path.isdir(self._config.gt_dir):
            raise ConfigurationError(
                f"Ground truth directory not found: {self._config.gt_dir}"
            )
        if not os.path.isdir(self._config.det_dir):
            raise ConfigurationError(
                f"Detection directory not found: {self._config.det_dir}"
            )
        if self._config.output_dir and not os.path.isdir(self._config.output_dir):
            raise ConfigurationError(
                f"Output directory not found: {self._config.output_dir}"
            )

    def _load_ground_truth(self) -> List:
        if not self._config.quiet:
            print(f"Loading ground truth annotations from: {self._config.gt_dir}")
        annotations = self._gt_loader.load()
        if not annotations:
            raise AnnotationLoadError(
                "No ground truth annotations found. Check format and directory."
            )
        if not self._config.quiet:
            print(f"  Loaded {len(annotations)} ground truth bounding boxes")
        return annotations

    def _load_detections(self) -> List:
        if not self._config.quiet:
            print(f"Loading detection annotations from: {self._config.det_dir}")
        annotations = self._det_loader.load()
        if not annotations:
            raise AnnotationLoadError(
                "No detection annotations found. Check format and directory."
            )
        if not self._config.quiet:
            print(f"  Loaded {len(annotations)} detection bounding boxes")
        return annotations

    def _evaluate(self, gt_annotations: List, det_annotations: List) -> Dict[str, Any]:
        if not self._config.quiet:
            print(f"Evaluating with metric: {self._config.metric.value}")
        return self._evaluator.evaluate(gt_annotations, det_annotations)

    def _generate_plots(
        self, gt_annotations: List, det_annotations: List, results: Dict[str, Any]
    ) -> None:
        if self._config.save_plots and self._config.output_dir:
            if not self._config.quiet:
                print(f"Saving plots to: {self._config.output_dir}")
            self._plot_generator.generate(gt_annotations, det_annotations, results)

    def _output_results(self, results: Dict[str, Any]) -> None:
        output = ResultsFormatter.format(results, self._config.output_format)
        print(output)

        if self._config.output_dir:
            output_file = os.path.join(self._config.output_dir, "results.json")
            with open(output_file, "w") as f:
                f.write(ResultsFormatter.format(results, "json"))
            if not self._config.quiet:
                print(f"Results saved to: {output_file}")

    def _print_error(self, message: str) -> None:
        print(f"Error: {message}", file=sys.stderr)


CLI_EPILOG = """\
Examples:
  # Pascal VOC evaluation
  %(prog)s --gt-dir ./gts --det-dir ./dets --gt-format pascalvoc --det-format xywh_abs

  # COCO evaluation with JSON output
  %(prog)s --gt-dir ./gts --det-dir ./dets --gt-format coco --det-format coco --metric coco

  # Full evaluation with plots
  %(prog)s --gt-dir ./gts --det-dir ./dets --gt-format pascalvoc --det-format xywh_abs \\
           --output-dir ./results --save-plots

Ground Truth Formats:
  coco       - COCO JSON format
  cvat       - CVAT XML format
  openimage  - OpenImage CSV format (requires --images-dir)
  labelme    - LabelMe JSON format
  pascalvoc  - Pascal VOC XML format
  imagenet   - ImageNet XML format
  yolo       - YOLO format (requires --images-dir and --gt-classes-file)
  text_abs   - Text format with absolute coordinates

Detection Formats:
  coco       - COCO JSON format
  xywh_abs   - Text format: <class> <confidence> <x> <y> <w> <h> (absolute)
  xyx2y2_abs - Text format: <class> <confidence> <x1> <y1> <x2> <y2> (absolute)
  yolo_rel   - YOLO format with relative coordinates (requires --images-dir)
"""


def create_argument_parser() -> argparse.ArgumentParser:
    """Factory function for creating the argument parser."""
    parser = argparse.ArgumentParser(
        description="Object Detection Metrics Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--gt-dir",
        required=True,
        help="Directory containing ground truth annotations",
    )
    required.add_argument(
        "--det-dir",
        required=True,
        help="Directory containing detection annotations",
    )
    required.add_argument(
        "--gt-format",
        required=True,
        choices=[f.value for f in GroundTruthFormat],
        help="Format of ground truth annotations",
    )
    required.add_argument(
        "--det-format",
        required=True,
        choices=[f.value for f in DetectionFormat],
        help="Format of detection annotations",
    )

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--output-dir",
        help="Directory to save results and plots",
    )
    optional.add_argument(
        "--images-dir",
        help="Directory containing images (required for some formats)",
    )
    optional.add_argument(
        "--gt-classes-file",
        help="Text file with class names for ground truth (one per line)",
    )
    optional.add_argument(
        "--det-classes-file",
        help="Text file with class names for detections (one per line)",
    )
    optional.add_argument(
        "--metric",
        choices=[m.value for m in MetricType],
        default=MetricType.ALL.value,
        help="Metric type to compute (default: all)",
    )
    optional.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for Pascal VOC metrics (default: 0.5)",
    )
    optional.add_argument(
        "--ap-method",
        choices=[m.value for m in APInterpolationMethod],
        default=APInterpolationMethod.EVERY_POINT.value,
        help="AP interpolation method (default: every_point)",
    )
    optional.add_argument(
        "--save-plots",
        action="store_true",
        help="Save precision-recall plots (requires --output-dir)",
    )
    optional.add_argument(
        "--output-format",
        choices=["json", "table"],
        default="json",
        help="Output format for results (default: json)",
    )
    optional.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser


def parse_arguments(args: Optional[List[str]] = None) -> CLIConfig:
    """Parse command line arguments and return configuration."""
    parser = create_argument_parser()
    parsed = parser.parse_args(args)

    return CLIConfig(
        gt_dir=parsed.gt_dir,
        det_dir=parsed.det_dir,
        gt_format=GroundTruthFormat(parsed.gt_format),
        det_format=DetectionFormat(parsed.det_format),
        output_dir=parsed.output_dir,
        images_dir=parsed.images_dir,
        gt_classes_file=parsed.gt_classes_file,
        det_classes_file=parsed.det_classes_file,
        metric=MetricType(parsed.metric),
        iou_threshold=parsed.iou,
        ap_method=APInterpolationMethod(parsed.ap_method),
        save_plots=parsed.save_plots,
        quiet=parsed.quiet,
        output_format=parsed.output_format,
    )


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    config = parse_arguments(args)
    app = CLIApplication(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
