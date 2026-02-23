#!/usr/bin/env python
"""
Command-line interface for object detection metrics evaluation.

This CLI provides access to the same evaluation functionality as the GUI,
enabling automated evaluation in scripts and CI/CD pipelines.

Usage examples:
    # Basic COCO evaluation
    python cli.py --gt-dir ./gts --det-dir ./dets --gt-format coco --det-format coco

    # Pascal VOC evaluation with custom IOU threshold
    python cli.py --gt-dir ./gts --det-dir ./dets --gt-format pascalvoc --det-format text_xywh \
                  --metric pascalvoc --iou-threshold 0.75

    # Save results to file
    python cli.py --gt-dir ./gts --det-dir ./dets --output results.json
"""

import argparse
import json
import os
import sys

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


GT_FORMATS = [
    "coco",
    "cvat",
    "openimage",
    "labelme",
    "pascalvoc",
    "imagenet",
    "yolo",
    "text_xywh",
    "text_xyx2y2",
]

DET_FORMATS = [
    "coco",
    "text_xywh",
    "text_xyx2y2",
    "text_yolo_rel",
]

METRICS = ["coco", "pascalvoc"]

AP_METHODS = ["every_point", "eleven_point"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate object detection metrics from ground truth and detection annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # COCO metrics with COCO format annotations
  python cli.py --gt-dir gts/ --det-dir dets/ --gt-format coco --det-format coco

  # Pascal VOC metrics with text format annotations
  python cli.py --gt-dir gts/ --det-dir dets/ --gt-format text_xywh --det-format text_xywh \\
                --metric pascalvoc --iou-threshold 0.5

  # Save PR curves and results
  python cli.py --gt-dir gts/ --det-dir dets/ --output-dir results/ --save-plots
        """,
    )

    # Required arguments
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Directory containing ground truth annotations",
    )
    parser.add_argument(
        "--det-dir",
        required=True,
        help="Directory containing detection annotations",
    )

    # Format arguments
    parser.add_argument(
        "--gt-format",
        choices=GT_FORMATS,
        default="coco",
        help=f"Ground truth annotation format. Choices: {GT_FORMATS} (default: coco)",
    )
    parser.add_argument(
        "--det-format",
        choices=DET_FORMATS,
        default="coco",
        help=f"Detection annotation format. Choices: {DET_FORMATS} (default: coco)",
    )

    # Optional directories for specific formats
    parser.add_argument(
        "--images-dir",
        help="Directory containing images (required for yolo, openimage, and relative coordinate formats)",
    )
    parser.add_argument(
        "--gt-classes-file",
        help="Text file with class names (one per line), required for yolo ground truth format",
    )
    parser.add_argument(
        "--det-classes-file",
        help="Text file with class names (one per line), required for text formats with class IDs",
    )

    # Metric selection
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="coco",
        help=f"Metric type to compute. Choices: {METRICS} (default: coco)",
    )

    # COCO metric options
    parser.add_argument(
        "--coco-metrics",
        nargs="+",
        choices=[
            "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
            "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge",
        ],
        help="Specific COCO metrics to compute (default: all)",
    )

    # Pascal VOC options
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IOU threshold for Pascal VOC metrics (default: 0.5)",
    )
    parser.add_argument(
        "--ap-method",
        choices=AP_METHODS,
        default="every_point",
        help=f"AP interpolation method for Pascal VOC. Choices: {AP_METHODS} (default: every_point)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results (JSON format). If not specified, prints to stdout",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files including plots",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save precision-recall plots (requires --output-dir)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def load_ground_truths(args):
    """Load ground truth annotations based on the specified format."""
    gt_dir = args.gt_dir
    gt_format = args.gt_format
    images_dir = args.images_dir
    classes_file = args.gt_classes_file

    if gt_format == "coco":
        return converter.coco2bb(gt_dir, bb_type=BBType.GROUND_TRUTH)
    elif gt_format == "cvat":
        return converter.cvat2bb(gt_dir)
    elif gt_format == "openimage":
        if not images_dir:
            raise ValueError("--images-dir is required for openimage format")
        return converter.openimage2bb(gt_dir, images_dir, bb_type=BBType.GROUND_TRUTH)
    elif gt_format == "labelme":
        return converter.labelme2bb(gt_dir)
    elif gt_format == "pascalvoc":
        return converter.vocpascal2bb(gt_dir)
    elif gt_format == "imagenet":
        return converter.imagenet2bb(gt_dir)
    elif gt_format == "yolo":
        if not images_dir:
            raise ValueError("--images-dir is required for yolo format")
        if not classes_file:
            raise ValueError("--gt-classes-file is required for yolo format")
        return converter.yolo2bb(gt_dir, images_dir, classes_file, bb_type=BBType.GROUND_TRUTH)
    elif gt_format == "text_xywh":
        return converter.text2bb(
            gt_dir,
            bb_type=BBType.GROUND_TRUTH,
            bb_format=BBFormat.XYWH,
            type_coordinates=CoordinatesType.ABSOLUTE,
        )
    elif gt_format == "text_xyx2y2":
        return converter.text2bb(
            gt_dir,
            bb_type=BBType.GROUND_TRUTH,
            bb_format=BBFormat.XYX2Y2,
            type_coordinates=CoordinatesType.ABSOLUTE,
        )
    else:
        raise ValueError(f"Unknown ground truth format: {gt_format}")


def load_detections(args):
    """Load detection annotations based on the specified format."""
    det_dir = args.det_dir
    det_format = args.det_format
    images_dir = args.images_dir
    classes_file = args.det_classes_file

    if det_format == "coco":
        return converter.coco2bb(det_dir, bb_type=BBType.DETECTED)
    elif det_format == "text_xywh":
        dets = converter.text2bb(
            det_dir,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYWH,
            type_coordinates=CoordinatesType.ABSOLUTE,
        )
    elif det_format == "text_xyx2y2":
        dets = converter.text2bb(
            det_dir,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYX2Y2,
            type_coordinates=CoordinatesType.ABSOLUTE,
        )
    elif det_format == "text_yolo_rel":
        if not images_dir:
            raise ValueError("--images-dir is required for text_yolo_rel format")
        dets = converter.text2bb(
            det_dir,
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.YOLO,
            type_coordinates=CoordinatesType.RELATIVE,
            img_dir=images_dir,
        )
    else:
        raise ValueError(f"Unknown detection format: {det_format}")

    # Replace class IDs with names if classes file is provided
    if classes_file and det_format in ["text_xywh", "text_xyx2y2", "text_yolo_rel"]:
        dets = general_utils.replace_id_with_classes(dets, classes_file)

    return dets


def compute_coco_metrics(gt_bbs, det_bbs, selected_metrics=None):
    """Compute COCO metrics."""
    results = get_coco_summary(gt_bbs, det_bbs)

    if selected_metrics:
        results = {k: v for k, v in results.items() if k in selected_metrics}

    return results


def compute_pascalvoc_metrics(gt_bbs, det_bbs, iou_threshold, ap_method):
    """Compute Pascal VOC metrics."""
    if ap_method == "every_point":
        method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION
    else:
        method = MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION

    results = get_pascalvoc_metrics(
        gt_bbs,
        det_bbs,
        iou_threshold=iou_threshold,
        method=method,
        generate_table=False,
    )

    # Convert to serializable format
    output = {
        "mAP": results["mAP"],
        "iou_threshold": iou_threshold,
        "ap_method": ap_method,
        "per_class": {},
    }

    for class_id, class_results in results["per_class"].items():
        output["per_class"][class_id] = {
            "AP": class_results["AP"],
            "total_positives": class_results["total positives"],
            "total_TP": int(class_results["total TP"]),
            "total_FP": int(class_results["total FP"]),
        }

    return output, results


def save_plots(results, output_dir, mAP):
    """Save precision-recall plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save combined plot
    plot_precision_recall_curve(
        results["per_class"],
        mAP=mAP,
        savePath=output_dir,
        showGraphic=False,
    )

    # Save individual class plots
    plot_precision_recall_curves(
        results["per_class"],
        showAP=True,
        savePath=output_dir,
        showGraphic=False,
    )


def format_results_text(results, metric_type):
    """Format results for console output."""
    lines = []

    if metric_type == "coco":
        lines.append("COCO Metrics:")
        lines.append("-" * 40)
        for metric, value in results.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.6f}")
            else:
                lines.append(f"  {metric}: {value}")
    else:
        lines.append("Pascal VOC Metrics:")
        lines.append("-" * 40)
        lines.append(f"  IOU Threshold: {results['iou_threshold']}")
        lines.append(f"  AP Method: {results['ap_method']}")
        lines.append(f"  mAP: {results['mAP']:.6f}")
        lines.append("")
        lines.append("  Per-class AP:")
        for class_id, class_results in results["per_class"].items():
            lines.append(f"    {class_id}: {class_results['AP']:.6f} "
                        f"(TP={class_results['total_TP']}, FP={class_results['total_FP']}, "
                        f"GT={class_results['total_positives']})")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Validate arguments
    if not os.path.isdir(args.gt_dir):
        print(f"Error: Ground truth directory not found: {args.gt_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.det_dir):
        print(f"Error: Detection directory not found: {args.det_dir}", file=sys.stderr)
        sys.exit(1)

    if args.save_plots and not args.output_dir:
        print("Error: --output-dir is required when using --save-plots", file=sys.stderr)
        sys.exit(1)

    # Load annotations
    if not args.quiet:
        print("Loading ground truth annotations...", file=sys.stderr)

    try:
        gt_bbs = load_ground_truths(args)
    except ValueError as e:
        print(f"Error loading ground truths: {e}", file=sys.stderr)
        sys.exit(1)

    if not gt_bbs:
        print("Error: No ground truth annotations found", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"  Loaded {len(gt_bbs)} ground truth bounding boxes", file=sys.stderr)
        print("Loading detection annotations...", file=sys.stderr)

    try:
        det_bbs = load_detections(args)
    except ValueError as e:
        print(f"Error loading detections: {e}", file=sys.stderr)
        sys.exit(1)

    if not det_bbs:
        print("Error: No detection annotations found", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"  Loaded {len(det_bbs)} detection bounding boxes", file=sys.stderr)
        print("Computing metrics...", file=sys.stderr)

    # Compute metrics
    if args.metric == "coco":
        results = compute_coco_metrics(gt_bbs, det_bbs, args.coco_metrics)
    else:
        results, full_results = compute_pascalvoc_metrics(
            gt_bbs, det_bbs, args.iou_threshold, args.ap_method
        )

        # Save plots if requested
        if args.save_plots:
            if not args.quiet:
                print(f"Saving plots to {args.output_dir}...", file=sys.stderr)
            save_plots(full_results, args.output_dir, full_results["mAP"])

    # Output results
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        if args.quiet:
            # In quiet mode, just output JSON
            print(json.dumps(results, indent=2))
        else:
            # In verbose mode, output formatted text
            # Flush stderr first to ensure proper ordering
            sys.stderr.flush()
            print()
            print(format_results_text(results, args.metric))


if __name__ == "__main__":
    main()
