import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.bounding_box import BoundingBox
from src.utils.enumerators import (BBFormat, CoordinatesType,
                                   MethodAveragePrecision)


def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


def group_boxes_by_class(
    gt_boxes: List[BoundingBox],
    det_boxes: List[BoundingBox]
) -> Tuple[Dict[str, Dict[str, List[BoundingBox]]], List[str]]:
    """Group ground truth and detection boxes by class.
    
    Returns:
        Tuple of (classes_bbs dict, list of GT-only class IDs)
    """
    gt_classes_only = []
    classes_bbs = {}
    
    for bb in gt_boxes:
        c = bb.get_class_id()
        gt_classes_only.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    
    for bb in det_boxes:
        c = bb.get_class_id()
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)
    
    return classes_bbs, gt_classes_only


def match_detections_to_ground_truth(
    detections: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float,
    generate_table: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, List]]]:
    """Match detections to ground truths and compute TP/FP arrays.
    
    Returns:
        Tuple of (TP array, FP array, optional table dict for DataFrame)
    """
    # Sort detections by decreasing confidence
    dects = sorted(detections, key=lambda bb: bb.get_confidence(), reverse=True)
    
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))
    
    # Create dictionary with amount of expected detections for each image
    detected_gt_per_image = Counter([bb.get_image_name() for bb in ground_truths])
    for key, val in detected_gt_per_image.items():
        detected_gt_per_image[key] = np.zeros(val)
    
    dict_table = None
    if generate_table:
        dict_table = {
            'image': [],
            'confidence': [],
            'TP': [],
            'FP': [],
            'acc TP': [],
            'acc FP': [],
            'precision': [],
            'recall': []
        }
    
    for idx_det, det in enumerate(dects):
        img_det = det.get_image_name()
        
        if generate_table:
            dict_table['image'].append(img_det)
            dict_table['confidence'].append(f'{100*det.get_confidence():.2f}%')
        
        # Find ground truth boxes for this image
        gt = [g for g in ground_truths if g.get_image_name() == img_det]
        
        # Get the maximum IoU among all ground truths in the image
        iouMax = sys.float_info.min
        id_match_gt = -1
        
        for j, g in enumerate(gt):
            iou = BoundingBox.iou(det, g)
            if iou > iouMax:
                iouMax = iou
                id_match_gt = j
        
        # Assign detection as TP or FP
        if iouMax >= iou_threshold and id_match_gt >= 0:
            if detected_gt_per_image[img_det][id_match_gt] == 0:
                TP[idx_det] = 1
                detected_gt_per_image[img_det][id_match_gt] = 1
                if generate_table:
                    dict_table['TP'].append(1)
                    dict_table['FP'].append(0)
            else:
                FP[idx_det] = 1
                if generate_table:
                    dict_table['FP'].append(1)
                    dict_table['TP'].append(0)
        else:
            FP[idx_det] = 1
            if generate_table:
                dict_table['FP'].append(1)
                dict_table['TP'].append(0)
    
    return TP, FP, dict_table


def compute_precision_recall(
    TP: np.ndarray,
    FP: np.ndarray,
    npos: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision and recall from TP/FP arrays.
    
    Returns:
        Tuple of (precision, recall, accumulated TP, accumulated FP)
    """
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    return prec, rec, acc_TP, acc_FP


def compute_average_precision(
    rec: np.ndarray,
    prec: np.ndarray,
    method: MethodAveragePrecision
) -> Tuple[float, Any, Any, Any]:
    """Compute average precision using the specified interpolation method.
    
    Returns:
        Tuple of (AP, interpolated precision, interpolated recall, indices)
    """
    if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
        return calculate_ap_every_point(rec, prec)
    elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
        return calculate_ap_11_point_interp(rec, prec)
    else:
        raise ValueError(f'Unknown AP calculation method: {method}')


def build_results_table(
    dict_table: Dict[str, List],
    acc_TP: np.ndarray,
    acc_FP: np.ndarray,
    prec: np.ndarray,
    rec: np.ndarray
) -> pd.DataFrame:
    """Build a pandas DataFrame from detection results."""
    dict_table['acc TP'] = list(acc_TP)
    dict_table['acc FP'] = list(acc_FP)
    dict_table['precision'] = list(prec)
    dict_table['recall'] = list(rec)
    return pd.DataFrame(dict_table)


def evaluate_class(
    class_gt_boxes: List[BoundingBox],
    class_det_boxes: List[BoundingBox],
    iou_threshold: float,
    method: MethodAveragePrecision,
    generate_table: bool = False
) -> Dict[str, Any]:
    """Evaluate metrics for a single class.
    
    Returns:
        Dictionary containing precision, recall, AP, and other metrics for the class.
    """
    npos = len(class_gt_boxes)
    
    TP, FP, dict_table = match_detections_to_ground_truth(
        class_det_boxes,
        class_gt_boxes,
        iou_threshold,
        generate_table
    )
    
    prec, rec, acc_TP, acc_FP = compute_precision_recall(TP, FP, npos)
    
    table = None
    if generate_table and dict_table is not None:
        table = build_results_table(dict_table, acc_TP, acc_FP, prec, rec)
    
    ap, mpre, mrec, ii = compute_average_precision(rec, prec, method)
    
    return {
        'precision': prec,
        'recall': rec,
        'AP': ap,
        'interpolated precision': mpre,
        'interpolated recall': mrec,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP),
        'method': method,
        'iou': iou_threshold,
        'table': table
    }


def compute_map(
    per_class_results: Dict[str, Dict[str, Any]],
    gt_classes_only: List[str]
) -> float:
    """Compute mean Average Precision across all GT classes."""
    return sum([v['AP'] for k, v in per_class_results.items() if k in gt_classes_only]) / len(gt_classes_only)


def get_pascalvoc_metrics(gt_boxes,
                          det_boxes,
                          iou_threshold=0.5,
                          method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                          generate_table=False):
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Args:
        boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP
        (dget_pascalvoc_metricsns:
        A dictioanry contains information and metrics of each class.
        The key represents the class and the values are:
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Positive detections;"""
    
    classes_bbs, gt_classes_only = group_boxes_by_class(gt_boxes, det_boxes)
    
    ret = {}
    for c, v in classes_bbs.items():
        # Report results only for classes present in GT
        if c not in gt_classes_only:
            continue
        
        ret[c] = evaluate_class(
            v['gt'],
            v['det'],
            iou_threshold,
            method,
            generate_table
        )
    
    mAP = compute_map(ret, gt_classes_only)
    return {'per_class': ret, 'mAP': mAP}


def plot_precision_recall_curve(results,
                                mAP=None,
                                showInterpolatedPrecision=False,
                                savePath=None,
                                showGraphic=True):
    result = None
    plt.close()
    # Each resut represents a class
    for classId, result in results.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        method = result['method']
        if showInterpolatedPrecision:
            if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
            elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                # Remove duplicates, getting only the highest precision of each recall value
                nrec = []
                nprec = []
                for idx in range(len(mrec)):
                    r = mrec[idx]
                    if r not in nrec:
                        idxEq = np.argwhere(mrec == r)
                        nrec.append(r)
                        nprec.append(max([mpre[int(id)] for id in idxEq]))
                plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
        plt.plot(recall, precision, label=f'{classId}')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    if mAP:
        map_str = "{0:.2f}%".format(mAP * 100)
        plt.title(f'Precision x Recall curve, mAP={map_str}')
    else:
        plt.title('Precision x Recall curve')
    plt.legend(shadow=True)
    plt.grid()
    if savePath is not None:
        plt.savefig(os.path.join(savePath, 'all_classes.png'))
    if showGraphic is True:
        plt.show()
        # plt.waitforbuttonpress()
        plt.pause(0.05)
    return results


def plot_precision_recall_curves(results,
                                 showAP=False,
                                 showInterpolatedPrecision=False,
                                 savePath=None,
                                 showGraphic=True):
    result = None
    # Each resut represents a class
    for classId, result in results.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        method = result['method']
        plt.close()
        if showInterpolatedPrecision:
            if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
            elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                # Remove duplicates, getting only the highest precision of each recall value
                nrec = []
                nprec = []
                for idx in range(len(mrec)):
                    r = mrec[idx]
                    if r not in nrec:
                        idxEq = np.argwhere(mrec == r)
                        nrec.append(r)
                        nprec.append(max([mpre[int(id)] for id in idxEq]))
                plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
        plt.plot(recall, precision, label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        if showAP:
            ap_str = "{0:.2f}%".format(average_precision * 100)
            # ap_str = "{0:.4f}%".format(average_precision * 100)
            plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
        else:
            plt.title('Precision x Recall curve \nClass: %s' % str(classId))
        plt.legend(shadow=True)
        plt.grid()
        ############################################################
        # Uncomment the following block to create plot with points #
        ############################################################
        # plt.plot(recall, precision, 'bo')
        # labels = ['R', 'Y', 'J', 'A', 'U', 'C', 'M', 'F', 'D', 'B', 'H', 'P', 'E', 'X', 'N', 'T',
        # 'K', 'Q', 'V', 'I', 'L', 'S', 'G', 'O']
        # dicPosition = {}
        # dicPosition['left_zero'] = (-30,0)
        # dicPosition['left_zero_slight'] = (-30,-10)
        # dicPosition['right_zero'] = (30,0)
        # dicPosition['left_up'] = (-30,20)
        # dicPosition['left_down'] = (-30,-25)
        # dicPosition['right_up'] = (20,20)
        # dicPosition['right_down'] = (20,-20)
        # dicPosition['up_zero'] = (0,30)
        # dicPosition['up_right'] = (0,30)
        # dicPosition['left_zero_long'] = (-60,-2)
        # dicPosition['down_zero'] = (-2,-30)
        # vecPositions = [
        #     dicPosition['left_down'],
        #     dicPosition['left_zero'],
        #     dicPosition['right_zero'],
        #     dicPosition['right_zero'],  #'R', 'Y', 'J', 'A',
        #     dicPosition['left_up'],
        #     dicPosition['left_up'],
        #     dicPosition['right_up'],
        #     dicPosition['left_up'],  # 'U', 'C', 'M', 'F',
        #     dicPosition['left_zero'],
        #     dicPosition['right_up'],
        #     dicPosition['right_down'],
        #     dicPosition['down_zero'],  #'D', 'B', 'H', 'P'
        #     dicPosition['left_up'],
        #     dicPosition['up_zero'],
        #     dicPosition['right_up'],
        #     dicPosition['left_up'],  # 'E', 'X', 'N', 'T',
        #     dicPosition['left_zero'],
        #     dicPosition['right_zero'],
        #     dicPosition['left_zero_long'],
        #     dicPosition['left_zero_slight'],  # 'K', 'Q', 'V', 'I',
        #     dicPosition['right_down'],
        #     dicPosition['left_down'],
        #     dicPosition['right_up'],
        #     dicPosition['down_zero']
        # ]  # 'L', 'S', 'G', 'O'
        # for idx in range(len(labels)):
        #     box = dict(boxstyle='round,pad=.5',facecolor='yellow',alpha=0.5)
        #     plt.annotate(labels[idx],
        #                 xy=(recall[idx],precision[idx]), xycoords='data',
        #                 xytext=vecPositions[idx], textcoords='offset points',
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        #                 bbox=box)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if savePath is not None:
            plt.savefig(os.path.join(savePath, classId + '.png'))
        if showGraphic is True:
            plt.show()
            # plt.waitforbuttonpress()
            plt.pause(0.05)
    return results
