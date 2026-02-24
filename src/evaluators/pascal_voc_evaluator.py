import os
import sys
from collections import Counter

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


def _group_boxes_by_class(gt_boxes, det_boxes):
    """Group ground truth and detection boxes by class.

    Parameters
    ----------
    gt_boxes : list
        List of ground truth BoundingBox objects.
    det_boxes : list
        List of detected BoundingBox objects.

    Returns
    -------
    tuple
        (classes_bbs, gt_classes_only) where classes_bbs is a dict mapping
        class_id to {'gt': [...], 'det': [...]}, and gt_classes_only is a
        list of unique class IDs present in ground truth.
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


def _match_detections_to_ground_truths(dects, gt_boxes, class_id, classes_bbs,
                                       iou_threshold, generate_table):
    """Match detections to ground truths and compute TP/FP arrays.

    Parameters
    ----------
    dects : list
        List of detection BoundingBox objects, sorted by confidence (descending).
    gt_boxes : list
        List of all ground truth BoundingBox objects.
    class_id : str
        The class being evaluated.
    classes_bbs : dict
        Dictionary mapping class_id to {'gt': [...], 'det': [...]}.
    iou_threshold : float
        IOU threshold for considering a detection as TP.
    generate_table : bool
        Whether to generate detailed table data.

    Returns
    -------
    tuple
        (TP, FP, dict_table) where TP and FP are numpy arrays indicating
        true/false positives for each detection, and dict_table contains
        per-detection details (or empty dict if generate_table is False).
    """
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))

    # Create dictionary with amount of expected detections for each image
    detected_gt_per_image = Counter([bb.get_image_name() for bb in gt_boxes])
    for key, val in detected_gt_per_image.items():
        detected_gt_per_image[key] = np.zeros(val)

    dict_table = {
        'image': [],
        'confidence': [],
        'TP': [],
        'FP': [],
        'acc TP': [],
        'acc FP': [],
        'precision': [],
        'recall': []
    } if generate_table else {}

    # Loop through detections
    for idx_det, det in enumerate(dects):
        img_det = det.get_image_name()

        if generate_table:
            dict_table['image'].append(img_det)
            dict_table['confidence'].append(f'{100*det.get_confidence():.2f}%')

        # Find ground truth boxes for this image and class
        gt = [g for g in classes_bbs[class_id]['gt'] if g.get_image_name() == img_det]

        # Get the maximum iou among all detections in the image
        iouMax = sys.float_info.min
        id_match_gt = -1

        # Given the detection det, find ground-truth with the highest iou
        for j, g in enumerate(gt):
            iou = BoundingBox.iou(det, g)
            if iou > iouMax:
                iouMax = iou
                id_match_gt = j

        # Assign detection as TP or FP
        if iouMax >= iou_threshold:
            # gt was not matched with any detection
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


def _compute_precision_recall(TP, FP, npos):
    """Compute precision and recall arrays from TP/FP counts.

    Parameters
    ----------
    TP : numpy.ndarray
        Array of true positive indicators (0 or 1) for each detection.
    FP : numpy.ndarray
        Array of false positive indicators (0 or 1) for each detection.
    npos : int
        Total number of ground truth positives.

    Returns
    -------
    tuple
        (rec, prec, acc_TP, acc_FP) where rec and prec are recall and
        precision arrays, and acc_TP/acc_FP are cumulative sums.
    """
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    return rec, prec, acc_TP, acc_FP


def _compute_average_precision(rec, prec, method):
    """Compute average precision using the specified interpolation method.

    Parameters
    ----------
    rec : numpy.ndarray
        Recall values.
    prec : numpy.ndarray
        Precision values.
    method : MethodAveragePrecision
        Interpolation method (EVERY_POINT or ELEVEN_POINT).

    Returns
    -------
    tuple
        (ap, mpre, mrec) where ap is the average precision, mpre is the
        interpolated precision, and mrec is the interpolated recall.
    """
    if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
        [ap, mpre, mrec, _] = calculate_ap_every_point(rec, prec)
    elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
        [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
    else:
        raise ValueError('method not defined')
    return ap, mpre, mrec


def _build_class_result(prec, rec, ap, mpre, mrec, npos, TP, FP,
                        method, iou_threshold, table):
    """Build the result dictionary for a single class.

    Parameters
    ----------
    prec : numpy.ndarray
        Precision values.
    rec : numpy.ndarray
        Recall values.
    ap : float
        Average precision.
    mpre : list
        Interpolated precision values.
    mrec : list
        Interpolated recall values.
    npos : int
        Total number of ground truth positives.
    TP : numpy.ndarray
        True positive indicators.
    FP : numpy.ndarray
        False positive indicators.
    method : MethodAveragePrecision
        Interpolation method used.
    iou_threshold : float
        IOU threshold used.
    table : pandas.DataFrame or None
        Detailed results table.

    Returns
    -------
    dict
        Dictionary containing all metrics for the class.
    """
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


def _build_results_table(dict_table, acc_TP, acc_FP, prec, rec):
    """Build a pandas DataFrame from the detection results table.

    Parameters
    ----------
    dict_table : dict
        Dictionary containing per-detection data.
    acc_TP : numpy.ndarray
        Cumulative true positives.
    acc_FP : numpy.ndarray
        Cumulative false positives.
    prec : numpy.ndarray
        Precision values.
    rec : numpy.ndarray
        Recall values.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing detailed results for each detection.
    """
    dict_table['acc TP'] = list(acc_TP)
    dict_table['acc FP'] = list(acc_FP)
    dict_table['precision'] = list(prec)
    dict_table['recall'] = list(rec)
    return pd.DataFrame(dict_table)


def _evaluate_class(class_id, classes_bbs, gt_boxes, gt_classes_only,
                    iou_threshold, method, generate_table):
    """Evaluate metrics for a single class.

    Parameters
    ----------
    class_id : str
        The class being evaluated.
    classes_bbs : dict
        Dictionary mapping class_id to {'gt': [...], 'det': [...]}.
    gt_boxes : list
        List of all ground truth BoundingBox objects.
    gt_classes_only : list
        List of class IDs present in ground truth.
    iou_threshold : float
        IOU threshold for considering a detection as TP.
    method : MethodAveragePrecision
        Interpolation method for AP calculation.
    generate_table : bool
        Whether to generate detailed table data.

    Returns
    -------
    dict or None
        Dictionary containing all metrics for the class, or None if the
        class is not in the ground truth.
    """
    # Report results only in the classes that are in the GT
    if class_id not in gt_classes_only:
        return None

    v = classes_bbs[class_id]
    npos = len(v['gt'])

    # Sort detections by decreasing confidence
    dects = [a for a in sorted(v['det'], key=lambda bb: bb.get_confidence(), reverse=True)]

    # Match detections to ground truths
    TP, FP, dict_table = _match_detections_to_ground_truths(
        dects, gt_boxes, class_id, classes_bbs, iou_threshold, generate_table)

    # Compute precision and recall
    rec, prec, acc_TP, acc_FP = _compute_precision_recall(TP, FP, npos)

    # Build results table if requested
    if generate_table:
        table = _build_results_table(dict_table, acc_TP, acc_FP, prec, rec)
    else:
        table = None

    # Compute average precision
    ap, mpre, mrec = _compute_average_precision(rec, prec, method)

    # Build and return class result
    return _build_class_result(prec, rec, ap, mpre, mrec, npos, TP, FP,
                               method, iou_threshold, table)


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
    ret = {}

    # Group boxes by class
    classes_bbs, gt_classes_only = _group_boxes_by_class(gt_boxes, det_boxes)

    # Evaluate each class
    for class_id in classes_bbs:
        result = _evaluate_class(class_id, classes_bbs, gt_boxes, gt_classes_only,
                                 iou_threshold, method, generate_table)
        if result is not None:
            ret[class_id] = result

    # Compute mAP (only for classes in GT)
    mAP = sum([v['AP'] for k, v in ret.items() if k in gt_classes_only]) / len(gt_classes_only)
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
