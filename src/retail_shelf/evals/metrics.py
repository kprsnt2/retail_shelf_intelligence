"""Evaluation Metrics for Retail Shelf Intelligence.

Computes IoU, mAP@0.5, Out-of-Stock (OOS) Void Recall, SKU Classification Accuracy,
and Price Tag Discrepancy F1.
"""
from typing import List, Dict, Any, Tuple
from retail_shelf.models.shelf import BoundingBox

def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x_left = max(box1.x1, box2.x1)
    y_top = max(box1.y1, box2.y1)
    x_right = min(box1.x2, box2.x2)
    y_bottom = min(box1.y2, box2.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1.area
    box2_area = box2.area
    union_area = float(box1_area + box2_area - intersection_area)

    return intersection_area / union_area if union_area > 0 else 0.0

def evaluate_facing_detections(
    predictions: List[BoundingBox],
    ground_truths: List[BoundingBox],
    iou_threshold: float = 0.50
) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1 for facing bounding boxes."""
    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

    false_negatives = len(ground_truths) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives
    }

def evaluate_oos_void_recall(
    pred_voids: List[BoundingBox],
    gt_voids: List[BoundingBox],
    iou_threshold: float = 0.40
) -> Dict[str, float]:
    """Calculate Out-Of-Stock Void Recall (Retail Safety Metric)."""
    return evaluate_facing_detections(pred_voids, gt_voids, iou_threshold=iou_threshold)
