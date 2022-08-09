import torch

def intersection_over_union(pred_boxes, label_boxes, format_boxes="midpoint"):
    """Calculates the intersection over union between the predicted
    box and the true boxes.

    Parameters
    ----------
    pred_boxes : tensor
        Predicted bounding box of the class of object. Has the shape of (N, 4)
        where N is the number of bounding boxes or batch_size and 4 is the
        number of corners of the box
    box_labels : tensor
        True bounding box of the class of object. Has the shape of (N, 4)
        where N is the number of bounding boxes or batch_size and 4 is the
        number of corners of the box
    format_boxes : str
        - midpoint : The midpoints of the boxes have been given and requires
        translation into corners coordinates (x,y,w,h)
        - corners : The corners coordinates have been given and requires no
        modification (x1,y1,x2,y2)

    Returns
    -------
    iou : tensor
        Intersection over union for all examples
    """
    if format_boxes == "midpoint":
        # Predicted boxes # (N, 1)
        # Shifting the coordinates to the left
        x1_box1 = pred_boxes[..., 0:1] - (pred_boxes[..., 2:3] / 2)
        y1_box1 = pred_boxes[..., 1:2] - (pred_boxes[..., 3:4] / 2)
        # Shifting the coordinates to the right
        x2_box1 = pred_boxes[..., 0:1] + (pred_boxes[..., 2:3] / 2)
        y2_box1 = pred_boxes[..., 1:2] + (pred_boxes[..., 3:4] / 2)

        # Labeled boxes # (N, 1)
        # Shifting the coordinates to the left
        x1_box2 = label_boxes[..., 0:1] - (label_boxes[..., 2:3] / 2)
        y1_box2 = label_boxes[..., 1:2] - (label_boxes[..., 3:4] / 2)
        # Shifting the coordinates to the right
        x2_box2 = label_boxes[..., 0:1] + (label_boxes[..., 2:3] / 2)
        y2_box2 = label_boxes[..., 1:2] + (label_boxes[..., 3:4] / 2)

    elif format_boxes == "corners":
        # Predicted boxes # (N, 1)
        x1_box1 = pred_boxes[..., 0:1]
        y1_box1 = pred_boxes[..., 1:2]
        x2_box1 = pred_boxes[..., 2:3]
        y2_box1 = pred_boxes[..., 3:4]

        # Labeled boxes # (N, 1)
        x1_box2 = label_boxes[..., 0:1]
        y1_box2 = label_boxes[..., 1:2]
        x2_box2 = label_boxes[..., 2:3]
        y2_box2 = label_boxes[..., 3:4]

    # Calculating the coordinates of the intersection
    x1 = torch.max(x1_box1, x1_box2)
    y1 = torch.max(y1_box1, y1_box2)

    x2 = torch.min(x2_box1, x2_box2)
    y2 = torch.min(y2_box1, y2_box2)

    # clamp(0) handles the cases where the boxes don't intersect
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    # Calculating the are of each of the boxes
    area_box1 = abs((x2_box1-x1_box1) * (y2_box1-y1_box1))
    area_box2 = abs((x2_box2-x1_box2) * (y2_box2-y1_box2))

    union = (area_box1 + area_box2) - intersection

    iou = intersection/(union + 1e-6)

    return iou
