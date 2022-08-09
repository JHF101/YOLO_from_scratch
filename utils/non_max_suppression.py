import torch

from .intersection_over_union import intersection_over_union

def non_max_suppression(
        bboxes,
        iou_threshold,
        threshold, # Probability threshold
        format_boxes="corners"):
    """
    Cleans up the multiple bounding boxes that could result from the algorithm.

    Takes out the highest scoring box per class and calculates the IOU between
    itself and the other boxes and gets rid of boxes that have an IOU above
    a certain threshold.

    Parameters
    ----------
    bboxes : list[list[]]
        The result has the following output structure:
            [ [ <class>, <probability>, <x_1>, <y_1>, <x_2>, <y_2> ], ....]

        where the class is the predicted class of the type of object that
        has been identified. The probability is the certainty of the algorithm
        that the object is of that particular class. x_1, y_1, x_2, y_2 are the
        bounding box coordinates.
    iou_threshold : float
        A user defined threshold which determines which bounding boxes will be
        removed.


    Returns
    -------

    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
                if box[0] != chosen_box[0]
                or intersection_over_union(
                    pred_boxes=torch.tensor(chosen_box[2:]),
                    label_boxes=torch.tensor(box[2:]),
                    format_boxes=format_boxes,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

    # # Ensuring that a list is received
    # assert type(bboxes) == list

    # remaining_bboxes = []

    # # Looping through all of the bounding boxes
    # for box in bboxes:
    #     # Only keeping ones with a high enough probability
    #     if box[1] > threshold:
    #         remaining_bboxes.append(box)

    # # Sorting the bboxes left over in descending order of their probability
    # remaining_bboxes = sorted(remaining_bboxes, key=lambda x:[1], reverse=True)

    # # Bounding boxes after non-max suppression
    # bboxes_after_nms = []

    # # Looping through the bounding boxes
    # while remaining_bboxes:
    #     chosen_box = remaining_bboxes.pop(0)

    #     clean_bbox = []
    #     for box in remaining_bboxes:

    #         # If they are not of the same class, not going to compare them
    #         condition_one = box[0] != chosen_box[0]

    #         # Calculating the intersection over union and checking threshold
    #         condition_two = intersection_over_union(
    #                 pred_boxes=torch.tensor(chosen_box[2:]),
    #                 label_boxes=torch.tensor(box[2:]),
    #                 format_boxes=format_boxes) < iou_threshold

    #         if condition_one or condition_two:
    #             clean_bbox.append(box)

    #     # Replacing the remaining bboxes after they have been filtered
    #     remaining_bboxes = clean_bbox

    #     bboxes_after_nms.append(chosen_box)

    # return bboxes_after_nms

