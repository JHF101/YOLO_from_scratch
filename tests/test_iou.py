import sys
import torch

from utils.intersection_over_union import intersection_over_union

# Credit: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML_tests/Object_detection_tests/iou_test.py

class TestIntersectionOverUnion:

    # test cases we want to run
    t1_box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
    t1_box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
    t1_correct_iou = 1 / 7

    t2_box1 = torch.tensor([0.95, 0.6, 0.5, 0.2])
    t2_box2 = torch.tensor([0.95, 0.7, 0.3, 0.2])
    t2_correct_iou = 3 / 13

    t3_box1 = torch.tensor([0.25, 0.15, 0.3, 0.1])
    t3_box2 = torch.tensor([0.25, 0.35, 0.3, 0.1])
    t3_correct_iou = 0

    t4_box1 = torch.tensor([0.7, 0.95, 0.6, 0.1])
    t4_box2 = torch.tensor([0.5, 1.15, 0.4, 0.7])
    t4_correct_iou = 3 / 31

    t5_box1 = torch.tensor([0.5, 0.5, 0.2, 0.2])
    t5_box2 = torch.tensor([0.5, 0.5, 0.2, 0.2])
    t5_correct_iou = 1

    # (x1,y1,x2,y2) format
    t6_box1 = torch.tensor([2, 2, 6, 6])
    t6_box2 = torch.tensor([4, 4, 7, 8])
    t6_correct_iou = 4 / 24

    t7_box1 = torch.tensor([0, 0, 2, 2])
    t7_box2 = torch.tensor([3, 0, 5, 2])
    t7_correct_iou = 0

    t8_box1 = torch.tensor([0, 0, 2, 2])
    t8_box2 = torch.tensor([0, 3, 2, 5])
    t8_correct_iou = 0

    t9_box1 = torch.tensor([0, 0, 2, 2])
    t9_box2 = torch.tensor([2, 0, 5, 2])
    t9_correct_iou = 0

    t10_box1 = torch.tensor([0, 0, 2, 2])
    t10_box2 = torch.tensor([1, 1, 3, 3])
    t10_correct_iou = 1 / 7

    t11_box1 = torch.tensor([0, 0, 3, 2])
    t11_box2 = torch.tensor([1, 1, 3, 3])
    t11_correct_iou = 0.25

    t12_bboxes1 = torch.tensor(
        [
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 3, 2],
        ]
    )
    t12_bboxes2 = torch.tensor(
        [
            [3, 0, 5, 2],
            [3, 0, 5, 2],
            [0, 3, 2, 5],
            [2, 0, 5, 2],
            [1, 1, 3, 3],
            [1, 1, 3, 3],
        ]
    )
    t12_correct_ious = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])

    # Accept if the difference in iou is small
    epsilon = 0.001

    def test_both_inside_cell_shares_area(self):
        iou = intersection_over_union(self.t1_box1, self.t1_box2, format_boxes="midpoint")
        assert (torch.abs(iou - self.t1_correct_iou) < self.epsilon)

    def test_partially_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t2_box1, self.t2_box2, format_boxes="midpoint")
        assert (torch.abs(iou - self.t2_correct_iou) < self.epsilon)

    def test_both_inside_cell_shares_no_area(self):
        iou = intersection_over_union(self.t3_box1, self.t3_box2, format_boxes="midpoint")
        assert (torch.abs(iou - self.t3_correct_iou) < self.epsilon)

    def test_midpoint_outside_cell_shares_area(self):
        iou = intersection_over_union(self.t4_box1, self.t4_box2, format_boxes="midpoint")
        assert (torch.abs(iou - self.t4_correct_iou) < self.epsilon)

    def test_both_inside_cell_shares_entire_area(self):
        iou = intersection_over_union(self.t5_box1, self.t5_box2, format_boxes="midpoint")
        assert (torch.abs(iou - self.t5_correct_iou) < self.epsilon)

    def test_box_format_x1_y1_x2_y2(self):
        iou = intersection_over_union(self.t6_box1, self.t6_box2, format_boxes="corners")
        assert (torch.abs(iou - self.t6_correct_iou) < self.epsilon)

    def test_additional_and_batch(self):
        ious = intersection_over_union(
            self.t12_bboxes1, self.t12_bboxes2, format_boxes="corners"
        )
        all_true = torch.all(
            torch.abs(self.t12_correct_ious - ious.squeeze(1)) < self.epsilon
        )
        assert all_true

    def test_specific_corners_iou(self):
        # Non-edge case
        predicted=torch.tensor([50, 10, 250, 300])
        labels=torch.tensor([100, 5, 400, 250])

        """
        Intersection:
            i_x1 = max(50,100) = 100
            i_y1 = max(10, 5) = 10

            i_x2 = min(250, 400) = 250
            i_y2 = min(300, 250) = 250


        Area of predicted:
            x2-x1 = 250 - 50 = 200
            y2-y1 = 300 - 10 = 290
                Area1 = 200*290 = 58000

        Area of labels:
            x2-x1 = 400 - 100 = 300
            y2-y1 = 250 - 5 = 245
                Area2 = 300*245 = 73500

        intersection    = (i_x1 - i_x2)*(i_y1 - i_y2)
                        = (100-250)*(10-250)
                        = 36000

        iou = intersection/(Area1+Area2-intersection)
            = 36000/(58000+73500-36000)
            = 0.3769633508
        """

        result = intersection_over_union(
            pred_boxes=predicted,
            label_boxes=labels,
            format_boxes='corners'
        )
        result = round(result.item(), 4)

        assert result == round(0.3769633508,4)

    def test_non_intersection_iou(self):
        # Non-edge case
        predicted=torch.tensor([50, 10, 100, 300])
        labels=torch.tensor([105, 5, 400, 250])

        result = intersection_over_union(
            pred_boxes=predicted,
            label_boxes=labels,
            format_boxes='corners'
        )
        result = round(result.item(), 4)

        assert result == round(0, 4)
