import unittest
from acvl_utils.instance_segmentation.instance_matching import match_instances_dice
import numpy as np


class TestInstanceMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.pred_instances = np.array(
            [
                [9, 0, 1, 1, 0],
                [9, 0, 1, 1, 0],
                [2, 2, 0, 0, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 3, 0, 0],
            ], dtype=np.uint8
        )
        self.gt_instances = np.array(
            [
                [0, 4, 4, 6, 7],
                [0, 0, 4, 6, 7],
                [2, 2, 0, 0, 0],
                [0, 3, 3, 0, 0],
                [0, 0, 0, 0, 0],
            ], dtype=np.uint8
        )

    def test_default(self):
        matches = match_instances_dice(self.gt_instances, self.pred_instances)
        expected_output = [(2, 2, 1.0, 2, 2),
                           (6, 1, 2 / 3, 2, 4),
                           (3, 3, 0.4, 2, 3),
                           (4, None, 0, 3, 0),
                           (7, None, 0, 2, 0),
                           (None, 9, 0, 0, 2)]
        for m, e in zip(matches, expected_output):
            self.assertTrue(m == e)

    def test_higher_dice_cutoff(self):
        matches = match_instances_dice(self.gt_instances, self.pred_instances, dice_cutoff=0.5)
        expected_output = [(2, 2, 1.0, 2, 2),
                           (6, 1, 2 / 3, 2, 4),
                           (3, None, 0, 2, 0),
                           (4, None, 0, 3, 0),
                           (7, None, 0, 2, 0),
                           (None, 3, 0, 0, 3),
                           (None, 9, 0, 0, 2)]

        for m, e in zip(matches, expected_output):
            self.assertTrue(m == e)

    def test_assert_if_wrong_dtype(self):
        self.assertRaises(AssertionError, match_instances_dice, self.gt_instances.astype(np.int16), self.pred_instances)
        self.assertRaises(AssertionError, match_instances_dice, self.gt_instances, self.pred_instances.astype(np.int16))
        self.assertRaises(AssertionError, match_instances_dice, self.gt_instances.astype(np.float32), self.pred_instances)
        self.assertRaises(AssertionError, match_instances_dice, self.gt_instances, self.pred_instances.astype(bool))

    def test_no_consume(self):
        matches = match_instances_dice(self.gt_instances, self.pred_instances, consume_instances=False)
        expected_output = [(2, 2, 1.0, 2, 2),
                           (6, 1, 2 / 3, 2, 4),
                           (4, 1, 4 / (4 + 1 + 2), 3, 4),
                           (3, 3, 0.4, 2, 3),
                           (7, None, 0, 2, 0),
                           (None, 9, 0, 0, 2)]
        for m, e in zip(matches, expected_output):
            self.assertTrue(m == e)