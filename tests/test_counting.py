import numpy as np

from spotcheck.counting import Counts, LotAggregator, agnostic_nms, count_detections, count_labels


def test_overlapping_boxes_of_different_classes_count_once():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    classes = np.array([0, 1, 0])
    scores = np.array([0.9, 0.6, 0.8])
    c = count_detections(boxes, classes, scores)
    assert (c.empty, c.occupied) == (2, 0)


def test_confidence_threshold():
    c = count_detections([[0, 0, 5, 5], [20, 20, 30, 30]], [0, 1], [0.1, 0.9], conf=0.25)
    assert (c.empty, c.occupied) == (0, 1)


def test_empty_input():
    assert count_detections([], [], []).total == 0


def test_nms_keeps_non_overlapping():
    boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)
    assert sorted(agnostic_nms(boxes, np.array([0.5, 0.6]))) == [0, 1]


def test_labels_and_aggregation():
    gt = count_labels([(0, .5, .5, .1, .1), (1, .2, .2, .1, .1), (1, .3, .3, .1, .1)])
    assert gt.as_dict()["occupancy_rate"] == round(2 / 3, 4)
    agg = LotAggregator()
    agg.add("lot_j", "s1", gt)
    agg.add("lot_j", "s1", Counts(1, 0))
    assert agg.table()["lot_j|s1"]["empty"] == 2
