def highest_confidence_box(boxes):
    """
    given an enumerable of boxes, each an array in form [ xmin, ymin, xmax, ymax, confidence]
    return the box with the highest confidence
    """

    # print('in highest_confidence_box boxes.shape={}'.format(boxes[1].shape))
    i, b = max(enumerate(boxes), key=lambda e: e[1][4])

    # print('selected max at index {} and value {}'.format(i, b))

    return i, b
