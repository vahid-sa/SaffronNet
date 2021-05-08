import numpy as np


def generate_anchors(angle_split, num_variables):
    """
    Generate (reference) anchors 
    """
    num_anchors = angle_split

    # initialize output anchors
    anchors = np.zeros((num_anchors, num_variables))

    b = (360 / angle_split)
    anchors[:, -1] = [0+b*n for n in range(angle_split)]
    print('base anchors: ', anchors)
    return anchors


def anchors_for_shape(
    image_shape,
    angle_split=None,
    num_variables=None,
    stride=None
):

    all_anchors = np.zeros((0, num_variables))
    anchors = generate_anchors(
        angle_split=angle_split, num_variables=num_variables)
    shifted_anchors = shift(image_shape, stride, anchors)
    all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    print('shape: ', shape)
    print('stride: ', stride)
    print('anchors: ', anchors)
    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1] // stride) + 0.5) * stride
    shift_y = (np.arange(0, shape[0] // stride) + 0.5) * stride
    print('shift_x: ', shift_x)
    print('shift_y: ', shift_y)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    print('shift_x: ', shift_x)
    print('shift_y: ', shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        np.zeros_like(shift_y).ravel()
    )).transpose()
    print('shifts: ', shifts)

    # add A anchors (1, A, 3) to
    # cell K shifts (K, 1, 3) to get
    # shift anchors (K, A, 3)
    # reshape to (K*A, 3) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 3)) +
                   shifts.reshape((1, K, 3)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 3))
    print('all_anchors: ', all_anchors)
    return all_anchors
