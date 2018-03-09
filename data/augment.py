import tensorflow as tf
import numpy as np
from collections import defaultdict, Counter
from pprint import pprint
from scipy.linalg import expm, norm
from scipy.ndimage.interpolation import affine_transform


def affineRotAroundCenter(arr, theta, axis=[0,0,1]):
    """
    scipy based rotation about center of cube, CPU
    :param arr: 3D array to be rotated
    :param theta: radients
    :param axis: axis
    :return: ratated 3D array
    """
    def Mrot3d(axis, theta):
        axis = np.array(axis)
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def Mrot2d(theta):
        return np.array(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])

    dim = len(arr.shape)
    if dim == 2:
        M = Mrot2d(theta)
    elif dim == 3:
        M = Mrot3d(axis, theta)
    else:
        raise RuntimeError("Affine rotation cannot rotate array with shape %s"%(arr.shape,))
    center = 0.5 * (np.array(arr.shape) - np.ones((dim,)))
    return affine_transform(
        arr,
        M,
        mode="nearest",
        offset = center - M.dot(center)
    )


def rand_transpose_rotate_3D_Z(batch):
    """
    random transpose and rotation of 3D images
    :param batch:   a batch of 3D images in shape [N, Z, Y, X, C]
    :return:        a randomly transposed and rotated, in X Y plane, batch of 3D images
    """
    with tf.variable_scope("random_transpose_rotate_3D_Z"):
        batch = tf.cond(tf.random_uniform(()) > 0.5, lambda: tf.transpose(batch, perm=[0,1,3,2,4]), lambda: batch)
        return tf.map_fn(lambda x: tf.contrib.image.rotate(x, tf.random_uniform((), -1, 1) * 2 * np.pi,
                                                           interpolation="BILINEAR"),
                         batch)


def scaling(input, _min=None, _max=None):
    with tf.variable_scope("scaling"):
        if _min is None:
            _min = tf.reduce_min(input)
        if _max is None:
            _max = tf.reduce_max(input)
        return (tf.clip_by_value(input, clip_value_min=_min, clip_value_max=_max) - _min)/(_max - _min)


def balance_classes(dct_r, dct_h5, class_key):
    """

    :param dct_r: class value -> ratio
    :param h5: h5 with stucture, exampleId -> class key -> class value
    :return: list of resampled exampleIds
    """
    patchId_by_class = defaultdict(list)
    for k, v in dct_h5.items():
        patchId_by_class[v[class_key][()]].append(k)

    print("before balancing")
    for c, l in patchId_by_class.items():
        print("%s: %s"%(c, len(l)))

    max_c, max_l = max(map(lambda t: (t[0], len(t[1])), patchId_by_class.items()), key=lambda t: t[1])
    r_tot = sum(dct_r.values())
    l_tot = max_l * 1.0 * r_tot / dct_r[str(max_c)]

    """
    lst_patchId = np.concatenate(list(map(
        lambda t: t[1] + list(np.random.choice(
            t[1], int(len(t[1]) * ((1.0 * dct_r[str(t[0])] / r_tot) / (1.0 * len(t[1]) / l_tot) - 1))
        )),
        patchId_by_class.items()
    ))).tolist()
    """

    lst_patchId = list()
    for cls, pids in patchId_by_class.items():
        cls = str(cls)
        old_size = len(pids)
        new_size = int(old_size * ((1.0 * dct_r[cls] / r_tot) / (1.0 * old_size / l_tot) - 1))
        replace = old_size < new_size
        lst_patchId += pids
        lst_patchId += np.random.choice(pids, new_size, replace=replace).tolist()

    print("After balancing")
    count_by_class = Counter()
    for i in lst_patchId:
        count_by_class[dct_h5[i][class_key][()]] += 1
    pprint(count_by_class)

    return lst_patchId

