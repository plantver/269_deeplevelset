# credit to https://github.com/pmneila/morphsnakes#id5

import numpy as np
from itertools import cycle
import numpy as np
import scipy
from scipy.ndimage import binary_dilation, binary_erosion


class Curvop:

    def SI(self, u):
        """SI operator."""
        if np.ndim(u) == 2:
            P = self._P2
        elif np.ndim(u) == 3:
            P = self._P3
        else:
            raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")

        if u.shape != self._aux.shape[1:]:
            self._aux = np.zeros((len(P),) + u.shape)

        for _aux_i, P_i in zip(self._aux, P):
            _aux_i[:] = binary_erosion(u, P_i)

        return self._aux.max(0)

    def IS(self, u):
        """IS operator."""
        if np.ndim(u) == 2:
            P = self._P2
        elif np.ndim(u) == 3:
            P = self._P3
        else:
            raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")

        if u.shape != self._aux.shape[1:]:
            self._aux = np.zeros((len(P),) + u.shape)

        for _aux_i, P_i in zip(self._aux, P):
            _aux_i[:] = binary_dilation(u, P_i)

        return self._aux.min(0)

    class Fcycle(object):

        def __init__(self, iterable):
            """Call functions from the iterable each time it is called."""
            self.funcs = cycle(iterable)

        def __call__(self, *args, **kwargs):
            f = next(self.funcs)
            return f(*args, **kwargs)

    def __init__(self):

        self._P2 = [np.eye(3), np.array([[0, 1, 0]] * 3), np.flipud(np.eye(3)), np.rot90([[0, 1, 0]] * 3)]
        self._P3 = [np.zeros((3, 3, 3)) for i in range(9)]

        self._P3[0][:, :, 1] = 1
        self._P3[1][:, 1, :] = 1
        self._P3[2][1, :, :] = 1
        self._P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
        self._P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
        self._P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
        self._P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
        self._P3[7][[0, 1, 2], [0, 1, 2], :] = 1
        self._P3[8][[0, 1, 2], [2, 1, 0], :] = 1

        self._aux = np.zeros((0))

    def get_curveop(self):
        SIoIS = lambda u: self.SI(self.IS(u))
        ISoSI = lambda u: self.IS(self.SI(u))
        return self.Fcycle([SIoIS, ISoSI])


class MSnake:

    def __init__(self,
                 mask,
                 img,
                 smoothing=3,
                 lambda1 = 1,
                 lambda2 = 1,
                 iterations = 50):
        """

        :param mask: 2D binary mask, inside == 1, outside == 0
        :param img: 2D image
        :param smoothing:
        :param lambda1:
        :param lambda2:
        :param iterations:
        """

        # currently works only for 2D
        assert len(mask.shape) == 2
        assert len(img.shape) == 2

        # cca, keep only largest component.
        labels, n = scipy.ndimage.measurements.label(mask)

        # largest CC
        # self.mask = np.zeros(mask.shape)
        # self.mask[labels == (np.bincount(labels.flat)[1:].argmax() + 1)] = 1

        # thresholded  CC
        cc_box_size = lambda t: (t[0].stop-t[0].start)*(t[1].stop-t[1].start)

        cc_slices = scipy.ndimage.measurements.find_objects(labels, max_label=99)
        self.mask = np.copy(mask)
        for i in range(n):
            t_s = cc_slices[i]
            if cc_box_size(t_s) < 100:
                self.mask[t_s] = 0

        self.init_mask = np.copy(self.mask)
        self.img = img
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.iterations = iterations

    def evolve(self):
        l_ev = list()
        curvop = Curvop().get_curveop()

        for i in range(self.iterations):
            inside = self.mask > 0
            outside = self.mask <= 0

            c0 = self.img[outside].sum() / float(outside.sum())
            c1 = self.img[inside].sum() / float(inside.sum())

            # Image attachment.
            dres = np.array(np.gradient(self.mask))

            abs_dres = np.abs(dres).sum(0)

            aux = abs_dres * (self.lambda1 * (self.img - c1) ** 2 - self.lambda2 * (self.img - c0) ** 2)

            self.mask[aux < 0] = 1
            self.mask[aux > 0] = 0

            for j in range(self.smoothing):
                self.mask = curvop(self.mask)

            l_ev.append(np.copy(self.mask))

        return l_ev
