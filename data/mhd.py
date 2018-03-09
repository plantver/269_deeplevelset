import SimpleITK as sitk
import numpy as np


def load_mhd(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    if isflip:
        numpyImage = numpyImage[:, ::-1, ::-1]
        print('flip!')

    numpyOrigin_zyx = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing_zyx = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin_zyx, numpySpacing_zyx