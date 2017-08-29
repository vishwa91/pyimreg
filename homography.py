from scipy import *
from scipy import linalg
from scipy import ndimage

def Haffine_from_points(fp,tp):
    """ find H, affine transformation, such that
        tp is affine transf of fp"""

    if fp.shape != tp.shape:
        raise RuntimeError

    #condition points
    #-from points-
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1))
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)

    #-to points-
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)

    #conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)

    #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    H = vstack((tmp2,[0,0,1]))

    #decondition
    H = dot(linalg.inv(C2),dot(H,C1))

    return H / H[2][2]

def affine_transform2(im, rot, shift):
    '''
        Perform affine transform for 2/3D images.
    '''
    if ndim(im) == 2:
        return ndimage.affine_transform(im, rot, shift)
    else:
        imr = ndimage.affine_transform(im[:, :, 0], rot, shift)
        img = ndimage.affine_transform(im[:, :, 1], rot, shift)
        imb = ndimage.affine_transform(im[:, :, 2], rot, shift)

        return dstack((imr, img, imb))
