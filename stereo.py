'''
README
To process an image pair:
1. Put stereo.py and two images (left.png and right.png) in the same directory.
2. From a terminal, activate the CS6476 virtual environment: ~$ conda activate CS6476
3. From the stereo.py directory, run stereo.py file: ~$ python3 stereo.py
4. You will be asked to enter ndisp value.
5. You will see outputs "Processing SSD left to right", "Processing SSD right to left" and "Processing DP", indicating which method is
being processed. When you see "All images completed", all the 3 outputs will be saved to the same directory.
'''

import numpy as np
import cv2


def ssd_disparity(left, right, left2right = 0, k_size = 3, ndisp = 100):
    """
    Calculate disparity map using SSD method.

    Parameters
    ----------
    left: left view image.
    right: right view image.
    left2right: 0 if calculate from left view to right view, 1 if calculate from right view to left view.
    k_size: SSD window size.
    ndisp: bound of possible disparity values.

    Returns
    -------
    disparity: non-normalized disparity map.
    """

    height, width = left.shape[0], right.shape[1]
    kernel = (np.ones((k_size,k_size),dtype = np.float64))/9
    ssds = []
    for i in range(1,ndisp):
        shift = np.zeros(left.shape,dtype = np.float64)
        if left2right:
            shift[:,:width-i] = right[:,i:]
            diff = left-shift
        else:
            shift[:,i:] = left[:,:width-i]
            diff = right - shift
        square_diff = diff*diff
        ssd = cv2.filter2D(square_diff,-1,kernel,borderType = cv2.BORDER_REPLICATE)
        ssds.append(ssd)
    disparity = np.zeros(left.shape,dtype = np.float64)
    for i in range(0,height):
        for j in range(0,width):
            smallest_diff = 0
            index = 0
            for k in range(len(ssds)):
                if smallest_diff == 0:
                    smallest_diff = ssds[k][i,j]
                elif smallest_diff > ssds[k][i,j]:
                    smallest_diff = ssds[k][i,j]
                    index = k
            disparity[i,j] = index + 1
    return disparity

def M_map(left,right,ndisp):
    """
    Calculate M matrix, term m(p,d) in equation (4).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    ndisp: bound of possible disparity values.

    Returns
    -------
    map: M matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for i in range(0, height):
        for j in range(0, width):
            for d in range(1, ndisp+1):
                if j + d < width:
                    b = right[i,j+d]
                else:
                    b = 0
                map[i,j,d] = abs(left[i,j] - b)
    return map

def MV_map(M,V,a):
    """
    Calculate MV matrix, term m'(p,d) in equation (7).

    Parameters
    ----------
    M: M matrix.
    V: V matrix.
    a: alpha weight.

    Returns
    -------
    map: MV matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for i in range(0, height):
        for j in range(0, width):
            for d in range(1, ndisp+1):
                map[i,j,d] = M[i,j,d] + a*(V[i,j,d]-np.min(V[i,j,1:]))
    return map




def F_map(left, right, P1, P2, P3, T, ndisp, MV):
    """
    Calculate forward horizontal energy matrix F, term F(p,d) in equation (5).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.
    MV: MV matrix.

    Returns
    -------
    map: F matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for i in range(0, height):
        for j in range(0, width):
            for d in range(1, ndisp+1):
                m = MV[i,j,d]
                if j == 0:
                    map[i,j,d] = m
                else:
                    if abs(left[i,j] - left[i,j-1]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = m + min(map[i,j-1,d],map[i,j-1,max(d-1,1)]+P1,map[i,j-1,min(d+1,ndisp)]+P1,np.min(map[i,j-1,1:])+P4)

    return map

def B_map(left, right, P1, P2, P3, T, ndisp, MV):
    """
    Calculate backward horizontal energy matrix B, term B(p,d) in equation (5).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.
    MV: MV matrix.

    Returns
    -------
    map: B matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for i in range(0, height):
        for j in range(width-1, -1, -1):
            for d in range(1, ndisp+1):
                m = MV[i,j,d]
                if j == width-1:
                    map[i,j,d] = m
                else:
                    if abs(left[i,j] - left[i,j+1]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = m + min(map[i,j+1,d],map[i,j+1,max(d-1,1)]+P1,map[i,j+1,min(d+1,ndisp)]+P1,np.min(map[i,j+1,1:])+P4)

    return map

def C_map(F, B, MV):
    """
    Calculate total horizontal energy matrix C, term C(p,d) in equation (5).

    Parameters
    ----------
    F: F matrix.
    B: B matrix.
    MV: MV matrix.

    Returns
    -------
    map: C matrix.
    """
    map = F+B-MV

    return map

def DH_map(left, right,C, P1, P2, P3, T, ndisp):
    """
    Calculate forward horizontal tree energy matrix DH, term l’(p,d) in equation (6).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    C: C matrix.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.

    Returns
    -------
    map: DH matrix.
    """

    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for i in range(0, height):
        for j in range(0, width):
            for d in range(1, ndisp+1):
                #print(i,j,d)
                if i == 0:
                    map[i,j,d] = C[i,j,d]
                else:
                    if abs(left[i,j] - left[i-1,j]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = C[i,j,d] + min(map[i-1,j,d],map[i-1,j,max(d-1,1)]+P1,map[i-1,j,min(d+1,ndisp)]+P1,np.min(map[i-1,j,1:])+P4)

    return map

def UH_map(left, right,C, P1, P2, P3, T, ndisp):
    """
    Calculate backward horizontal tree energy matrix UH, term l’(p,d) in equation (6).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    C: C matrix.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.

    Returns
    -------
    map: UH matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for i in range(height-1,-1,-1):
        for j in range(0, width):
            for d in range(1, ndisp+1):
                #print(i,j,d)
                if i == height-1:
                    map[i,j,d] = C[i,j,d]
                else:
                    if abs(left[i,j] - left[i+1,j]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = C[i,j,d] + min(map[i+1,j,d],map[i+1,j,max(d-1,1)]+P1,map[i+1,j,min(d+1,ndisp)]+P1,np.min(map[i+1,j,1:])+P4)

    return map

def H_map(DH, UH, C):
    """
    Calculate total horizontal tree energy matrix H.

    Parameters
    ----------
    DH: DH matrix.
    UH: UH matrix.
    C: C matrix.


    Returns
    -------
    map: H matrix.
    """
    map = DH+UH-C

    return map


def D_map(left, right, P1, P2, P3, T, ndisp, M):
    """
    Calculate forward vertical energy matrix D, term F(p,d) in equation (5).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.
    M: M matrix.

    Returns
    -------
    map: D matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for j in range(0, width):
        for i in range(0, height):
            for d in range(1, ndisp+1):
                m = M[i,j,d]
                if i == 0:
                    map[i,j,d] = m
                else:
                    if abs(left[i,j] - left[i-1,j]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = m + min(map[i-1,j,d],map[i-1,j,max(d-1,1)]+P1,map[i-1,j,min(d+1,ndisp)]+P1,np.min(map[i-1,j,1:])+P4)

    return map

def U_map(left, right, P1, P2, P3, T, ndisp, M):
    """
    Calculate backward vertical energy matrix U, term F(p,d) in equation (5).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.
    M: M matrix.

    Returns
    -------
    map: U matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for j in range(0, width):
        for i in range(height-1, -1, -1):
            for d in range(1, ndisp+1):
                m = M[i,j,d]
                if i == height-1:
                    map[i,j,d] = m
                else:
                    if abs(left[i,j] - left[i+1,j]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = m + min(map[i+1,j,d],map[i+1,j,max(d-1,1)]+P1,map[i+1,j,min(d+1,ndisp)]+P1,np.min(map[i+1,j,1:])+P4)

    return map

def Z_map(D, U, M):
    """
    Calculate total vertical energy matrix Z, term C(p,d) in equation (5).

    Parameters
    ----------
    D: D matrix.
    U: U matrix.
    M: M matrix.

    Returns
    -------
    map: C matrix.
    """
    map = D+U-M

    return map

def RV_map(left, right,Z, P1, P2, P3, T, ndisp):
    """
    Calculate forward vertical tree energy matrix RV, term l’(p,d) in equation (6).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    Z: Z matrix.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.

    Returns
    -------
    map: RV matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for j in range(0, width):
        for i in range(0, height):
            for d in range(1, ndisp+1):
                #print(i,j,d)
                if j == 0:
                    map[i,j,d] = Z[i,j,d]
                else:
                    if abs(left[i,j] - left[i,j-1]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = Z[i,j,d] + min(map[i,j-1,d],map[i,j-1,max(d-1,1)]+P1,map[i,j-1,min(d+1,ndisp)]+P1,np.min(map[i,j-1,1:])+P4)

    return map

def LV_map(left, right,Z, P1, P2, P3, T, ndisp):
    """
    Calculate backward vertical tree energy matrix LV, term l’(p,d) in equation (6).

    Parameters
    ----------
    left: left view image.
    right: right view image.
    Z: Z matrix.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.

    Returns
    -------
    map: LV matrix.
    """
    height, width = left.shape[0], left.shape[1]
    map = np.zeros((height,width,ndisp+1), dtype = np.float64)
    for j in range(width-1, -1, -1):
        for i in range(0, height):
            for d in range(1, ndisp+1):
                #print(i,j,d)
                if j == width-1:
                    map[i,j,d] = Z[i,j,d]
                else:
                    if abs(left[i,j] - left[i,j+1]) < T:
                        P4 = P3*P2
                    else:
                        P4 = P2
                    map[i,j,d] = Z[i,j,d] + min(map[i,j+1,d],map[i,j+1,max(d-1,1)]+P1,map[i,j+1,min(d+1,ndisp)]+P1,np.min(map[i,j+1,1:])+P4)

    return map

def V_map(RV, LV, Z):
    """
    Calculate total vertical tree energy matrix V.

    Parameters
    ----------
    RV: RV matrix.
    LV: LV matrix.
    Z: Z matrix.


    Returns
    -------
    map: V matrix.
    """
    map = RV+LV-Z
    return map


def simple_tree_DP(left, right, P1, P2, P3, T, ndisp, a):
    """
    Calculate disparity map using simple tree DP method.

    Parameters
    ----------
    left: left view image.
    right: right view image.
    P1: P1 parameter
    P2: P2 parameter
    P3: P3 parameter
    T: T parameter
    ndisp: bound of possible disparity values.
    a: alpha weight.

    Returns
    -------
    disparity: non-normalized disparity map.
    """
    M=M_map(left, right, ndisp)
    D=D_map(left, right, P1, P2, P3, T, ndisp,M)
    U = U_map(left, right, P1, P2, P3, T, ndisp,M)
    Z = Z_map(D, U, M)
    RV = RV_map(left, right,Z, P1, P2, P3, T, ndisp)
    LV = LV_map(left, right,Z, P1, P2, P3, T, ndisp)
    V = V_map(RV, LV, Z)
    MV=MV_map(M, V, a)
    F=F_map(left, right, P1, P2, P3, T, ndisp,MV)
    B = B_map(left, right, P1, P2, P3, T, ndisp,MV)
    C = C_map(F, B, MV)
    DH = DH_map(left, right,C, P1, P2, P3, T, ndisp)
    UH = UH_map(left, right,C, P1, P2, P3, T, ndisp)
    H = H_map(DH, UH, C)
    height, width = left.shape[0], left.shape[1]
    disparity = np.zeros((height,width),dtype = np.float64)
    for i in range(0,height):
        for j in range(0,width):
            disparity[i,j] = np.argmin(H[i,j,1:])+1

    return disparity




ndisp = int(input("Enter ndisp value: "))

left = (cv2.imread("left.png", cv2.IMREAD_GRAYSCALE)).astype(np.float64)
right = (cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)).astype(np.float64)

print("Processing SSD left to right")
ssd_l2r = ssd_disparity(left, right, 0, 3, ndisp)
cv2.normalize(ssd_l2r,ssd_l2r,0,255,cv2.NORM_MINMAX)
ssd_l2r = ssd_l2r.astype(np.uint8)
jet_l2r = cv2.applyColorMap(ssd_l2r,cv2.COLORMAP_JET)
cv2.imwrite("SSD_left2right.png",jet_l2r)

print("Processing SSD right to left")
ssd_r2l = ssd_disparity(left, right, 1, 3, ndisp)
cv2.normalize(ssd_r2l,ssd_r2l,0,255,cv2.NORM_MINMAX)
ssd_r2l = ssd_r2l.astype(np.uint8)
jet_r2l = cv2.applyColorMap(ssd_r2l,cv2.COLORMAP_JET)
cv2.imwrite("SSD_right2left.png",jet_r2l)

print("Processing simple tree DP")
P1,P2,P3,T,a = 20,30,4,30,0.025
disparity = simple_tree_DP(right, left, P1, P2, P3, T, ndisp, a)
cv2.normalize(disparity,disparity,0,255,cv2.NORM_MINMAX)
disparity = disparity.astype(np.uint8)
jet_disparity = cv2.applyColorMap(disparity,cv2.COLORMAP_JET)
cv2.imwrite("simple_tree_DP.png",jet_disparity)
