'''
README
To compare student outputs with ground truth:
1. Put metrics.py and two images (student.png and ground.png) in the same directory.
2. From a terminal, activate the CS6476 virtual environment: ~$ conda activate CS6476
3. From the metrics.py directory, run metrics.py file: ~$ python3 metrics.py
4. You will be asked to enter ndisp value.
5. You will see outputs "Comparing histogram", "The histogram correlation is xxx", "Comparing normalized averaged difference", "The normalized averaged difference is xxx",
which are the two metrics used for discussion in the report.
'''

import numpy as np
import cv2



def compareHis(student, ground):
    """
    Compare histograms of two images, using correlation method.

    Parameters
    ----------
    student: student disparity map, in gray scale.
    ground: ground truth disparity map, in gray scale.

    Returns
    -------
    correlation: The correlation of histograms of student result and ground truth, in the range
    of 0-1, while 0 represents the lowest similarity and 1 represents the highest similarity
    """

    his1 = cv2.calcHist([student], [0], None, [256], [0, 255])
    his2 = cv2.calcHist([ground], [0], None, [256], [0, 255])
    correlation = cv2.compareHist(his1, his2, method=cv2.HISTCMP_CORREL)
    return correlation

def compareDiff(student, ground, ndisp):
    """
    Compare histograms of two images, using correlation method.

    Parameters
    ----------
    student: student disparity map, in gray scale.
    ground: ground truth disparity map, in gray scale.
    ndisp: bound of possible disparity values.

    Returns
    -------
    ave_diff: The normalized averaged difference of student result and ground truth.
    """
    height, width = student.shape[0], student.shape[1]
    diff = np.absolute(student-ground)
    sum_diff = np.sum(diff)
    ave_diff = sum_diff/(height*width)/250
    return ave_diff


ndisp = int(input("Enter ndisp value: "))

student = cv2.imread("student.png", cv2.IMREAD_GRAYSCALE)
ground = cv2.imread("ground.png", cv2.IMREAD_GRAYSCALE)

print("Comparing histogram")
correlation = compareHis(student, ground)
print("The histogram correlation is {}.\n".format(correlation))

print("Comparing normalized averaged difference")
ave_diff = compareDiff(student, ground, ndisp)
print("The normalized averaged difference is {}.\n".format(ave_diff))
