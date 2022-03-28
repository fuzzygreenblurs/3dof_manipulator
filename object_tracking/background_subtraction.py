import numpy as np
import cv2
import ipdb

CM_TO_PIX = 12.9/640.0

def rotation_matrix():
    # TO_DO: memoize 
    rot_180_x = [[1, 0, 0],[0, -1, 0],[0, 0, -1]]
    rot_n90_z = [[0, 1, 0],[-1, 0, 0],[0, 0,  1]]
    return np.dot(rot_180_x, rot_n90_z)

def displacement_vector():
    # TO_DO: memoize 
    # respective dx, dy, dz values between camera and zero frame datums

    # Q: why does swapping the dx and dy positions in this array fix the problem???
    return np.array([[-0.69], [-5.85],[0]])
    # return np.array([[-5.85], [-0.69],[0]])

def homogeneous_transformation_matrix():
    # memoize
    # ipdb.set_trace()
    mat = np.concatenate((rotation_matrix(), displacement_vector()), 1)
    return np.concatenate((mat, np.array([[0,0,0,1]])), 0) # include this row to facilitate square matrix structure

def difference_frame(detection_frame, reference_frame):
    diff = np.absolute(
        np.matrix(np.int16(detection_frame)) -
        np.matrix(np.int16(reference_frame))
    )

    diff[diff > 255] = 255
    diff = np.uint8(diff)
    diff[diff <= 100] = 0
    diff[diff > 100] = 255
    
    return diff

# refactor to generalized function with a yield?
def y_coordinate(frame):
    # ipdb.set_trace()
    sums = np.matrix(np.sum(frame, 0))
    ids  = np.matrix(np.arange(640))
    weighted_cols = np.multiply(sums, ids)
    weighted_subtotal = np.sum(weighted_cols)
    frame_total = np.sum(np.sum(frame))

    c_location = weighted_subtotal/frame_total
    return c_location * CM_TO_PIX
    # return c_location

def x_coordinate(frame):
    sums = np.matrix(np.sum(frame, 1)).transpose()
    ids  = np.matrix(np.arange(480))
    weighted_cols = np.multiply(sums, ids)
    weighted_subtotal = np.sum(weighted_cols)
    frame_total = np.sum(np.sum(frame))

    c_location = weighted_subtotal/frame_total
    return c_location * CM_TO_PIX
    # return c_location

def normalized_c_vector(x,y):
    return np.array([[x], [y], [0], [1]])

def store_frame():
    _, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def interrupted():
    k = cv2.waitKey(5) 
    return k == 27

cap = cv2.VideoCapture(0)
print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)

i = 0
t = 0
x_avg = 0
y_avg = 0
sample = 500

# reference frame measurement
while(1):
    reference_frame = store_frame()


    cv2.imshow('reference', reference_frame)
    if interrupted(): break

# detection frame measurement
while(1):
    detection_frame = store_frame()
    # cv2.imshow('detection', detection_frame)
    
    diff = difference_frame(detection_frame, reference_frame)
    cv2.imshow('diff', diff)

    c_position = normalized_c_vector(x_coordinate(diff), y_coordinate(diff))
    o_position = np.dot(homogeneous_transformation_matrix(), c_position)
    
    if i == sample - 1:
        print(t, round(x_avg/sample, 2), round(y_avg/sample, 2))
        i, x_avg, y_avg = 0, 0, 0
    
    for i in range(0, sample):
        #Q: these needed to be switched as well!
        # the transform is yielding the x and y coordinates in the O_frame in a swapped position

        x_avg += o_position[1][0]
        y_avg += o_position[0][0]
        # x_avg += x_location(diff)
        # y_avg += y_location(diff)
        t += 1

    if interrupted(): break