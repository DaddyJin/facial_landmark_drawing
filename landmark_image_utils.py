import cv2
import numpy as np

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

def gen_ldmk_image_poly_68(landmarks, frame_size):
    '''
    :param landmarks: input facial landmark coordinates
    :parem frame_size: image size
    :return: cv2_image
    '''
    black_image = np.zeros(frame_size, np.uint8)

    jaw = reshape_for_polyline(landmarks[0:17])
    left_eyebrow = reshape_for_polyline(landmarks[22:27])
    right_eyebrow = reshape_for_polyline(landmarks[17:22])
    nose_bridge = reshape_for_polyline(landmarks[27:31])
    lower_nose = reshape_for_polyline(landmarks[31:36])
    left_eye = reshape_for_polyline(landmarks[42:48])
    right_eye = reshape_for_polyline(landmarks[36:42])
    outer_lip = reshape_for_polyline(landmarks[48:60])
    inner_lip = reshape_for_polyline(landmarks[60:68])

    color = (255, 255, 255)
    thickness = 1

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image

def gen_ldmk_image_poly_98(landmarks, frame_size):
    '''
    :param landmarks: input facial landmark coordinates
    :parem frame_size: image size
    :return: cv2_image
    '''
    black_image = np.zeros(frame_size, np.uint8)

    jaw = reshape_for_polyline(landmarks[0:33])
    left_eyebrow = reshape_for_polyline(landmarks[42:51])
    right_eyebrow = reshape_for_polyline(landmarks[33:42])
    nose_bridge = reshape_for_polyline(landmarks[51:55])
    lower_nose = reshape_for_polyline(landmarks[54:59])
    left_eye = reshape_for_polyline(landmarks[68:76])
    right_eye = reshape_for_polyline(landmarks[60:68])
    outer_lip = reshape_for_polyline(landmarks[76:88])
    inner_lip = reshape_for_polyline(landmarks[88:96])

    color = (255, 255, 255)
    thickness = 1

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], True, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], True, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image

def gen_ldmk_image_dot(landmarks, frame_size):
    black_image = np.zeros(frame_size, np.uint8)
    color = (255, 255, 255)
    thickness = -1
    for ldmk in landmarks:
        cv2.circle(black_image, tuple(ldmk), 1, color, thickness)
    return black_image

def list2ldmk(ldmk_list, ldmk_num=68):

    assert len(ldmk_list) == ldmk_num * 2
    coords = np.zeros((ldmk_num, 2), dtype=int)
    for i in range(ldmk_num):
        coords[i] = (int(float(ldmk_list[2 * i])), int(float(ldmk_list[2 * i + 1])))
    return coords

