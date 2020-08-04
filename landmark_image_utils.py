import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL

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

def gen_ldmk_image_color_68(landmarks, frame_size):
    # adapted from https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models/blob/save_disc/dataset/preprocess.py
    '''
    :param landmarks: input facial landmark coordinates
    :parem frame_size: image size
    :return: cv2_image
    '''
    preds = landmarks
    dpi = 100
    fig = plt.figure(figsize=(frame_size[1] / dpi, frame_size[0] / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.ones( (frame_size[0], frame_size[1], 3) ))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # chin
    ax.plot(preds[0:17, 0], preds[0:17, 1], marker='', markersize=5, linestyle='-', color='green', lw=2)
    # left and right eyebrow
    ax.plot(preds[17:22, 0], preds[17:22, 1], marker='', markersize=5, linestyle='-', color='orange', lw=2)
    ax.plot(preds[22:27, 0], preds[22:27, 1], marker='', markersize=5, linestyle='-', color='orange', lw=2)
    # nose
    ax.plot(preds[27:31, 0], preds[27:31, 1], marker='', markersize=5, linestyle='-', color='blue', lw=2)
    ax.plot(preds[31:36, 0], preds[31:36, 1], marker='', markersize=5, linestyle='-', color='blue', lw=2)
    # left and right eye
    ax.plot(preds[36:42, 0], preds[36:42, 1], marker='', markersize=5, linestyle='-', color='red', lw=2)
    ax.plot(preds[42:48, 0], preds[42:48, 1], marker='', markersize=5, linestyle='-', color='red', lw=2)
    # outer and inner lip
    ax.plot(preds[48:60, 0], preds[48:60, 1], marker='', markersize=5, linestyle='-', color='purple', lw=2)
    ax.plot(preds[60:68, 0], preds[60:68, 1], marker='', markersize=5, linestyle='-', color='pink', lw=2)
    ax.axis('off')

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #196608
    # print(fig.canvas.get_width_height()[::-1] + (3,)) # 256,256,3
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data


def gen_ldmk_image_color_68_V2(landmarks, frame_size):
    # adapted from https://github.com/grey-eye/talking-heads/blob/master/dataset/dataset.py
    """
        Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
        matching the landmarks.
        The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
        plot to an image.
        Things to watch out for:
        * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
        only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
        * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
        * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.
        :param frame: Image with a face matching the landmarks.
        :param landmarks: Landmarks of the provided frame,
        :return: RGB image with the landmarks as a Pillow Image.
        """
    dpi = 100
    fig = plt.figure(figsize=(frame_size[0] / dpi, frame_size[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones( (frame_size[0], frame_size[1], 3) ))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    data = np.array(data)
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data