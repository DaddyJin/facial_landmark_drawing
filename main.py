from landmark_image_utils import gen_ldmk_image_dot, \
    gen_ldmk_image_poly_68, gen_ldmk_image_poly_98, list2ldmk
import numpy as np
import cv2

if __name__ == '__main__':
    # load test image
    test_image_path = './test_image/P_Trump_257.jpg_0000000009.jpg'
    img = cv2.imread(test_image_path)
    image_shape = img.shape
    cv2.imshow('img', img)
    cv2.waitKey()

    # use dlib or relevant face landmark detection tool to get 68/98 landmarks
    # here we pre-compute the 68/98 landmarks and store it in a txt file
    ldm_txt_path = './test_image/landmark_info.txt'
    images_info = []
    with open(ldm_txt_path, 'r') as f:
        for line in f:
            images_info.append(line.strip('\n'))
    ldmk_info_68 = images_info[0]
    ldmk_info_98 = images_info[1]

    ldmk_98 = list2ldmk(ldmk_info_98.split(' ')[:-1], 98)
    ldmk_68 = list2ldmk(ldmk_info_68.split(' ')[:-1], 68)

    # call utils to draw corresponding landmark 0/1 images
    poly_image_68 = gen_ldmk_image_poly_68(ldmk_68, image_shape)
    poly_image_98 = gen_ldmk_image_poly_98(ldmk_98, image_shape)
    dot_image_68 = gen_ldmk_image_dot(ldmk_68, image_shape)
    dot_image_98 = gen_ldmk_image_dot(ldmk_98, image_shape)

    final = cv2.hconcat([poly_image_68, poly_image_98, dot_image_68, dot_image_98])
    cv2.imshow('poly_68/poly_98/dot_68/dot_98', final)
    cv2.waitKey()