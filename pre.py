import cv2
import dlib
import sys
import numpy as np

# 이미지 크기 조절을 위한 스케일 변수
scaler = 1.0

# 얼굴 탐지를 위한 dlib의 face detector 생성
detector = dlib.get_frontal_face_detector()
# 얼굴 특징점 추출을 위한 dlib의 shape predictor 생성
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 웹캠에서 영상을 받아옴
cap = cv2.VideoCapture(0)
# 마스크 오버레이 이미지를 불러옴 (예시: Ironman.png)
overlay = cv2.imread('samples/Ironman.png', cv2.IMREAD_UNCHANGED)

# 투명한 이미지를 배경 이미지 위에 오버레이하는 함수
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)
    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

face_roi = []
face_sizes = []

result = None
