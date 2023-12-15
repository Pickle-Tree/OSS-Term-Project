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
