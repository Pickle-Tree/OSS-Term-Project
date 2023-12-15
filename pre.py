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

# 프로그램이 실행되는 동안 계속 반복
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    if len(face_roi) == 0:
        faces = detector(img, 1)
    else:
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        faces = detector(roi_img)
        
    if len(faces) == 0:
        print('No faces!')
        result = ori.copy()  
    else:
        for face in faces:
            if len(face_roi) == 0:
                dlib_shape = predictor(img, face)
                shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
            else:
                dlib_shape = predictor(roi_img, face)
                shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

            # 얼굴 특징점에 원을 그림
            for s in shape_2d:
                cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            center_x, center_y = np.mean(shape_2d, axis=0).astype(int)

            min_coords = np.min(shape_2d, axis=0)
            max_coords = np.max(shape_2d, axis=0)
            cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            face_size = int(np.max(max_coords - min_coords))
            face_sizes.append(face_size)
            if len(face_sizes) > 10:
                del face_sizes[0]
            mean_face_size = int(np.mean(face_sizes) * 1.8)

            face_roi = np.array([
                int(max(0, min_coords[1] - face_size / 2)),
                int(min(img.shape[0], max_coords[1] + face_size / 2)),
                int(max(0, min_coords[0] - face_size / 2)),
                int(min(img.shape[1], max_coords[0] + face_size / 2))
            ])

            result = overlay_transparent(ori, overlay, center_x + 8, center_y - 25, overlay_size=(mean_face_size, mean_face_size))

    # 각각의 이미지를 화면에 출력
    cv2.imshow('original', ori)
    cv2.imshow('facial landmarks', img)
    cv2.imshow('result', result)

    # 'q'를 누르면 프로그램 종료
    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)