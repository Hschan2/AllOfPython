## 패턴 인식
# 손글씨, 도형 등 이미지에 나타나는 패턴을 인식하다

## 이미지 안에 원형 패턴 인식하기
from cv2 import cv2
import numpy as np

# 원형 찾기 함수
def houghCircle():
    image_1 = cv2.imread('원형이 있는 이미지.png')
    image_2 = image_1.copy()

    image_2 = cv2.GaussianBlur(image_2, (9, 9), 0) # GaussianBlur로 이미지 흐리게 만들기
    image_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY) # 이미지 흑백으로 만들기

    # 원본과 비율, 찾은 원형 모양 간 최소 중심거리, param1, param2를 조절하여 원 찾기
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 10, param1 = 60, param2 = 50, minRadius = 0, maxRadius = 0)

    if circles is not None: # 이미지 안에 원형 모양이 있다면
        circles = np.unit16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(image_1, (i[0], i[1]), i[2], (255, 255, 0), 2)

        cv2.imshow('ori', image_2)
        cv2.imshow('HoughCircle', image_1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("이미지 안에 원형 모양이 없음")

houghCircle()

## 숫자 인식
