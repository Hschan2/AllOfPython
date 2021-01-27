## openCV
# 실시간 컴퓨터 비전을 처리하는 목적으로 만들어진 라이브러리
# 단일 이미지, 동영상의 이미지를 원하는 결과를 분석 및 추출하기 위한 API
# 이미지를 대상으로 어떤 처리를 수행하여 이미지를 읽고 화면에 표시하는 것은 매우 중요하고 기초적인 내용

# 이미지를 읽어 화면에 표시하는 예제
from cv2 import cv2
import numpy as np

# 이미지 불러오기 (불러올 이미지, 어떤 포맷으로 메모리에 적재하여 numpy 자료 구조의 객체를 생성할 것인가)
# IMREAD_GRAYSCALE => Gray 색상으로 해석해 이미지 객체를 반환 (IMREAD_COLOR = 기본값이며 투명도 무시, IMREAD_UNCHANGED = 투명도인 Alpha 채널을 포함하여 읽기)
img = cv2.imread('../img/TVN_Logo.png', cv2.IMREAD_GRAYSCALE)
# 이미지 표시하기 (식별자, 이미지 객체)
cv2.imshow('image', img)

# 인자값이 없거나 0일 경우 시간 제한없이 사용자 키 입력 대기, 인자가 지정되면 해당값의 시간만큼 키 입력 대기
k = cv2.waitKey()

if k == ord('s'):
    # 다른 파일로 이미지 저장
    cv2.imwrite('d:/tvn.jpg', img)

cv2.destroyAllWindows()

## matplotlib를 함께 이용하기
from matplotlib import pyplot as plt

img2 = cv2.imread('../img/TVN_Logo.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

## 카메라 영상 처리
# 카메라(웹캠)으로부터 영상을 받아 처리하기 위해 VideoCapture 클래스 사용
# 여러 카메라를 사용할 지 결정은 카메라 아이디로 전달하고 일반적으로 0은 첫 번째 카메라(Default)를 사용 (두 번째 카메라 사용은 1, 세 번째 카메라 사용은 2...)
# VideoCapture 클래스의 read() 메서드를 호출하여 카메라 이미지(프레임)을 가져올 수 있다.

# 0: default camera
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("test.mp4") # 동영상 파일 읽기

# 동영상이 열려있을 때까지
while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    if success:
        # 프레임 출력하기
        cv2.imshow('Camera Windwo', frame)

        # ESC로 종료
        key = cv2.waitKey(1) & 0xFF
        if(key == 27):
            break

cap.release()
cv2.destroyAllWindows()

## 카메라 영상 저장하기
cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("size: {0}x{1}".format(width, height))

# 영상 저장을 위한 VideoWriter 인스턴스 생성하기
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('text.avi', fourcc, 24, (int(width), int(height)))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 프레임 저장하기
        writer.write(frame)
        cv2.imshow('Video Window', frame)

        # q로 종료하기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
# 저장 종료
writer.release()
cv2.destroyAllWindows()