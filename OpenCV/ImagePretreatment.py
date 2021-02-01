# 이미지 데이터 전처리
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## 흑백 이미지 전처리
image = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE) # 흑백 이미지 가져오기
plt.imshow(image, cmap = 'gray'), plt.axis('off') # 이미지 출력
plt.show()

type(image) # 데이터 타입 확인
image # 이미지 데이터 확인
image.shape # 해상도 확인

## 흑백 이미지를 컬러이미지로 변경
image_bgr = cv2.imread('../img/black.jpg', cv2.IMREAD_COLOR) # 이미지를 컬러로 로드
image_bgr[0, 0] # 픽셀 확인
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # 이미지 RGB로 변환
plt.imshow(image_rgb), plt.axis('off') # 이미지 출력
plt.show()

## 이미지 크기 변경
# 이미지의 크기를 줄여서 메모리 사용량 줄이기 가능
# 자주 사용하는 이미지 크기 = 32x32, 64x64, 96x96, 245x256
image_Size = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
image_50x50 = cv2.resize(image_Size, (50, 50)) # 이미지 크기를 50x50 Pixel로 변경
plt.imshow(image_50x50, cmap = 'gray'), plt.axis('off') # 이미지 출력
plt.show()

## 이미지 자르기
# 이미지 주변 제거하여 차원 줄이기 가능
# 이미지는 2차원 Numpy 배열로 저장
# 특정 부분을 행과 열을 선택하여 이미지 자르기 가능
image_Cut = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
image_cropped = image_Cut[:, :128] # 열의 처음 절반과 모든 행 선택
plt.imshow(image_cropped, cmap = 'gray'), plt.axis('off') # 이미지 출력
plt.show()

## 이미지 투명도 처리
image_Blur = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
image_blurry = cv2.blur(image_Blur, (5, 5)) # Blur 처리
plt.imshow(image_blurry, cmap = 'gray'), plt.axis('off')
plt.show()
image_very_blurry = cv2.blur(image_Blur, (100, 100)) # 더 강하게 Blur 처리
plt.imshow(image_very_blurry, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()

## Kernel을 이용하여 이미지 투명도 처리
# 커널은 이미지를 선명하게 만드는 것부터 경계선 감지까지 이미지 처리 작업을 하는데 많이 사용
# 커널 PCA와 서포트 벡터 머신이 사용하는 비선형 함수를 커널
# 커널의 크기는 (너비, 높이)로 지정
# 주변 픽셀값의 평균을 계산하는 커널은 이미지를 흐리게 처리
# Blur 함수는 각 픽셀에 커널 개수의 역수를 곱하여 모두 더하기
kernel = np.ones((5, 5)) / 25.0 # 커널 생성
image_kernel = cv2.filter2D(image_Blur, -1, kernel) # 커널 적용
plt.imshow(image_kernel, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()

image_very_blurry_kernel = cv2.GaussianBlur(image_Blur, (5, 5), 0) # GaussianBlur 적용
plt.imshow(image_very_blurry_kernel, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()

## kernel + vector + Blur
# GaussianBlur의 세번째 매개변수 = X축 9 너비 방향의 표준편차
# GaussianBlur에 사용한 커널은 각 축 방향으로 가우시안 분포를 따르는 1차원 배열을 만든 다음 외적으로 생성
# getGaussianKernel를 사용하여 1차원 배열을 만들고 Numpy outer 함수로 외적 계산
gaus_vector = cv2.getGaussianKernel(5, 0)
gaus_kernel = np.outer(gaus_vector, gaus_vector) # Vector를 외적으로 커널 생성
image_kernel_vector  = cv2.filter2D(image_Blur, -1, gaus_kernel) # 커널 적용
plt.imshow(image_kernel, cmap = 'gray'), plt.xticks([]), plt.yticks([])
plt.show()

## 이미지 선명하게
# 대상 픽셀을 강조하는 커널 생성 후 filter2D를 사용하여 이미지 커널에 적용
# 중앙 픽셀을 부각하는 커널을 생성하면 이미지 경계선에서 대비가 더욱 두드러지는 효과 발생
image_clear = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
kernel_clear = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # 커널 생성

image_sharp = cv2.filter2D(image_clear, -1, kernel_clear) # 이미지 선명하게 생성
plt.imshow(image_sharp, cmap = 'gray'), plt.axis('off')
plt.show()

## 이미지 대비 높이기
# 히스토그램 평활화는 객체의 형태가 두드러지도록 만들어주는 이미지 처리 도구
# Y는 루마 또는 밝기. U, V는 컬러
# 흑백 이미지에 equalizeHist를 바로 적용 가능
image_high = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(image_high) # 이미지 대비 높이기
plt.imshow(image_sharp, cmap = 'gray'), plt.axis('off')
plt.show()

image_bgr_high = cv2.imread('../img/black.jpg') # 이미지 로드
image_yuv = cv2.cvtColor(image_bgr_high, cv2.COLOR_BGR2YUV) # YUV로 변경
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0]) # 히스토그램 평활화 적용
image_rgb_high = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB) # RGB로 변환
plt.imshow(image_sharp, cmap = 'gray'), plt.axis('off')
plt.show()

## 색상 구분
# 색 범위를 정의하고 이미지에 마스크 적용
# 이미지를 HSV(색상, 채도, 명도)로 변환하고 격리시킬 값의 범위 정의 후 이미지에 적용할 마스크 생성 (마스크의 흰색 영역만 유지)
# bitwise_and는 마스크를 적용하고 원하는 포맷으로 변환
image_bgr = cv2.imread('../img/black.jpg') # 이미지 로드
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # BGR에서 HSV로 변환
lower_blue = np.array([50, 100, 50]) # HSV에서 파랑 값의 범위를 정의
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(image_hsv, lower_blue, upper_blue) # 마스크를 생성
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask) # 이미지에 마스크를 적용
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 변환

plt.imshow(image_rgb), plt.axis("off") # 이미지를 출력
plt.show()

plt.imshow(mask, cmap='gray'), plt.axis("off") # 마스크 출력
plt.show()

## 이미지 이진화
# 이진화(임계처리) thresholding은 어떤 값보다 큰 값을 가진 픽셀을 흰색으로 만들고 작은 값을 가진 픽셀을 검은색으로 만드는 과정
# 픽셀의 임계값이 주변 픽셀의 강도에 의해 결정
# 이미지 안의 영역마다 빛 조건이 달라질 때 도움
image_grey = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
# adaptiveThreshold의 max_output_value 매개 변수는 출력 픽셀 강도의 최대값 결정
# ADAPTIVE_THRESH_GAUSSIAN_C = 픽셀의 임계값을 주변 픽셀 강도의 가중치 합으로 설정
image_binarized = cv2.adaptiveThreshold(image_grey, max_output_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighborhood_size, subtract_from_mean) # 적응적 임계처리를 적용
plt.imshow(image_binarized, cmap="gray"), plt.axis("off") # 이미지 출력
plt.show()

# ADAPTIVE_THRESH_MEAN_C = 픽셀의 임계값을 주변 픽셀의 평균으로 설정
image_mean_threshold = cv2.adaptiveThreshold(image_grey, max_output_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, neighborhood_size, subtract_from_mean)
plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()

## 이미지의 배경 제거
# 원하는 전경 주위의 사각형 박스를 그리고 그랩컷 알고리즘 실행
# 그랩컷 = 사각형 밖에 있는 모든 것이 배경이라고 가정하고 이 정보를 사용하여 사각형 안에 있는 배경 찾기
# 검은색 영역은 배경이라고 확실하게 가정한 사각형의 바깥쪽 영역이며, 회색 영역은 그랩컷이 배경이라고 생각하는 영역, 흰색 영역은 전경
image_bgr = cv2.imread('../img/black.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # RGB로 변환

rectangle = (0, 56, 256, 150) # 사각형 좌표: 시작점의 x, 시작점의 y, 너비, 높이

mask = np.zeros(image_rgb.shape[:2], np.uint8) # 초기 마스크를 만듭니다.

bgdModel = np.zeros((1, 65), np.float64) # grabCut에 사용할 임시 배열을 만듭니다.
fgdModel = np.zeros((1, 65), np.float64)

# 그랩컷(grabCut) 실행
cv2.grabCut(image_rgb, # 원본 이미지
            mask, # 마스크
            rectangle, # 사각형
            bgdModel, # 배경을 위한 임시 배열
            fgdModel, # 전경을 위한 임시 배열
            5, # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 사용한 초기화
            
# 배경인 곳은 0, 그외에는 1로 설정한 마스크를 만듭니다.
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱해 배경을 제외합니다.
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
plt.imshow(image_rgb_nobg), plt.axis("off") # 이미지 출력
plt.show()

plt.imshow(mask, cmap = 'gray'), plt.axis("off") # 마스크 출력
plt.show()

plt.imshow(mask_2, cmap = 'gray'), plt.axis("off") # 마스크 출력
plt.show()

## 경계선 감지
# 캐니(Canny) 경계선 감지기와 같은 경계선 감지 기술 사용
# 컴퓨터 비전의 주요 관심 대상, 많은 정보가 담긴 영역
# 낮은 임계값과 높은 임계값이 필수 매개 변수
image_gray = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
median_intensity = np.median(image_gray) # 픽셀 강도의 중간값을 계산

# 중간 픽셀 강도에서 위아래 1 표준 편차 떨어진 값을 임계값으로 지정합니다.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 캐니 경계선 감지기를 적용합니다.
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

plt.imshow(image_canny, cmap = "gray"), plt.axis("off") # 이미지 출력
plt.show()

## 모서리 감지
# cornerHarris = 해리스 모서리 감지
# 두 개의 경계선이 교차하는 지점을 감지하는 방법
# 윈도(이웃, 패치)안 픽셀이 작은 움직임에도 크게 변하는 윈도 찾기
image_bgr = cv2.imread('../img/black.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

block_size = 2 # 모서리 감지 매개변수를 설정
aperture = 29
free_parameter = 0.04

# block_size = 각 픽셀에서 모서리 감지에 사용되는 이웃 픽셀 크기
# aperture = 사용하는 소벨 커널 크기
detector_responses = cv2.cornerHarris(image_gray, block_size, aperture, free_parameter) # 모서리를 감지
detector_responses = cv2.dilate(detector_responses, None) # 모서리 표시를 부각시킵니다.

# 임계값보다 큰 감지 결과만 남기고 흰색으로 표시합니다.
threshold = 0.02
image_bgr[detector_responses > threshold * detector_responses.max()] = [255,255,255]

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # 흑백으로 변환

plt.imshow(image_gray, cmap = "gray"), plt.axis("off") # 이미지 출력
plt.show()

# 가능성이 높은 모서리를 출력합니다.
plt.imshow(detector_responses, cmap = 'gray'), plt.axis("off")
plt.show()

# 모서리 감지
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 감지할 모서리 개수
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

corners = cv2.goodFeaturesToTrack(image_gray, corners_to_detect, minimum_quality_score, minimum_distance) # 모서리를 감지
corners = np.float32(corners)

for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (x,y), 10, (255, 255, 255), -1) # 모서리마다 흰 원을 그립니다.
    
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # 흑백 이미지로 변환
plt.imshow(image_rgb, cmap = 'gray'), plt.axis("off") # 이미지를 출력
plt.show()

## 머신러닝 특성 생성
# 이미지를 머신러닝에 필요한 샘플로 변환하기 위해 Numpy의 flatten(이미지 데이터가 담긴 다차원 배열을 샘플 값이 담긴 벡터로 변환) 사용
# 이미지가 흑백일 때 각 픽셀은 하나의 값으로 표현
# 컬럼 이미지라면 각 픽셀이 하나의 값이 아닌 여러 개의 값으로 표현
# 이미지가 커질수록 특성의 개수 크게 증가
image = cv2.imread('../img/black.jpg', cv2.IMREAD_GRAYSCALE)
image_10x10 = cv2.resize(image, (10, 10)) # 이미지를 10x10 픽셀 크기로 변환
image_10x10.flatten() # 이미지 데이터를 1차원 벡터로 변환

plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

image_10x10.shape
image_10x10.flatten().shape

## 평균 색을 특성으로 인코딩
# 이미지 각 픽셀은 여러 색 채널(RGB)의 조합으로 표현, 채널 평균값을 계산하여 이미지 평균 컬러를 나타내는 3개의 컬럼 특성 생성
image_bgr = cv2.imread("../img/black.jpg", cv2.IMREAD_COLOR) # 색상 이미지 가져오기
channels = cv2.mean(image_bgr) # 각 채널의 평균을 계산

# 파랑과 빨강을 변경(BGR에서 RGB로 만듭니다)
observation = np.array([(channels[2], channels[1], channels[0])])
observation # 채널 평균 값을 확인
plt.imshow(observation), plt.axis("off") # 이미지를 출력
plt.show()

## 색상 히스토그램을 특성으로 인코딩
image_bgr = cv2.imread("../img/black.jpg", cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)# RGB로 변환
features = [] # 특성 값을 담을 리스트
colors = ("r", "g", "b") # 각 컬러 채널에 대해 히스토그램을 계산

# 각 채널을 반복하면서 히스토그램을 계산하고 리스트에 추가
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # 이미지
                             [i], # 채널 인덱스
                             None, # 마스크 없음
                             [256], # 히스토그램 크기
                             [0, 256]) # 범위
    features.extend(histogram)
    
observation = np.array(features).flatten() # 샘플의 특성 값으로 벡터를 생성
observation[0:5] # 5개 출력

image_rgb[0,0] # RGB 채널 값을 확인

data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5]) # 예시 데이터
data.hist(grid = False) # 히스토그램을 출력
plt.show()

colors = ("r", "g", "b") # 각 컬러 채널에 대한 히스토그램을 계산
# 컬러 채널을 반복하면서 히스토그램을 계산하고 그래프를 그리기
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # 이미지
                             [i], # 채널 인덱스
                             None, # 마스크 없음
                             [256], # 히스토그램 크기
                             [0, 256]) # 범위
    plt.plot(histogram, color = channel)
    plt.xlim([0, 256])
    
plt.show() # 그래프를 출력