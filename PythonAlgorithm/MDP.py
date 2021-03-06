## Markov Decision Process (마코프 의사 결정)

##  마코프 의사 결정이 필요한 이유
# 자율 주행 자동차에 필요한 알고리즘을 만들 때, 확률계에서 순차적 의사 결정 문제를 푸는 것이 중요.
# 자동차 제어 시, 빗길, 비포장 도로 등 다양한 원인으로 인해 원하지 않는 결과를 가져올 수 있음.
# 이 때, 예상과 일치하지 않은 상황을 확률계라고 한다.
# 자율 주행 자동차가 안전하게 운전할 수 있기 위해 일련의 결과로 얻어지는 상태들이 모두 안전해야 한다.
# 그래서 알고리즘을 개발하기 위해 "강화 학습"은 확률적 의사 결정을 푸는 방법론을 지칭

## 강화 학습
# 지도 학습은 학습 데이터를 통해 유의미한 정보를 얻어내는 기계 학습 방법론
# 입 출력 데이터가 주어졌을 때 새로운 입력에 대한 출력을 예측하는 방법론이며 입 출력 데이터가 모두 주어진 상태에서 학습

## 비지도 학습
# 지도 학습과 반대되는 개념으로 학습 데이터가 미리 주어지지 않는 방법
# 대신 강화 학습에서는 "보상 함수"가 주어진다.
# 강화 학습은 미래에 얻어질 보상값들의 평균값을 최대로 하는 정책 함수를 찾는 것
# 여기서 "미래"란 "기대값"을 의미하며 이 문제를 풀기 위해 마코프 의사 결정 과정을 차용

## 마코프 의사 결정
# 불확실한 상황 하에서 의사 결정을 하기 위해서는 "확률"에 기초하여 분석이 필요
# 어떠한 사건이 발생할 확률 값이 시간에 따라 변화해 가는 과정을 확률적 과정이라고 하며, 이 중에서 특별한 과정이 마코프 과정
# 어떤 상태가 일정한 간격으로 변하고 다음 상태는 현재 상태에만 의존하며 확률적으로 변하는 경우의 상태의 변화를 뜻한다.
# 즉, 현재 상태에 대해서만 다음 상태가 결정, 현재 상태에 이르기 까지의 과정은 고려할 사항이 아니다.

## 마코프 연쇄
# 마코프 과정에서 연속적인 시간 변화를 고려하지 않고 이산적인 경우만 고려한 경우
# 각 시행의 결과가 여러 개의 미리 정해진 결과 중 하나가 되고, 각 시행의 결과는 과거의 역사와는 무관하며 오직 바로 직전 시행 결과에만 영향을 받는다.

## 최적 정책 함수를 찾기
# 강화 학습을 푸는 가장 기본적인 방법 - 값 반복, 정책 반복
# 여기서 "값"이란 특정 상태에서 시작했을 때, 얻을 수 있을 것으로 기대하는 미래 보상의 합을 구할 수 있다면 해당 함수를 매번 최대로 만드는 행동을 선택할 수 있고 최적의 정책 함수를 구할 수 있다.
# 미래에 얻을 수 있는 보상들의 합의 기대 값을 값 함수
# 값 함수는 현재 상태와 미래 상태들, 그 상태에서 얻을 수 있는 보상을 구해야 하기 때문에 직관적으로 정의 불가
# 강화 학습에서 이 값 함수를 구하기 위해 이퀘이션을 활용

