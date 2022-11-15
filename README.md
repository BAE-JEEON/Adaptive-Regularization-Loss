## Adaptive Regularization Loss for Improving Quality of  Pseudo-label in Class Imbalanced Semi-Supervised Learning
- 2022-2 독립 심화 학습
- 지도 교수: 배성호
- 주저자: 배제언
- 논문: Adaptive Regularization Loss for Improving Quality of  Pseudo-label in Class Imbalanced Semi-Supervised Learning                 (KSC2022)

## 주요 내용

- 준 지도 학습은 라벨링 되지 않은 많은 양의 데이터를 이용하여 라벨링의 어려움을 해결했을 뿐 아니라 지도
학습 못지않은 우수한 성능을 보였다. 그러나 클래스 불균형 상황에서 준 지도 학습은 지도 학습보다 강건하지
못하다는 문제가 있다. 최근 해결방안이 제시됐지만, 특정 알고리즘에 의존, 추가적인 학습 시간 필요, 라벨링
되지 않은 데이터에 대한 비현실적인 가정과 같은 문제가 여전히 존재했다. 본 논문은 클래스 불균형 준 지도
학습에 특화된 손실함수를 제안하여 이러한 문제를 극복하고자 했다. 이를 위해 먼저 객체 탐지 클래스 불균
형에서 자주 사용되는 Focal Loss가 준지도 학습에서 어떤 의미를 가지는지 분석했다. 그 결과 Focal Loss는
다수 클래스에 편향된 모델이 만든 Pseudo-label에 규제를 줄 수 있음이 확인되었다. 그러나 Focal Loss는
하이퍼파라미터에 민감하며 모든 샘플에 같은 크기의 규제를 준다는 한계점이 있었다. 이를 해결하기 위해 본
논문에서는 레이블에 대한 Confidence 및 불균형 정도를 기준으로 규제의 강도를 적응적으로 조절하는 손실
함수를 제안했다. 제안 방법은 CIFAR10-LT에서 앞선 제안 방법 대비 세 가지 한계점과 Focal Loss의 한계점
을 개선하여, 결과적으로 기존 손실함수와 비교하여 최대 7%의 성능향상을 보였다.

## 실험 결과

- 데이터셋:CIFAR10-LT
- 라벨링 데이터의 비율: 20%
- 사용한 준지도 학습 알고리즘: FixMatch
- 모델: Wide ResNet-28-2
- Threshold : 0.95
- 기타: 다른 하이퍼파라미터 값은 FixMatch 논문에서 사용한 최적 값을 그대로 사용하였다.  

- 불균형에 따른 손실함수별 성능 비교 (CIFAR10-LT)

|                |     CE     |     BCE     |     Ours(𝜸 = 1)     |     Ours(𝜸 = 2)     |     Ours+     |
|:--------------:|:---------: | :---------: | :-----------------: | :-----------------: | :-----------: |
| imb ratio= 50  |    82.06   |    84.42    |        85.11        |       84.00         |     85.13     |
| imb ratio= 100 |    73.17   |    76.61    |        79.17        |       78.37         |     80.00     | 
| imb ratio= 200 |    66.37   |    69.45    |        70.62        |       71.52         |     72.61     |



- 학습 수렴 ±50epoch의 Pseudo-label의 Confidence-Threshold의 평균 통과율 (APR)과 학습 효율도 (LER)



|                |     CE     |     BCE     |     Ours(𝜸 = 1)     |     Ours(𝜸 = 2)     |     Ours+     |
|:--------------:|:---------: | :---------: | :-----------------: | :-----------------: | :-----------: |
|      APR       | 0.98±0.02  |  0.85±0.05  |     0.82±0.06       |      0.77±0.08      |   0.78±0.08   |
|      LER       |    0.755   |      0.9    |        0.96         |        1.03         |     1.05      |




## 실행 방법
```
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --label_ratio 20 --num_max 1000  --imb_ratio 100 --imbalancetype long --out results/New_loss  --threshold 0.95 

```
## 레퍼 런스
- https://github.com/kekmodel/FixMatch-pytorch
- https://github.com/google-research/fixmatch
- https://github.com/ildoonet/pytorch-randaugment
- https://github.com/LeeHyuck/ABC
- https://github.com/AdeelH/pytorch-multi-class-focal-loss

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

