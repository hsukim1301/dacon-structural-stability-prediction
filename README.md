# 구조물 안전성 예측 (Multi-View Structural Stability Prediction)

> **대회 목표:** front(정면) + top(상단) 2장의 이미지를 이용해 구조물의 안정성(stable / unstable) 예측  
> **평가 지표:** Binary Log-Loss (낮을수록 좋음)  
> **환경:** Google Colab (GPU), PyTorch

---

## 버전별 성능 요약

| 버전 | 백본 | 융합 방식 | 손실 함수 | 풀링 | 학습 전략 | Val Log-Loss |
|------|------|-----------|-----------|------|-----------|:------------:|
| Baseline | ResNet18 | Concatenate | BCE | AvgPool | Adam, 단일 학습 | 2.37 |
| V1 | EfficientNet-B5 | Cross-View Attention | Label Smoothing BCE | AvgPool | AdamW, 5-Fold, Cosine Annealing | 0.45 (↑ 향상) |
| V2 | EfficientNetV2-M | Cross-View Attention | Focal + Label Smoothing | GeM | 5-Fold + Pseudo Labeling | 0.15 (↑ 향상) |
| V3 | EfficientNetV2-M | Multi-Head Attention | Focal + Label Smoothing | GeM | 5-Fold + MixUp + Pseudo Labeling | 0.32 (↓ 하락) |

---

## 접근 방법 및 실험 과정

### Baseline — 대회 제공 기본 코드

front/top feature를 단순 concatenate 후 분류하는 가장 기본적인 구조.

- 백본: ResNet18 (224×224)
- 융합: Concatenate
- 학습: Adam, 단일 학습, 기본 Augmentation (HorizontalFlip, Rotation, ColorJitter)

---

### V1 — 백본 강화 + Multi-View Attention 설계

Baseline의 두 가지 핵심 한계를 개선했다.

**한계 1: ResNet18은 표현력이 부족하다.**  
EfficientNet-B5로 교체하고 이미지 크기를 224→380으로 키워 더 세밀한 특징을 추출했다.

**한계 2: Concatenate는 두 뷰의 관계를 학습하지 못한다.**  
구조물의 안전성은 정면과 상단이 서로 어떻게 연관되는지가 중요하다는 점에 착안해,  
front가 top을 참조하고 top이 front를 참조하는 **Cross-View Attention** 모듈을 직접 설계했다.

```
front feature (f1) ──┐
                     ├── 양방향 Cross-View Attention ──► fused (f1', f2') ──► classifier
top feature   (f2) ──┘
```

**주요 변경점:**
- 백본: ResNet18 → EfficientNet-B5
- 뷰 융합: Concat → Cross-View Attention (양방향 + Residual + LayerNorm)
- 손실: BCE → Label Smoothing BCE (α=0.05)
- 옵티마이저: Adam → AdamW + Differential LR (backbone 1e-5 / head 1e-4)
- 스케줄러: ReduceLROnPlateau → Cosine Annealing with Warmup
- 학습: 단일 → 5-Fold Cross Validation + TTA 앙상블
- 기타: Mixed Precision (AMP), Gradient Clipping, Early Stopping

**결과: Baseline 대비 성능 향상**

---

### V2 — 3D 모델링 시도 → Pseudo Labeling으로 전환

V1 이후, **2장의 2D 이미지를 기반으로 3D 구조를 모델링하면 더 정확한 안전성 판단이 가능하지 않을까** 라는 아이디어를 떠올렸다.

그러나 공부하면서 3D 복원(NeRF, MVSNet 등)은 수십~수백 장의 뷰 이미지가 필요하다는 것을 알게 됐다. front와 top 2장만으로는 측면 정보와 깊이 정보가 없어 복원 품질이 너무 낮고, 오히려 노이즈가 누적될 가능성이 높았다. 이미 V1의 Cross-View Attention이 2D feature 수준에서 두 뷰의 관계를 학습하고 있다는 점에서, 별도의 3D 복원보다 **데이터를 더 잘 활용하는 방향**을 탐색했다.

이 과정에서 **Pseudo Labeling**을 접하게 되어 적용했다. 학습된 모델로 test 데이터를 예측하고, 확신도 높은 샘플을 학습 데이터에 추가해 재학습하는 방식이다.

**주요 변경점:**
- **[A] 백본:** EfficientNet-B5 → EfficientNetV2-M (더 빠르고 정확, 480 해상도)
- **[B] 손실:** Label Smoothing → Focal + Label Smoothing 혼합  
  어려운 샘플에 학습을 집중시켜 결정 경계 근처 성능 개선 (γ=1.5, α=0.75)
- **[C] 풀링:** AvgPool → GeM Pooling (학습 가능한 파라미터 p로 자동 최적화)
- **[D] 데이터:** Pseudo Labeling 2단계 학습  
  Phase 1 학습 → test 예측 → 고신뢰 샘플(prob < 0.05 or > 0.95) 선택 → Phase 2 재학습  
  Phase 1 vs Phase 2 자동 비교 후 최선 채택

**결과: V1 대비 성능 향상**

---

### V3 — 추가 정규화 및 표현력 강화

V2 구조를 유지하면서 학습 안정성과 표현력을 높이기 위한 기법들을 추가로 적용했다.

**주요 변경점:**
- **[E] MixUp Augmentation:** 배치 내 샘플을 혼합해 정규화 효과 (런타임 증가 없음)
- **[F] Classifier Head BatchNorm 추가:** 수렴 가속
- **[G] Warmup Steps 확대:** 2 epoch → 5 epoch (초기 학습 불안정 해소)
- **[H] Multi-Head Cross-View Attention:** V2 single-head → 8-head로 업그레이드  
  PyTorch `nn.MultiheadAttention` 사용, CUDA 최적화 연산으로 속도 향상
- **[I] Pseudo Label 임계값 완화:** 0.05 → 0.10 (학습 샘플 수 증가)

**결과: V2 대비 성능 하락**

여러 기법을 한꺼번에 적용하면서 어떤 요소가 성능을 저하시켰는지 특정하기 어려웠다.  
시간이 있었다면 각 기법을 하나씩 ablation study로 검증하는 과정이 필요했을 것이다.

---

## 회고 및 향후 방향

### 실험을 마치며

V1→V2까지는 각 기법의 목적이 명확했고 성능도 일관되게 개선됐다.  
V3에서는 여러 기법을 한꺼번에 적용하면서 성능이 오히려 하락했는데,  
변경 사항이 많을수록 원인 파악이 어렵고, 한 번에 하나씩 검증하는 것이 얼마나 중요한지를 실감한 경험이었다.

### 향후 방향 — PyBullet 기반 합성 데이터 생성

대회 막바지에 **[PyBullet](https://pybullet.org)** 을 발견했다.  
PyBullet은 물리 시뮬레이션 라이브러리로, 구조물의 물리적 안정성을 시뮬레이션하는 기능을 제공한다.  
대회에서 제공한 데이터와 유사한 형태의 구조물 이미지를 직접 생성할 수 있다는 점이 흥미로웠다.

시간이 더 있었다면 아래 방식을 시도했을 것이다.

1. PyBullet으로 다양한 구조물 배치를 시뮬레이션해 **front + top 이미지를 직접 생성**
2. 물리 엔진에서 안정성 여부를 자동으로 레이블링
3. 생성된 대량의 합성 데이터로 사전 학습 후, 실제 대회 데이터로 fine-tuning

이 접근 방식을 적용했다면 데이터 부족 문제를 해소하고, 더 높은 성능을 기대할 수 있었을 것이다.

---

## 실행 방법

### 환경
```
Google Colab (T4 / A100 GPU 권장)
Python 3.10+
PyTorch 2.x
torchvision
scikit-learn
```

### 데이터 경로 설정
각 노트북의 `CFG['BASE_PATH']`를 본인의 Google Drive 경로로 변경:
```python
CFG = {
    'BASE_PATH': '/content/drive/MyDrive/YOUR_PATH/Stability_Project',
    ...
}
```

### 노트북 실행 순서
```
notebooks/baseline.ipynb
notebooks/v1_efficientnet_attention.ipynb
notebooks/v2_focal_gem_pseudo.ipynb
notebooks/v3_multihead_mixup.ipynb
```

### 런타임 재연결 후 이어서 실행
1. 셀 1~9 순서대로 실행 (라이브러리 및 변수 초기화)
2. Phase 1 학습 셀 실행 → 완료된 fold는 `done.txt` 확인 후 자동 스킵
3. 중단된 지점부터 재개

---

## 디렉토리 구조

```
dacon-structural-stability-prediction/
├── README.md               
└── notebooks/              
    ├── [Baseline]_Multi_View_ResNet_기반_구조물_안정성_예측.ipynb      
    ├── improved_multiview_stability.ipynb   
    ├── improved_v2_multiview_stability.ipynb   
    └── improved_v3_multiview_stability.ipynb   
```

---

## 참고 자료
- [EfficientNetV2 논문](https://arxiv.org/abs/2104.00298)
- [Focal Loss 논문](https://arxiv.org/abs/1708.02002)
- [GeM Pooling 논문](https://arxiv.org/abs/1711.02512)
- [PyBullet 공식 문서](https://pybullet.org)
