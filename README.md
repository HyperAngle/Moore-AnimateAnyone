# Moore-AnimateAnyone 마스터 노트

---

## 목차
1. 프로젝트 개요
2. 깃허브 코드 구조
3. 실행 방법 (Colab 기준)
4. 아키텍처 구성 및 입출력 흐름
5. 학습 과정
6. 공부 로드맵

---

## 1. 프로젝트 개요

**Moore-AnimateAnyone**는 인물 사진 1장 + 포즈 영상을 입력으로 받아, 해당 인물이 포즈대로 움직이는 영상을 생성하는 AI 프로젝트.

- 원본 논문: AnimateAnyone (HumanAIGC)
- 구현체: MooreThreads 재현 버전
- 권장 환경: Python >= 3.10, CUDA = 11.7

```
입력: 인물 사진 1장 + 포즈 영상
출력: 인물이 포즈대로 움직이는 영상
```

---

## 2. 깃허브 코드 구조

```
Moore-AnimateAnyone/
├── scripts/
│   ├── pose2vid.py         ← 추론 진입점 (AnimateAnyone 실행)
│   └── lmks2vid.py         ← 추론 진입점 (Face Reenactment 실행)
│
├── src/
│   └── models/
│       ├── unet_2d_condition.py    ← Denoising UNet
│       ├── unet_3d.py              ← Motion Module 포함 UNet
│       ├── reference_net.py        ← ReferenceNet
│       └── pose_guider.py          ← Pose Guider
│
├── configs/
│   ├── inference/
│   │   ├── ref_images/             ← 예시 레퍼런스 이미지
│   │   ├── pose_videos/            ← 예시 포즈 영상 (_kps 붙은 것)
│   │   └── inference.yaml          ← 추론 설정
│   └── train/
│       ├── stage1.yaml             ← 1단계 학습 설정
│       └── stage2.yaml             ← 2단계 학습 설정
│
├── tools/
│   ├── download_weights.py         ← 사전학습 가중치 자동 다운로드
│   └── vid2pose.py                 ← 일반 영상 → 포즈 영상 변환
│
├── train_stage_1.py                ← 1단계 학습 실행
├── train_stage_2.py                ← 2단계 학습 실행
├── app.py                          ← Gradio 데모 앱
└── requirements.txt
```

### 핵심 파일 3개만 기억하기
| 파일 | 역할 |
|------|------|
| `scripts/pose2vid.py` | 추론 진입점, 여기서 시작 |
| `src/models/unet_3d.py` | Motion Module 포함 핵심 모델 |
| `train_stage_2.py` | 전체 학습 파이프라인 |

---

## 3. 실행 방법 (Colab 기준)

### 환경 세팅 시 주의사항
Colab 기본 환경이 Python 3.12라 아래 패키지 버전 충돌 발생 → 수동으로 수정 필요

| 패키지 | requirements.txt 원본 | 수정 버전 |
|--------|----------------------|-----------|
| numpy | 1.23.5 | 1.26.4 |
| onnxruntime-gpu | 1.16.3 | 1.17.0 |
| torch | 2.0.1 | 2.2.0 |
| torchvision | 0.15.2 | 0.17.0 |

### 셀 순서

**셀 1 - 설치**
```python
%cd /content
!git clone https://github.com/MooreThreads/Moore-AnimateAnyone.git
%cd Moore-AnimateAnyone

!apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config -q

!pip install "pip<24.1" -q
!pip install av==11.0.0 -q

!sed -i 's/numpy==1.23.5/numpy==1.26.4/' requirements.txt
!sed -i 's/onnxruntime-gpu==1.16.3/onnxruntime-gpu==1.17.0/' requirements.txt
!sed -i 's/torch==2.0.1/torch==2.2.0/' requirements.txt
!sed -i 's/torchvision==0.15.2/torchvision==0.17.0/' requirements.txt

!pip install -r requirements.txt --ignore-installed av -q
```

**셀 2 - 가중치 다운로드** (수 GB, 시간 걸림)
```python
!python tools/download_weights.py
```

**셀 3 - 예시 파일 확인**
```python
!find . -name "*.jpg" -o -name "*.png" -o -name "*.mp4" | head -20
# 레퍼런스: ./configs/inference/ref_images/anyone-1.png
# 포즈영상: ./configs/inference/pose_videos/anyone-video-1_kps.mp4
```

**셀 4 - config 생성**
```python
import yaml, os

config = {
    "pretrained_base_model_path": "./pretrained_weights/stable-diffusion-v1-5",
    "pretrained_vae_path": "./pretrained_weights/sd-vae-ft-mse",
    "image_encoder_path": "./pretrained_weights/image_encoder",
    "denoising_unet_path": "./pretrained_weights/denoising_unet.pth",
    "reference_unet_path": "./pretrained_weights/reference_unet.pth",
    "pose_guider_path": "./pretrained_weights/pose_guider.pth",
    "motion_module_path": "./pretrained_weights/motion_module.pth",
    "inference_config": "./configs/inference/inference.yaml",
    "test_cases": {
        "./configs/inference/ref_images/anyone-1.png": [
            "./configs/inference/pose_videos/anyone-video-1_kps.mp4"
        ]
    }
}

os.makedirs("configs/prompts", exist_ok=True)
with open("configs/prompts/my_animation.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

**셀 5 - 추론 실행**
```python
!python -m scripts.pose2vid \
    --config ./configs/prompts/my_animation.yaml \
    -W 512 \
    -H 784 \
    -L 32   # 프레임 수 (줄이면 빠름)
```

**셀 6 - 결과 확인 및 다운로드**
```python
from IPython.display import Video, display
from google.colab import files
import glob

output_files = glob.glob("output/**/*.mp4", recursive=True)
display(Video(output_files[-1], embed=True))
files.download(output_files[-1])
```

---

## 4. 아키텍처 구성 및 입출력 흐름

### 전체 파이프라인

```
레퍼런스 사진 ──→ ReferenceNet ──→ 외형 feature map
                                        ↓ Cross-Attention
포즈 영상 ──→ DWPose ──→ 스틱맨 ──→ Pose Guider ──→ 포즈 조건
                                        ↓
순수 노이즈 ──────────────────→ Denoising UNet
                                    ↑ (16~24프레임 묶음)
                               Motion Module
                               (Temporal Attention)
                                        ↓
                                완성된 영상 프레임들
```

### 각 컴포넌트 입출력

**DWPose**
- 입력: 일반 영상 프레임 (RGB 이미지)
- 처리: YOLOX로 사람 감지 → 관절 위치 추출 (.onnx 모델)
- 출력: 18개 관절 좌표 → 스틱맨 이미지

**ReferenceNet**
- 입력: 레퍼런스 인물 사진 1장 (512x768)
- 처리: UNet 인코더 구조로 특징 추출
- 출력: 외형 feature map (옷, 얼굴, 체형 정보가 담긴 벡터)

**Pose Guider**
- 입력: DWPose가 만든 스틱맨 이미지
- 처리: 경량 CNN
- 출력: 포즈 조건 벡터 → Denoising UNet에 더해짐(add)

**Denoising UNet (핵심)**
- 입력: 노이즈 이미지 + ReferenceNet feature + Pose 조건 + CLIP embedding
- 처리: Cross-Attention으로 외형 반영, 노이즈 제거 50스텝 반복
- 출력: 깨끗한 이미지 프레임

**Motion Module**
- 입력: UNet이 처리 중인 N개 프레임의 feature map
- 처리: Temporal Attention으로 프레임 간 관계 학습
- 출력: 시간적으로 일관성 있게 조정된 feature map

### Attention 메커니즘 (핵심 개념)
```
Query  : 나는 뭘 찾고 있나? (노이즈 이미지의 픽셀)
Key    : 나는 어떤 정보를 갖고 있나? (레퍼런스의 각 부분)
Value  : 실제 정보 내용

→ Query와 Key 유사도 계산 → 유사도 비율대로 Value 가져옴
→ 이게 "참고한다"는 것의 실체
```

| Attention 종류 | 역할 |
|----------------|------|
| Self-Attention | 이미지 내부 픽셀끼리 참조 |
| Cross-Attention | 노이즈 이미지가 레퍼런스 참조 |
| Temporal Attention | 프레임끼리 시간축으로 참조 |

---

## 5. 학습 과정

### 학습 데이터
- 인터넷에서 수집한 인물 댄스 영상
- 별도 라벨링 없이 자기 자신이 정답 (self-supervised)

```
하나의 영상에서:
├── 랜덤 프레임 1장 → 레퍼런스 이미지 (입력)
├── 전체 프레임의 포즈 추출 → 포즈 시퀀스 (조건)
└── 전체 프레임 → 타겟 영상 (정답)
```

### 2단계 학습 구조

**Stage 1 - 공간적 외형 학습**
- ReferenceNet + Pose Guider만 학습
- Motion Module 없음
- 인물 외형을 포즈에 맞게 그리는 법 학습

**Stage 2 - 시간적 움직임 학습**
- Motion Module 추가
- Stage 1 가중치 불러와서 이어서 학습
- 프레임 간 자연스러운 움직임 학습

### 손실 함수
```
정답 프레임 → 노이즈 추가 → 모델이 노이즈 예측 → 예측 오차로 학습

Loss = ||실제 노이즈 - 모델이 예측한 노이즈||²
```
- 원본 이미지를 직접 맞추는 게 아님
- "내가 추가한 노이즈가 뭔지"를 맞추는 것을 학습
- 레퍼런스 이미지는 조건으로만 사용 (Cross-Attention)

### 프레임 처리 방식
```
포즈 영상 전체를 16~24프레임 묶음으로 처리
[1~16프레임] → 한 번에 처리
[9~24프레임] → 한 번에 처리 (슬라이딩 윈도우)
...
레퍼런스 사진은 모든 프레임에 동일하게 적용 (고정)
````

6. 겪었던 오류 모음

실제로 겪은 오류들. 같은 오류 나면 여기서 바로 찾기.

오류 1: av 빌드 실패
error: subprocess-exited-with-error
× Getting requirements to build wheel did not run successfully.
원인: FFmpeg 개발 헤더 없음
해결:
python!apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config -q
!pip install av==11.0.0

오류 2: numpy 빌드 실패
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
원인: numpy==1.23.5가 Python 3.12 미지원
해결:
python!sed -i 's/numpy==1.23.5/numpy==1.26.4/' requirements.txt

오류 3: onnxruntime-gpu 버전 없음
ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu==1.16.3
원인: Python 3.12용 onnxruntime-gpu는 1.17.0부터 지원
해결:
python!sed -i 's/onnxruntime-gpu==1.16.3/onnxruntime-gpu==1.17.0/' requirements.txt

오류 4: torch 버전 없음
ERROR: Could not find a version that satisfies the requirement torch==2.0.1
원인: torch==2.0.1이 Python 3.12 미지원, 2.2.0부터 지원
해결: torch + torchvision 같이 바꿔야 함
python!sed -i 's/torch==2.0.1/torch==2.2.0/' requirements.txt
!sed -i 's/torchvision==0.15.2/torchvision==0.17.0/' requirements.txt

오류 5: torchsde 메타데이터 오류
WARNING: Ignoring version 0.2.5 of torchsde since it has invalid metadata
ERROR: Could not find a version that satisfies the requirement torchsde==0.2.5
원인: pip 버전이 너무 높아서 torchsde 메타데이터를 거부함
해결:
python!pip install "pip<24.1" -q
!pip install torchsde==0.2.5 -q

오류 6: xformers 충돌
ERROR: Cannot install torch==2.2.0 and xformers==0.0.22 because these package versions have conflicting dependencies.
원인: xformers==0.0.22가 torch==2.0.1 전용
해결:
python!sed -i 's/xformers==0.0.22/xformers==0.0.25/' requirements.txt

오류 7: PositionNet import 오류
ImportError: cannot import name 'PositionNet' from 'diffusers.models.embeddings'
원인: diffusers 버전이 너무 높아서 구버전 API 없어짐
해결:
python!pip install diffusers==0.24.0 -q

오류 8: hf_cache_home import 오류
ImportError: cannot import name 'hf_cache_home' from 'huggingface_hub.constants'
원인: huggingface_hub 버전이 너무 높음
해결: huggingface_hub 직접 낮추면 연쇄 충돌 발생 → transformers 버전 올려서 해결
python!sed -i 's/transformers==4.30.2/transformers==4.38.0/' requirements.txt

오류 9: tokenizers 빌드 실패
ERROR: Failed building wheel for tokenizers
ERROR: Could not build wheels for tokenizers
원인: transformers==4.30.2가 요구하는 tokenizers 버전이 Rust 컴파일 필요
해결: transformers 버전 올리고 tokenizers는 이미 설치된 것 사용
python!pip install -r requirements.txt --ignore-installed av tokenizers transformers -q

WARNING vs ERROR 구분
메시지의미WARNING경고만, 넘어가도 됨ERROR반드시 해결해야 함DEPRECATION나중에 없어질 거라는 예고, 지금은 괜찮음
자주 나오는 무시해도 되는 경고:
WARNING: Error parsing dependencies of torchsde: .* suffix can only be used...
→ torchsde 메타데이터 경고, 설치는 정상적으로 됨

## 참고 링크
- 레포: https://github.com/MooreThreads/Moore-AnimateAnyone
- HuggingFace 데모: https://huggingface.co/spaces/xunsong/Moore-AnimateAnyone
- AnimateAnyone 원논문: https://arxiv.org/pdf/2311.17117.pdf
