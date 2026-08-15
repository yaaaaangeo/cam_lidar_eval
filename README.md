# Cam–LiDAR Calibration Evaluation Tool

**이미 존재하는** camera–LiDAR extrinsic calibration(`T_CL`)의 품질을
GT(Ground Truth) 없이 평가하는 툴입니다. 이 툴은 새로운 calibration을
계산하지 않습니다 — 이미 갖고 있는 calibration을 얼마나 신뢰할 수 있는지,
GT 없이 알려줍니다.

> 전체 설계 근거는 [`evaluation_metric_spec.md`](evaluation_metric_spec.md)에
> 있습니다(코드 전반의 docstring·주석에 흩어져 있던 근거를 취합해
> 복원한 문서입니다 — 자세한 경위는 그 문서 맨 위 안내 참고). 이 README는
> "어떻게 실행하는지 / 어떻게 만들어졌는지"를 다루는 실전 가이드입니다.

---

## 1. 왜 GT-free인가?

실제 운영 환경에서는 비교할 ground-truth `T_CL`을 거의 가질 수 없습니다.
GT 없이도 측정 가능한 것들:

- 이 calibration이 *지금* LiDAR 구조를 이미지 edge에 맞게 투영하는가? (**M2**)
- 여러 시간대에 걸쳐 일관되게 유지되는가, 아니면 특정 장면에서만 우연히 좋았던 건가? (**M3**)
- 프레임별 정확도가 안정적인가, 아니면 예측 불가능하게 튀는가? (**M4**)
- M2~M4를 신뢰하기 전에, 애초에 투영 자체가 구조적으로 말이 되는가? (**M0**)

여기서 다루는 모든 것은 "이 T가 일관되고 안정적인가"에 답할 뿐, "이 T가
몇 mm/도 단위로 정확한가"에는 답하지 않습니다 — 후자는 ground truth가
필요한 질문이며, 의도적으로 스코프 밖에 둡니다.

---

## 2. 설치

```bash
git clone https://github.com/yaaaaangeo/cam_lidar_eval.git
cd cam_lidar_eval
pip install -e .
```

`numpy`, `opencv-python`, `scipy`, `matplotlib`, `PyYAML`이 설치되고,
`cam-lidar-eval` 콘솔 명령어가 추가됩니다. GPU도, 외부 서비스도
필요 없습니다.

rosbag(rosbag1 `.bag` / rosbag2 디렉토리) 소스로 데이터를 읽으려면
추가로 `pip install -e ".[rosbag]"`을 쓰세요 — 이건 순수 Python
bag 리더(`rosbags`/`rosbags-image`)라서 **실제 ROS/rclpy 설치는
여전히 필요 없습니다**. `--config`의 `camera.source`/`lidar.source`를
`rosbag`으로 지정하면 사용할 수 있습니다 (§3 참고). 단, 살아있는
ROS 노드를 구독하는 `ros_topic` 소스는 지원하지 않습니다 — 그건
녹화된 bag 파일을 읽는 것과 달리 실시간 ROS2 미들웨어/DDS 연결이
필요한, 근본적으로 다른 문제라서 이 툴의 스코프 밖입니다.

editable install 없이 순수 pip만 쓰고 싶다면 `pip install -r requirements.txt`도
가능합니다 (이 경우 `cam-lidar-eval` 명령어 대신 레포 루트에서
`python -m app.cli ...`로 실행).

---

## 3. Quickstart

```bash
# 데이터 없이 — 내장된 합성(synthetic) 장면으로 바로 실행
cam-lidar-eval --demo --output-dir out/

# calibration이 시퀀스 중간부터 나빠지는 시나리오
cam-lidar-eval --demo --scenario drift --output-dir out/

# out/report.html을 브라우저로 열기
```

(pip로 설치하지 않았다면 `python -m app.cli ...`로 실행하세요.)

실제 데이터 대상:

```bash
cam-lidar-eval --config my_config.yaml --output-dir out/
```

전체 YAML config 스키마(camera intrinsics/distortion, LiDAR sensor spec,
extrinsic, evaluation 파라미터)는 `python -m app.cli --help`의 맨 아래
epilog나 `app/cli.py`의 모듈 docstring을 참고하세요. 필수 키가
빠지거나 형식이 잘못된 경우 어떤 필드가 문제인지 전부 나열하고 스키마
위치를 안내하는 에러 메시지가 뜹니다(raw `KeyError`가 아닙니다). 최소
예시:

```yaml
camera:
  image_dir: /data/images
  width: 1920
  height: 1080
  model: pinhole
  intrinsics: {fx: 1400.0, fy: 1400.0, cx: 960.0, cy: 540.0}
  distortion: {model: plumb_bob, coeffs: {k1: -0.1, k2: 0.02, p1: 0.0, p2: 0.0}}

lidar:
  pcd_dir: /data/pointclouds
  sensor_spec:
    horizontal_resolution_deg: 0.2
    channels: 32
    vertical_fov_deg: 40.0
    range_accuracy_m: 0.02

extrinsic:
  parent: lidar
  child: camera
  translation: [0.12, -0.04, 0.15]
  rotation: [0.5, -1.2, 90.1]
  rotation_format: rpy_deg

evaluation:
  n_blocks: 4
  min_frames_per_block: 30
  min_frames_m4: 30
  depth_jump_threshold_m: 0.3
  edge_radius_px: 3.0
```

`image_dir`/`pcd_dir` 대신 rosbag(rosbag1 `.bag` 또는 rosbag2 디렉토리)에서
바로 읽고 싶다면 `source: rosbag`을 지정하세요 (`pip install -e ".[rosbag]"`
필요):

```yaml
camera:
  source: rosbag
  rosbag_path: /data/my_recording.bag   # 또는 rosbag2 디렉토리
  topic: /camera/image_raw              # 생략 가능(이미지 topic이 하나뿐이면)
  width: 1920
  height: 1080
  model: pinhole
  intrinsics: {fx: 1400.0, fy: 1400.0, cx: 960.0, cy: 540.0}

lidar:
  source: rosbag
  rosbag_path: /data/my_recording.bag
  topic: /lidar/points                  # 생략 가능(PointCloud2 topic이 하나뿐이면)
  sensor_spec:
    horizontal_resolution_deg: 0.2
```

`camera`와 `lidar`를 서로 다른 소스로 섞어 쓸 수도 있습니다(예: 카메라는
rosbag, LiDAR는 `pcd_dir`). 프레임 timestamp는 메시지의 `header.stamp`를
우선 쓰고, unstamped(0으로 채워진) 메시지는 bag 기록 시각으로 대체하며
그 사실을 warning으로 남깁니다.

유용한 플래그: `--fail-on-bad` (overall quality가 BAD/FAIL이면 exit code
0이 아니게 — CI용), `--fail-on-partial` (M2/M3/M4 중 하나라도 완전히
FAIL해서 Overall Quality 계산에서 제외됐다면 exit code 0이 아니게 —
overall quality 자체는 나머지 카테고리만으로 GOOD/WARNING이 나올 수
있으므로 `--fail-on-bad`와는 별개로 필요; CI용), `--advanced` (Phase-5
진단도 함께 실행), `--no-visuals` (이미지 생성 생략, 훨씬 빠름),
`--json-only` (`report.html` 자체를 생성하지 않고 `report.json`만 —
CI 게이트 용도로 HTML이 필요 없을 때 `--no-visuals`보다 더 가벼움),
`--weights geometry=0.5,...` (카테고리 가중치 오버라이드), `--frame-index
N` (M2의 대표값을 어느 프레임에서 가져올지 지정), `--version` (설치된
버전 확인).

---

## 4. 아키텍처

```
              Existing System
                    │
        ┌───────────┴───────────┐
      Camera                  LiDAR
        │                       │
        └──────────┬────────────┘
                    ▼
             Existing T_CL
                    │
                    ▼
      ┌──────────────────────────┐
      │ M0 Sanity Gate           │   evaluation/sanity_gate.py
      │ (게이트, 점수 아님)       │
      └──────────────┬───────────┘
                      ▼
   ┌─────────────────────────────────────┐
   │         Evaluation Engine           │
   │  M2 Geometry │ M3 Generalization    │   evaluation/edge_alignment.py
   │              │ M4 Stability         │   evaluation/holdout_consistency.py
   └───────┬──────┴──────────┬───────────┘   evaluation/multiframe_consistency.py
           ▼                 ▼
   quality/noise_floor.py → floor(Z), 센서 상대적 threshold
   quality/normalization.py → 0-100 점수 곡선, floor(Z) 배수에 고정
   quality/quality_score.py → Geometry/Generalization/Stability → Overall Quality
                      │
                      ▼
        visualization/  (overlay, trajectory, histogram — PNG로 임베드)
                      │
                      ▼
             report/ (JSON + 단일 HTML 파일)
                      │
                      ▼
                 app/cli.py  (전체를 하나로 묶는 진입점)

   (선택 사항, --advanced)
   evaluation/plane_consistency.py   — 지배적 평면의 경계 정합성
   evaluation/perturbation.py        — T_CL이 local minimum 근처에 있는가?
   evaluation/temporal_drift.py      — 시퀀스에 걸쳐 에러가 추세를 보이는가?
```

### 디렉토리 구조

```
cam_lidar_eval/
├── input/                    원본 camera/LiDAR/extrinsic 데이터를 표준 모델로 파싱
│   ├── camera.py               CameraModel, image_dir 로더
│   ├── lidar.py                LidarModel, PCD/PLY 로더 (직접 구현한 파서)
│   ├── extrinsic.py            T_CL/T_LC 정규화 + sanity check (회전 유효성, 단위)
│   └── dataset.py              Camera↔LiDAR 타임스탬프 동기화, M3용 time_blocks()
│
├── geometry/                  순수 수학, I/O 없음
│   ├── transform.py             SE(3): rpy/quaternion→matrix, 합성, 역변환, transform_points
│   └── projection.py            Pinhole/fisheye 투영, LiDAR→이미지 투영 파이프라인
│
├── evaluation/                 실제 metric들
│   ├── sanity_gate.py           M0 — pass/fail 게이트, 점수화 안 됨
│   ├── edge_alignment.py        M2 — LiDAR edge point vs 이미지 edge (distance transform)
│   ├── holdout_consistency.py   M3 — 연속 시간 블록에 걸친 동일 T 평가
│   ├── multiframe_consistency.py M4 — 프레임별 에러 안정성, outlier 검출
│   ├── plane_consistency.py     [advanced] 지배적 평면 경계 정합성
│   ├── perturbation.py          [advanced] T_CL을 살짝 흔들어 local minimum 여부 확인
│   └── temporal_drift.py        [advanced] M4 프레임별 시퀀스에 대한 선형 추세 검정
│
├── quality/                    px 측정값을 판단으로 변환
│   ├── noise_floor.py           floor(Z): 센서 상대적 "이론상 최선"의 px 불확실성
│   ├── normalization.py         floor(Z)에 고정된 0-100 점수 곡선
│   └── quality_score.py         Geometry/Generalization/Stability → Overall Quality
│
├── visualization/               HTML 리포트용 PNG 생성
│   ├── overlay.py                투영된 LiDAR 점을 이미지 위에 GOOD/WARNING/BAD 색상으로 표시
│   ├── trajectory.py             M4 프레임별 에러 라인 차트 (matplotlib)
│   └── histogram.py              M2 per-point 에러 분포 (matplotlib)
│
├── report/                     JSON + HTML 리포트 생성
│   ├── builder.py                모든 결과를 하나의 plain-dict 리포트 구조로 조립
│   ├── json.py                   NaN-safe한 엄격한 JSON 직렬화
│   └── html.py                   단일 파일로 완결되는 다크 테마 HTML 리포트 (base64 이미지 임베드)
│
├── app/
│   └── cli.py                   진입점: config/demo → pipeline → report → 콘솔 요약
│
├── tests/                      20개 파일에 걸친 276개 테스트 (§7 참고)
│
├── pyproject.toml               패키지 메타데이터, 의존성, `cam-lidar-eval` 콘솔 스크립트
├── requirements.txt             `pip install -e .`의 순수 pip 대안
├── run_tests.sh                 전체 테스트 스위트 실행, CI 친화적 exit code
├── .github/workflows/ci.yaml     GitHub Actions: lint(ruff) + Python 3.10-3.13 install/test/smoke test
└── LICENSE                      MIT
```

---

## 5. Metric 레퍼런스

| | 무엇을 측정하는가 | 점수화? | 카테고리 |
|---|---|---|---|
| **M0** Sanity Gate | T_CL + 데이터 조합이 애초에 구조적으로 말이 되는가 (FOV coverage, depth sanity, occlusion plausibility)? | 아니오 — pass/fail 게이트 | 데이터 품질 |
| **M2** Edge Alignment | 투영된 LiDAR 구조가 지금 실제 이미지 edge에 맞는가? | 예 | Geometry |
| **M3** Hold-out Consistency | 동일한 T_CL이 서로 다른 연속 시간 구간에서 일관되게 동작하는가? | 예 | Generalization |
| **M4** Multi-frame Consistency | 프레임별 에러가 안정적인가, 특정 프레임만 튀는가? | 예 | Stability |
| **Plane Consistency** *(advanced)* | 지배적인 평면(바닥/벽)의 외곽선이 이미지 실루엣과 일치하는가? | 아니오 | Geometry (보조) |
| **Perturbation Sensitivity** *(advanced)* | T_CL이 에러 곡면의 local minimum 근처에 있는가, 살짝 흔들면 더 나아지는가? | 아니오 | Sensitivity |
| **Temporal Drift** *(advanced)* | 시퀀스에 걸쳐 에러가 (노이즈가 아니라) 방향성 있는 추세를 보이는가? | 아니오 | Stability (보조) |

### 센서 상대적 noise floor: `floor(Z)`

이 툴의 모든 threshold(GOOD/WARNING/BAD 경계, 0-100 점수 곡선)는 절대
픽셀값이 아니라, *특정* camera+LiDAR 조합이 주어진 거리에서 가질 수
있는 이론상 최선의 px 불확실성인 `floor(Z)`의 배수로 표현됩니다:

```
floor(Z) = sqrt(floor_angular² + floor_range(Z)² + floor_edge²)

floor_angular       = fx · θ_res                    (LiDAR 각해상도)
floor_range(Z)      = fx · baseline · σ_r / Z²       (LiDAR range 노이즈, 거리 제곱에 반비례)
floor_edge          ≈ 0.5 px                          (edge detector의 sub-pixel 한계)
```

즉 16채널 LiDAR + 4K 카메라 조합과 128채널 LiDAR + VGA 카메라 조합이
하나의 고정된 픽셀 기준이 아니라 *각자에 맞게 스케일된* threshold를
갖게 됩니다. 전체 유도 과정과, 센서 스펙을 완전히 모를 때의 fallback
규칙은 `quality/noise_floor.py`를 참고하세요.

### 0-100 점수화

점수는 classification 경계에 정확히 고정된 곡선을 사용하므로, 점수와
classification이 서로 모순될 수 없습니다:

```
score(r) = 100 / (1 + (r / warning_mult)^p),   r = value_px / floor_px

r = 0            → 100
r = good_mult     → 80   (GOOD/WARNING 경계)
r = warning_mult  → 50   (WARNING/BAD 경계)
r → ∞             → 0
```

배수 스킴별로 500개의 랜덤 ratio에 대한 property-based test
(`tests/test_normalization.py`)로 검증됨.

### Quality Score 집계

Geometry(M2) / Generalization(M3) / Stability(M4)는 기본적으로 동일한
가중치(각 1/3)를 가집니다 — 아직 하나를 더 중요하게 볼 데이터 기반
근거가 없고, 가중치는 하드코딩된 상수가 아니라 파라미터입니다(CLI의
`--weights`). 한 카테고리의 metric이 완전히 FAIL하면, 그 카테고리는
제외되고 나머지 가중치가 재정규화됩니다 (조용히 0점 처리하지 않음 —
"측정 못함"과 "측정했더니 최악"은 다른 것이므로).

단, 카테고리가 하나라도 제외된 채로 계산된 Overall Quality는 남은
카테고리 점수가 아무리 좋아도 classification이 **WARNING을 넘지
못하도록 캡**됩니다 (숫자 점수 자체는 캡하지 않음). 예를 들어 M2/M3가
FAIL하고 M4만 100점이어도 Overall Quality는 "100.0 WARNING"이지
"100.0 GOOD"이 아닙니다 — 부분적으로만 평가된 결과가 완전히 평가된
결과와 같은 신뢰도로 보이면 안 되기 때문입니다. CI에서 이런 "부분
평가" 자체를 실패로 처리하고 싶다면 `--fail-on-partial`을 쓰세요.

---

## 6. 리포트 출력

매 실행마다 다음이 생성됩니다:

- **`report.json`** — 기계가 읽을 수 있는, 엄격하게 유효한 JSON
  (NaN/Inf는 `null`로 sanitize, `allow_nan=False`를 안전장치로 강제).
  CI 파이프라인, 대시보드, 추가 툴링을 위한 용도.
- **`report.html`** — 단일 파일로 완결되는 리포트 (이미지가 base64로
  임베드되어 있어 다른 파일을 함께 가져갈 필요 없음). 다크 "계기판"
  테마: overall-quality 게이지(순수 CSS `conic-gradient`, JS 없음),
  GOOD/WARNING/BAD/FAIL 뱃지를 전체에 일관되게 사용, metric별 상세
  테이블, 그리고 (`--no-visuals`가 아니라면) 실제 LiDAR-온-이미지
  overlay, M4 에러 추이 차트, M2 에러 히스토그램까지 포함.

---

## 7. 테스트

```bash
./run_tests.sh                            # 모든 테스트 파일 실행, 결과 집계
python3 tests/test_noise_floor.py         # 또는 개별 파일 직접 실행
```

`run_tests.sh`는 뭔가 실패하면 exit code가 0이 아니게 되며,
`.github/workflows/ci.yaml`이 매 push/PR마다 (Python 3.10-3.13) 실행하는
것과 정확히 동일한 스크립트입니다. CI는 이와 별개로 `ruff check .`를
돌리는 `lint` job도 하나 더 실행합니다(pyflakes 동급 규칙만 켜져
있음 — `dev` extras의 `ruff`로 로컬에서도 동일하게 돌릴 수 있습니다:
`pip install -e ".[dev]" && ruff check .`).

pytest 의존성 불필요 — 모든 테스트 파일은 자체 러너
(`if __name__ == "__main__":`)를 내장하고 있어 테스트별 PASS/FAIL을
출력하고 실패 시 exit code가 0이 아니게 됩니다. 따라서
`python3 tests/test_X.py`는 단독으로도, CI에서도 그대로 동작합니다.

**20개 파일에 걸친 276개 테스트**, 전부 통과 (전체 실행 약 2~3분):

| 파일 | 테스트 수 | 커버리지 |
|---|---|---|
| `test_transform.py` | 18 | SE(3) 수학 |
| `test_projection.py` | 13 | Pinhole/fisheye 투영 |
| `test_camera.py` | 12 | Camera 로더 (rosbag 포함) |
| `test_lidar.py` | 19 | PCD/PLY 파서, LiDAR 로더 (rosbag 포함) |
| `test_extrinsic.py` | 11 | 회전 포맷, T_CL/T_LC 방향 처리 |
| `test_dataset.py` | 9 | 타임스탬프 동기화, time_blocks() |
| `test_noise_floor.py` | 21 | floor(Z) 유도 및 fallback 규칙 |
| `test_normalization.py` | 21 | 점수 곡선, 500-sample property test 포함 |
| `test_edge_alignment.py` | 15 | M2, 합성 depth-step 장면 포함 |
| `test_holdout_consistency.py` | 8 | M3, drift 검출 시나리오 포함 |
| `test_multiframe_consistency.py` | 9 | M4, outlier 검출 포함 |
| `test_sanity_gate.py` | 10 | M0, occlusion-violation 검출 포함 |
| `test_plane_consistency.py` | 10 | 평면 피팅 + 경계 정합성 |
| `test_perturbation.py` | 7 | Local-minimum 검출 |
| `test_temporal_drift.py` | 9 | 추세 회귀, 유의성 게이팅 |
| `test_quality_score.py` | 14 | 카테고리 집계, 가중치 처리, partial-result WARNING 캡 |
| `test_report.py` | 20 | JSON/HTML 생성, NaN 안전성, visuals 임베딩 |
| `test_visualization.py` | 13 | Overlay/trajectory/histogram 렌더링 |
| `test_m0_report_integration.py` | 2 | M0 → report end-to-end |
| `test_cli.py` | 35 | Demo 모드, config 로딩(rosbag 포함), 전체 파이프라인, exit code |

모든 MVP metric(M2/M3/M4)은 **known, controllable한 ground truth를 가진
합성 장면**(알려진 `T_CL` 아래에서 그려진 이미지 edge와 정확히 일치하도록
배치한 depth step)으로 검증됩니다. 즉 테스트는 코드가 크래시 없이
돌아간다는 것뿐 아니라, calibration을 흔들었을 때 실제 숫자가 *올바른
방향*으로 움직이는지까지 검증합니다.

개발 과정에서 이 방식으로 실제 버그 여러 개를 잡았고, 고친 지점에
코드 주석으로 남겨뒀습니다: `floor(Z)`의 worst-case-axis 선택 버그,
초기 합성 장면의 원근(perspective) row-shift 아티팩트, perturbation
테스트 장면의 grid-quantization 민감도 아티팩트.

---

## 8. 알려진 한계 / 구현하지 않은 것

- **Re-calibration repeatability** (스펙 §13의 "Level 2": subset별로
  calibration을 재수행해서 결과 T들을 비교)는 명시적으로 스코프 밖입니다
  — 이 툴은 *기존* calibration을 평가할 뿐, 새로 계산하지 않습니다.
  구현하려면 사실상 별도의 camera-LiDAR calibration 알고리즘 툴을 새로
  만드는 셈이라, 이 프로젝트의 정체성(GT-free *평가* 툴)과 충돌합니다.
- **Photometric consistency**는 원 설계 노트에 따라 보류했습니다
  (illumination/exposure/reflectance가 calibration 품질과 너무 쉽게
  섞여버려서 첫 버전에는 부적합). 이건 "아직 안 만듦"이 아니라 "충분한
  근거 없이 만들면 신뢰도 낮은 metric이 나올 위험이 있다"는 의도적
  판단이라, 억지로 구현하기보다 이 판단을 유지합니다.
- **GT mode** (실제 ground-truth transform 대비 정확도, 연구/벤치마크용)는
  스펙에 별도 모드로 설명되어 있지만 여기서는 구현하지 않았습니다 —
  이 툴은 GT-free 모드 전용입니다.
- Advanced metric(`--advanced`)들은 진단용이며 **quality_score에 영향을
  주지 않습니다** — MVP 셋만큼 검증되지 않았고, headline 점수에 기여하기보다
  다른 질문(local optimality, 추세, 단일 표면 체크)에 답하기 때문에
  의도적으로 분리했습니다. (이건 한계가 아니라 의도된 설계입니다 —
  advanced metric이 quality_score에 영향을 주기 시작하면 검증 수준이
  다른 두 metric 집합이 섞여버립니다.)
