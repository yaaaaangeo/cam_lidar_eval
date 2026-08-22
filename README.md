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

콘솔 출력 예시(STEP1 Input Validation, STEP2 Synchronization이 calibration
metric들보다 먼저 표시됩니다):

```
------------------------------------------------------
 Cam-LiDAR Calibration Quality
------------------------------------------------------
 Synchronization
   Matched frames        : 982 / 1000
   Mean Δt               : +17.4 ms
   Offset std            : 2.1 ms
   Drop ratio             : 1.8%
   Status                 : GOOD
------------------------------------------------------
 M0 Sanity Gate          : PASS
 Geometry (M2)           : 86.7 / 100     [GOOD]
 Generalization (M3)     : 100.0 / 100    [GOOD]
 Stability (M4)          : 100.0 / 100    [GOOD]
------------------------------------------------------
 OVERALL QUALITY         : 95.6 / 100     [GOOD]
------------------------------------------------------
```

입력 자체가 깨진 경우("Calibration BAD"가 아니라 "INPUT INVALID"로
표시되고, 파이프라인은 실행되지 않고 exit code 5로 종료됩니다):

```
error: INPUT INVALID:
  - camera.timestamps_monotonic: Camera timestamps are not monotonic
    (frames are out of time order, or contain duplicate timestamps).
```

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

STEP5 (Motion Deskew, opt-in — quality score에는 전혀 영향 없는 순수
진단용): `--deskew-linear-velocity vx,vy,vz` / `--deskew-angular-velocity
wx,wy,wz` (LiDAR body-frame, m/s / rad/s) 중 하나라도 주면 headline
프레임에 대해 constant-velocity deskew를 실행하고 리포트에 "Motion
Deskew" 섹션(전/후 BEV 오버레이 + correction 히스토그램)을 추가합니다.
`--deskew-scan-period-s`(기본 0.1), `--deskew-reference-time-s`(기본:
scan 중간 지점), `--deskew-clockwise`(azimuth 스윕 방향, per-point
timestamp가 없을 때의 근사에만 영향)로 세부 조정 가능:

```bash
cam-lidar-eval --demo \
  --deskew-linear-velocity 10.0,0.0,0.0 \
  --deskew-angular-velocity 0.0,0.0,0.2 \
  --output-dir out/
```

STEP6(M2 correspondence matching)과 STEP7(per-point noise/uncertainty
model)은 별도 플래그 없이 **M2의 기본 동작 자체**입니다. 리포트의 M2
섹션에 `Match rate (STEP6)`(orientation/strength/local consistency를
모두 통과해서 실제로 매칭된 point 비율)와 `Mean normalized error`
(actual error / 그 point의 depth에서 기대되는 sensor noise) 통계, 그리고
depth별 오차를 floor(Z) 곡선과 함께 보여주는 산점도가 자동으로 추가되어
있습니다. 기존 방식(단순 nearest-distance)이 필요하면
`evaluate_edge_alignment(..., use_correspondence_matching=False)`로
직접 호출하세요 (CLI 플래그로는 아직 노출되지 않음).

STEP9(Depth / Spatial Analysis)도 별도 플래그 없이 **항상 계산**됩니다
(STEP5/STEP8과 달리 추가 입력이 필요 없어서). 리포트에 M2 섹션 바로 다음
"Depth / Spatial Analysis" 섹션이 추가되어, 단일 mean_px 대신 depth
bin(0-10m/10-20m/20-30m/30-50m/50m+)별과 카메라 영역
(LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM — 두 개의 독립적인 축)별
mean/median/p95/std/valid·failure count를 보여주고, depth bin 평균이
단조 증가/감소하는지 자동 판정한 문구도 함께 표시됩니다.

STEP10(M3/M4 Robustness 개선)도 별도 플래그 없이 **M3/M4의 기본 동작
자체**입니다. M3는 각 block에 depth/edge density/point count/FOV
coverage를 함께 기록해서, block 하나가 유독 나쁘면 다른 block들과 scene
특성을 비교해 "Long-range scenes" 같은 원인 후보를 자동으로 제시합니다
(`diagnose_instability`). M4는 기존 "5×median" 대신 **MAD/IQR/Hampel
기반 robust 통계를 기본값**으로 사용하고(`--outlier-method`는 아직 CLI
플래그로 노출되지 않았고 `evaluate_multiframe_consistency(...,
outlier_method="multiplier"|"hampel"|"iqr")`로 직접 호출 시 선택 가능),
valid ratio/failure ratio/outlier ratio를 각각 분리해서 리포트합니다.

STEP11(Calibration Sensitivity Analysis)은 기존 `--advanced`(Phase-5 진단)
안에 포함되어 있습니다. `--advanced`를 주면 "Perturbation Sensitivity"
섹션에 roll/pitch/yaw(±0.05~1.0°), tx/ty/tz(±1~20mm) 각 축을 스펙 그대로의
간격으로 흔들어보고 M2 오차 변화를 측정해서, floor(Z) 상대 기준으로
HIGH/MEDIUM/LOW sensitivity 랭킹을 막대그래프로 보여줍니다(local-minimum
판정과는 별개로 항상 계산됨). Timestamp 축(±5~100ms)은 STEP5의
`motion.deskew` 인프라를 재사용해서 계산하는데, 플랫폼 속도(선속도/각속도)를
알아야만 의미가 있어서 `evaluate_perturbation_sensitivity(...,
linear_velocity_mps=..., angular_velocity_rps=...)`를 직접 호출할 때만
계산됩니다 (CLI 플래그로는 아직 노출 안 됨; 안 주면 timestamp 축은 생략되고
리포트에 그 이유가 명시됩니다).

STEP12(⭐ Root Cause Diagnosis Engine)는 별도 플래그 없이 **항상 계산**됩니다
(STEP9와 마찬가지로 추가 입력이 필요 없어서 — 다만 STEP8/STEP11처럼 opt-in인
데이터가 있으면 그것도 같이 활용합니다). Quality score 바로 아래에
"⭐ Root Cause Diagnosis" 섹션이 추가되어, 지금까지의 모든 진단(sync/M2/
M3/M4/depth·spatial analysis/dynamic ratio/sensitivity)을 rule 기반으로
교차 검증해서 "Yaw misalignment — HIGH", "Dynamic object contamination —
MEDIUM" 같은 순위가 매겨진 원인 후보 목록을 만듭니다. AI/ML은 전혀 쓰지
않고 IF-THEN 규칙만 사용합니다(스펙의 명시적 권고). 근거가 충분하지 않으면
confidence를 낮게 매기거나(예: 공간적 비대칭 없이 sensitivity만 HIGH인
경우 "unconfirmed"로 LOW 처리) 아예 후보를 만들지 않습니다 — 확신 없는
진단을 강요하지 않는 걸 우선했습니다. 후보가 하나도 없으면 섹션 자체가
리포트에서 생략됩니다.

STEP13(Quality/Confidence/Coverage 분리)도 별도 플래그 없이 **항상
계산**됩니다. 기존 Overall Quality 점수 계산 로직은 전혀 안 건드리고,
그 옆에 두 개의 새 0-100 점수를 추가합니다: **Confidence**(이번 측정
"과정" 자체를 얼마나 믿을 수 있는가 — sync 품질, M2 match rate, M3 유효
block 비율, M4 valid ratio, input validation 상태의 평균)와
**Coverage**(센서의 depth/FOV 범위를 실제로 얼마나 커버했는가 — STEP9의
depth bin 5개·카메라 영역 6개 중 몇 개에 데이터가 있었는지, STEP10 M3의
평균 FOV coverage). 둘 다 기존 Quality와 같은 80/50 GOOD/WARNING/BAD
경계선을 공유해서 세 숫자를 나란히 비교할 수 있습니다. 같은 Quality
점수라도 Confidence/Coverage가 다르면 완전히 다른 의미라는 게 스펙의
핵심 요지입니다 — 예를 들어 `--demo`를 그냥 돌리면 Quality는 GOOD인데
Coverage는 BAD로 나오는데(단일 depth·좁은 화면 영역만 테스트했으므로),
이건 실제로 발생하는 정직한 신호입니다. 각 카드의 "why?"를 펼치면 어떤
구성 요소가 점수를 만들었는지 볼 수 있습니다.

STEP14(Visualization / HTML Report)는 스펙이 요구하는 5가지 최소 구성
(① Projection Overlay ② Error Heatmap ③ Depth Error ④ Parameter
Sensitivity ⑤ Diagnosis)이 이미 STEP3·STEP7/9·STEP11·STEP12에서 각각
만들어져서 하나의 HTML 리포트에 전부 통합되어 있습니다. 이번 단계에서
새로 채운 갭은 ⑤ Diagnosis 패널 하나입니다 — 스펙의 예시(`🔴 Yaw
misalignment / 🟠 Tx misalignment / 🟢 Timestamp OK / 🟢 Sensor quality
OK`)는 문제와 확인된 정상 항목을 **같은 목록에 섞어서** 보여주는데,
STEP12 시점의 구현은 문제만 보고하고 있었습니다. 이제 sync GOOD,
dynamic contamination 무시할 만함, M3 block-to-block 일관성 GOOD,
모든 축 sensitivity LOW 같은 확인 사항도 🟢 OK로 같은 "Root Cause
Diagnosis" 테이블에 함께 표시됩니다 — 문제도 확인 사항도 하나도 없을
때만 섹션 자체가 생략됩니다.

STEP15(Benchmark / Regression Test, 로드맵의 마지막 단계)는 새 프로덕션
기능이 아니라 `evaluation/benchmark.py` + `tests/test_benchmark.py`로 구현된
**벤치마크/회귀 테스트 스위트**입니다. 스펙의 두 가지 요구사항을 그대로
구현했습니다:

1. **Monotonicity 벤치마크**: 정답 T_CL을 알고 있는 synthetic scene에서
   `evaluation/perturbation.py`의 `_perturb_rotation`/`_perturb_translation`/
   `_perturb_timestamp_points`(STEP11)를 재사용해서, roll/pitch/yaw
   (스펙 예시 그대로 0°→0.1°→0.2°→0.5°) + tx/ty/tz + timestamp(0/10/20/50/100ms)
   전부에 대해 M2 오차가 정말로 단조 증가하는지 확인합니다. **실제로 검증하면서
   발견한 사실**: 하나의 장면이 모든 축을 잘 보여주지 못합니다 — 단일 수직
   경계선 장면은 roll에는 잘 맞지만 pitch/tx는 서브픽셀 이산화 잡음으로
   비단조적이었고, 반대로 여러 개의 near/far 줄무늬가 있는 장면은 pitch/yaw/
   tx/ty/tz에는 깨끗하게 단조 증가했지만 roll에서는 edge point 수 자체가
   불안정해졌습니다. 이건 숨기지 않고 테스트에 그대로 문서화했습니다 — STEP11의
   "축마다 민감도가 다르다"는 개념이 장면 기하학에도 그대로 적용된다는,
   벤치마크가 아니었으면 몰랐을 정직한 발견입니다.
2. **Known-cause → correct-diagnosis 벤치마크**: STEP12 자체 테스트는 가벼운
   fake 객체로 룰 하나하나를 검증하지만, STEP15는 **mock을 전혀 쓰지 않고**
   실제 STEP8 dynamic contamination 시나리오(진짜 `evaluate_edge_alignment`
   + 진짜 `compare_with_without_dynamic_filtering`)와 실제 STEP2 sync
   엔진(진짜 timestamp 어긋남이 있는 프레임 시퀀스 → 진짜 `classify_sync`)을
   끝까지 돌려서 `diagnose_root_cause`의 1위 후보가 실제로
   `DYNAMIC_CONTAMINATION`/`TEMPORAL_OFFSET`으로 나오는지 확인합니다 —
   "알려진 문제 → 시스템 → 올바른 진단?"을 문자 그대로 구현한 것입니다.

STEP8(Dynamic Object Filtering, opt-in — quality score에는 전혀 영향
없는 순수 진단용): `--dynamic-filter`를 주면 headline 프레임 주변
`--dynamic-filter-window`(기본 5)개 프레임에 걸쳐 multi-frame motion
consistency로 각 (ring, azimuth) 셀을 STATIC/DYNAMIC/UNKNOWN으로
분류하고, M2를 "overall"(필터 없음)과 "static only"(움직이는 물체로
분류된 point 제외) 두 번 계산해서 비교합니다. **중요한 전제**: 이
방법은 센서 플랫폼이 프레임 윈도우 동안 거의 정지해 있다는 가정에
의존합니다 (움직이는 플랫폼에서는 정적 장면 전체가 "dynamic"처럼
보여서 이 방법 자체가 무의미해집니다 — 진짜 ego-motion 보정이 없으면
풀 수 없는 문제이고, 스펙 자체도 이걸 "나중에"로 미뤄둔 부분입니다).
이미 자체 object detector/tracker 결과가 있다면
`evaluation.dynamic_filter.apply_external_dynamic_mask()`로 직접 마스크를
넣을 수 있습니다 (CLI 플래그로는 아직 노출되지 않음).

```bash
cam-lidar-eval --demo --dynamic-filter --output-dir out/
```

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
      │ Input Validation (STEP1) │   input/validation.py
      │ INPUT_VALID/WARNING/     │   Calibration 평가 전에 raw 입력
      │ INVALID 게이트           │   자체(NaN/Inf, timestamp 순서,
      └──────────────┬───────────┘   빈 PCD, 잘못된 K 등)를 걺
                      ▼
      ┌──────────────────────────┐
      │ Timestamp Sync (STEP2)   │   input/dataset.py
      │ candidate window +       │   단순 nearest-neighbor 대신
      │ monotonic matching +     │   후보 윈도우 + 단조 매칭 +
      │ offset 추정              │   Δt(camera-lidar clock) 추정/보정
      └──────────────┬───────────┘
                      ▼
      ┌──────────────────────────┐
      │ Projection 검증 (STEP3)  │   geometry/projection.py (unit test)
      │ 좌표변환 unit test +      │   visualization/projection_overlay.py
      │ depth-colored overlay    │   M2 없이도 "투영이 말이 되는가"를
      └──────────────┬───────────┘   먼저 눈으로 확인
                      ▼
      ┌──────────────────────────┐
      │ LiDAR Ring/Topology      │   geometry/range_image.py
      │ (STEP4)                 │   ring×azimuth range image +
      │ 구조 기반 native edge    │   LiDAR-native depth-discontinuity
      └──────────────┬───────────┘   추출 (image-projection 무관)
                      ▼
      ┌──────────────────────────┐
      │ Motion Deskew (STEP5)    │   motion/deskew.py
      │ constant-velocity 모델   │   visualization/deskew_comparison.py
      │ (opt-in, 진단 전용)      │   --deskew-* 로 플랫폼 속도 입력 시만
      └──────────────┬───────────┘   활성화, quality score엔 영향 없음
                      ▼
      ┌──────────────────────────┐
      │ M0 Sanity Gate           │   evaluation/sanity_gate.py
      │ (게이트, 점수 아님)       │
      └──────────────┬───────────┘
                      ▼
   ┌─────────────────────────────────────┐
   │         Evaluation Engine           │   evaluation/edge_alignment.py
   │  M2 Geometry │ M3 Generalization    │   (M2 correspondence matching:
   │              │ M4 Stability         │    STEP6 evaluation/edge_correspondence.py)
   └───────┬──────┴──────────┬───────────┘   evaluation/holdout_consistency.py
           ▼                 ▼                (STEP10: scene metadata + instability
                                                diagnosis — "Long-range scenes" 등)
                                                evaluation/multiframe_consistency.py
                                                (STEP10: MAD/IQR/Hampel robust 통계,
                                                valid/failure/outlier ratio 분리)
   quality/noise_floor.py → floor(Z), 센서 상대적 threshold
                              (STEP7: compute_floor_array — point별 개별 depth에서
                               noise floor 계산 → normalized_error = actual/expected,
                               visualization/uncertainty_plot.py로 시각화)
   quality/normalization.py → 0-100 점수 곡선, floor(Z) 배수에 고정
   quality/quality_score.py → Geometry/Generalization/Stability → Overall Quality
                      │
                      ▼
      ┌──────────────────────────┐
      │ Depth/Spatial Analysis   │   evaluation/spatial_analysis.py
      │ (STEP9, 항상 계산됨)     │   visualization/spatial_analysis_plot.py
      │ depth bin × 카메라 영역  │   단순 평균 대신 depth bin별/
      └──────────────┬───────────┘   LEFT·CENTER·RIGHT·TOP·BOTTOM별 breakdown
                      ▼
      ┌──────────────────────────┐
      │ Dynamic Filtering (STEP8)│   evaluation/dynamic_filter.py
      │ multi-frame motion       │   visualization/dynamic_filter_overlay.py
      │ consistency (opt-in)     │   --dynamic-filter 로 M2 overall vs
      └──────────────┬───────────┘   static-only 비교, quality score엔 영향 없음
                      ▼
      ┌──────────────────────────┐
      │ ⭐ Root Cause Diagnosis   │   evaluation/root_cause.py
      │ Engine (STEP12)          │   sync/M2/M3/M4/spatial/dynamic/
      │ rule 기반, 항상 계산됨   │   sensitivity를 모두 종합 → ranked
      └──────────────┬───────────┘   HIGH/MEDIUM/LOW 원인 후보 (quality score엔 영향 없음)
                      ▼
      ┌──────────────────────────┐
      │ Quality/Confidence/      │   quality/confidence_coverage.py
      │ Coverage 분리 (STEP13)   │   같은 Quality 점수라도 측정 신뢰도
      │ 항상 계산됨              │   (Confidence)·범위(Coverage)에 따라
      └──────────────┬───────────┘   완전히 다른 의미일 수 있음을 분리해서 표시
                      ▼
      ┌──────────────────────────┐
      │ Benchmark / Regression   │   evaluation/benchmark.py
      │ Test (STEP15)            │   6축(roll/pitch/yaw/tx/ty/tz) + timestamp
      │ tests/test_benchmark.py  │   monotonicity 검증, known-cause →
      └──────────────┬───────────┘   correct-diagnosis end-to-end 벤치마크
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
                                        (STEP11: roll/pitch/yaw/tx/ty/tz/timestamp
                                         축별 HIGH/MEDIUM/LOW sensitivity 랭킹)
   evaluation/temporal_drift.py      — 시퀀스에 걸쳐 에러가 추세를 보이는가?
```

### 디렉토리 구조

```
cam_lidar_eval/
├── input/                    원본 camera/LiDAR/extrinsic 데이터를 표준 모델로 파싱
│   ├── camera.py               CameraModel, image_dir 로더
│   ├── lidar.py                LidarModel, PCD/PLY 로더 (직접 구현한 파서)
│   ├── extrinsic.py            T_CL/T_LC 정규화 + sanity check (회전 유효성, 단위)
│   ├── dataset.py              STEP2 — Timestamp Sync: 후보 윈도우 + 단조 매칭 + Δt 오프셋
│   │                              추정/보정, M3용 time_blocks()
│   └── validation.py           STEP1 — Input Validation: INPUT_VALID/WARNING/INVALID 게이트
│
├── geometry/                  순수 수학, I/O 없음
│   ├── transform.py             SE(3): rpy/quaternion→matrix, 합성, 역변환, transform_points
│   ├── projection.py            Pinhole/fisheye 투영, LiDAR→이미지 투영 파이프라인
│   └── range_image.py           STEP4 — ring×azimuth range image, LiDAR-native depth-discontinuity 추출
│
├── motion/                     STEP5 — LiDAR Motion Deskew (opt-in, 진단 전용)
│   └── deskew.py                 constant-velocity 모델로 per-point 프레임 내 시간 보정,
│                                  정지/이동 전후 비교 (compare_before_after)
│
├── evaluation/                 실제 metric들
│   ├── sanity_gate.py           M0 — pass/fail 게이트, 점수화 안 됨
│   ├── edge_alignment.py        M2 — LiDAR edge point vs 이미지 edge, STEP7 per-point uncertainty
│   ├── edge_correspondence.py   STEP6 — M2 correspondence matching: candidate search + orientation
│   │                              + gradient strength + local consistency (기존 nearest-distance 대체)
│   ├── holdout_consistency.py   M3 — 연속 시간 블록에 걸친 동일 T 평가. STEP10: block별 scene
│   │                              metadata(depth/edge density/point count/FOV coverage) 기록,
│   │                              instability 원인 자동 진단 (diagnose_instability)
│   ├── multiframe_consistency.py M4 — 프레임별 에러 안정성, outlier 검출. STEP10: MAD/IQR/
│   │                              Hampel 기반 robust 통계 (기본값), valid/failure/outlier
│   │                              ratio 분리, per-frame robust z-score
│   ├── spatial_analysis.py      STEP9 — depth bin(0-10/10-20/20-30/30-50/50m+) × 카메라 영역
│   │                              (LEFT/CENTER/RIGHT, TOP/CENTER/BOTTOM)별 mean/median/p95/std/
│   │                              valid·failure count, depth trend 자동 판정 (항상 계산됨)
│   ├── dynamic_filter.py        STEP8 — multi-frame motion consistency로 STATIC/DYNAMIC/UNKNOWN
│   │                              분류, M2 overall vs static-only 비교 (opt-in, --dynamic-filter)
│   ├── root_cause.py            STEP12 ⭐ — sync/M2/M3/M4/spatial/dynamic/sensitivity를 rule 기반으로
│   │                              종합해서 ranked HIGH/MEDIUM/LOW 원인 후보 생성 (AI 없음, 항상 계산됨).
│   │                              STEP14: 🟢 confirmation(OK) — 확인된 정상 항목도 같은 목록에 표시
│   ├── benchmark.py             STEP15 — monotonicity(6축+timestamp) + known-cause→correct-diagnosis
│   │                              벤치마크. tests/test_benchmark.py가 실제 벤치마크 스위트
│   ├── plane_consistency.py     [advanced] 지배적 평면 경계 정합성
│   ├── perturbation.py          [advanced] T_CL을 살짝 흔들어 local minimum 여부 확인.
│   │                              STEP11: 스펙 그대로의 delta grid(rotation ±0.05~1.0°,
│   │                              translation ±1~20mm, timestamp ±5~100ms — STEP5 motion.deskew
│   │                              재사용, 플랫폼 속도 필요), 축별 HIGH/MEDIUM/LOW sensitivity 랭킹
│   └── temporal_drift.py        [advanced] M4 프레임별 시퀀스에 대한 선형 추세 검정
│
├── quality/                    px 측정값을 판단으로 변환
│   ├── noise_floor.py           floor(Z): 센서 상대적 "이론상 최선"의 px 불확실성
│   ├── normalization.py         floor(Z)에 고정된 0-100 점수 곡선
│   ├── quality_score.py         Geometry/Generalization/Stability → Overall Quality
│   └── confidence_coverage.py   STEP13 — Quality와 별개로 Confidence(측정 신뢰도)·
│                                  Coverage(depth/FOV 범위) 0-100 점수 분리 (항상 계산됨)
│
├── visualization/               HTML 리포트용 PNG / 인터랙티브 3D 씬 생성
│   ├── overlay.py                투영된 LiDAR edge point를 이미지 위에 GOOD/WARNING/BAD 색상으로 표시
│   ├── projection_overlay.py     STEP3 — M2 없이도 되는 raw 투영 sanity-check (depth colormap)
│   ├── range_image.py            STEP4 — range image 렌더링 + LiDAR-native edge 하이라이트
│   ├── deskew_comparison.py      STEP5 — deskew 전/후 BEV 오버레이 + correction 히스토그램
│   ├── uncertainty_plot.py       STEP7 — depth vs error 산점도 + floor(Z) 곡선/GOOD·WARNING·BAD 밴드
│   ├── dynamic_filter_overlay.py STEP8 — static(초록)/dynamic(빨강)/unknown(회색) 분류 오버레이
│   ├── spatial_analysis_plot.py  STEP9 — depth bin/수평/수직 3-패널 막대그래프 (mean±std, P95, n/failed)
│   ├── sensitivity_plot.py       STEP11 — 축별 sensitivity 수평 막대그래프 (HIGH/MEDIUM/LOW)
│   ├── error_heatmap.py          이미지를 그리드로 나눠 셀별 평균 오차를 반투명 GOOD/WARNING/BAD 색으로 오버레이
│   ├── colorized_pointcloud.py   LiDAR 포인트를 카메라 픽셀 색으로 칠한 컬러라이즈드 포인트클라우드 (3D + BEV)
│   ├── camera_frustum.py         라이다 좌표계에 카메라 위치·frustum을 그린 rig 배치 개략도
│   ├── bev_dual_panel.py         카메라 이미지 + bird's-eye view를 나란히, 같은 edge point를 색까지 맞춰 강조
│   ├── interactive_viewer.py     컬러라이즈드 포인트클라우드 + frustum을 회전 가능한 Plotly 3D 씬으로 결합
│   ├── trajectory.py             M4 프레임별 에러 라인 차트 (matplotlib)
│   └── histogram.py              M2 per-point 에러 분포 (matplotlib)
│
├── report/                     JSON + HTML 리포트 생성
│   ├── builder.py                모든 결과를 하나의 plain-dict 리포트 구조로 조립
│   ├── json.py                   NaN-safe한 엄격한 JSON 직렬화
│   ├── html.py                   단일 파일로 완결되는 다크 테마 HTML 리포트 (base64 이미지 임베드,
│   │                              인터랙티브 3D 뷰어가 있을 때만 vendored plotly.js 인라인 포함)
│   └── vendor/                   plotly.js gl3d 파셜 번들 (MIT, CDN 대신 오프라인 임베드용으로 vendoring)
│
├── app/
│   └── cli.py                   진입점: config/demo → pipeline → report → 콘솔 요약
│
├── tests/                      44개 파일에 걸친 테스트 (§7 참고)
│
├── pyproject.toml               패키지 메타데이터, 의존성, `cam-lidar-eval` 콘솔 스크립트
├── requirements.txt             `pip install -e .`의 순수 pip 대안
├── run_tests.sh                 전체 테스트 스위트 실행, CI 친화적 exit code
├── .github/workflows/ci.yaml     GitHub Actions: lint(ruff) + Python 3.10-3.12 install/test/smoke test
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
  CI 파이프라인, 대시보드, 추가 툴링을 위한 용도. `input_validation`
  (STEP1)과 `synchronization`(STEP2) 섹션이 최상단에 포함되어, calibration
  metric들과 별개로 "입력 자체가 유효했는가"/"두 센서가 실제로 얼마나 잘
  맞춰졌는가"를 확인할 수 있습니다.
- **`report.html`** — 단일 파일로 완결되는 리포트 (이미지는 base64로,
  인터랙티브 3D 씬이 있을 때만 plotly.js까지 인라인으로 임베드되어
  있어 다른 파일을 함께 가져갈 필요가 없고 오프라인에서도 그대로
  열립니다). 다크 "계기판" 테마: overall-quality 게이지(순수 CSS
  `conic-gradient`), GOOD/WARNING/BAD/FAIL 뱃지를 전체에 일관되게
  사용, metric별 상세 테이블. `--no-visuals`가 아니라면 아래
  시각화까지 자동으로 포함됩니다:

  | 시각화 | 보여주는 것 |
  |---|---|
  | LiDAR-온-이미지 overlay | 투영된 LiDAR edge point를 이미지 위에 GOOD/WARNING/BAD 색으로 표시 |
  | 오차 히트맵 | 이미지를 그리드로 나눠 셀별 평균 오차를 반투명 색으로 오버레이 — 프레임의 어느 영역에서 오차가 집중되는지 |
  | 컬러라이즈드 포인트클라우드 | LiDAR 포인트를 투영된 카메라 픽셀 색으로 칠한 3D + bird's-eye view — 경계에서 색이 번지면 정렬 오차 |
  | 카메라 프러스텀 | 라이다 좌표계에 카메라 위치·시야각을 그린 rig 배치 개략도 — 리포트 상단 요약 |
  | BEV 듀얼 패널 | 카메라 이미지와 bird's-eye view를 나란히, 같은 edge point를 색까지 맞춰 강조 — 거리/좌우별 오차 분포 확인 |
  | 인터랙티브 3D 뷰어 | 컬러라이즈드 포인트클라우드 + 프러스텀을 회전/줌 가능한 Plotly 씬으로 결합 (Rig Geometry 섹션에서 정적 이미지와 토글) |
  | M4 에러 추이 차트 | 프레임별 에러 추이, outlier 표시 |
  | M2 에러 히스토그램 | per-point 에러 분포 |
  | 시퀀스 오버레이 GIF (`--sequence-gif`, opt-in) | M2 overlay를 시퀀스 전체에서 샘플링한 프레임들로 이어붙인 애니메이션 — 정합 품질이 시간에 따라 유지되는지/드리프트하는지 |

### CI 연동

- **`--validate-config`**: 실제 데이터 로딩/평가 없이 `--config` YAML
  스키마만 검사하고 종료 (exit 0/1). pre-commit hook이나 config 변경
  PR에서 빠르게 쓰기 좋습니다.
  ```bash
  cam-lidar-eval --config path/to/config.yaml --validate-config
  ```
- **`--compare-to` / `--fail-on-regression`**: 이전 실행의 `report.json`과
  비교해서 overall/카테고리별 점수·classification이 나빠졌는지 감지합니다.
  ```bash
  cam-lidar-eval --config config.yaml --output-dir out/ \
    --compare-to previous_report.json --fail-on-regression   # 회귀 시 exit 4
  ```
- **`--format github-comment`**: GOOD/WARNING/BAD/FAIL 이모지 + 점수 표를
  GitHub-flavored markdown으로 stdout에 출력합니다. `report.json`/
  `report.html`은 평소대로 그대로 기록됩니다.
  ```bash
  cam-lidar-eval --config config.yaml --output-dir out/ \
    --format github-comment > comment.md
  gh pr comment "$PR_NUMBER" --body-file comment.md
  ```

---

## 7. 테스트

```bash
./run_tests.sh                            # 모든 테스트 파일 실행, 결과 집계
python3 tests/test_noise_floor.py         # 또는 개별 파일 직접 실행
```

`run_tests.sh`는 뭔가 실패하면 exit code가 0이 아니게 되며,
`.github/workflows/ci.yaml`이 매 push/PR마다 (Python 3.10-3.12) 실행하는
것과 정확히 동일한 스크립트입니다. CI는 이와 별개로 `ruff check .`를
돌리는 `lint` job도 하나 더 실행합니다(pyflakes 동급 규칙만 켜져
있음 — `dev` extras의 `ruff`로 로컬에서도 동일하게 돌릴 수 있습니다:
`pip install -e ".[dev]" && ruff check .`).

pytest 의존성 불필요 — 모든 테스트 파일은 자체 러너
(`if __name__ == "__main__":`)를 내장하고 있어 테스트별 PASS/FAIL을
출력하고 실패 시 exit code가 0이 아니게 됩니다. 따라서
`python3 tests/test_X.py`는 단독으로도, CI에서도 그대로 동작합니다.

**44개 파일에 걸친 709개 테스트**, 전부 통과 (전체 실행 약 3분):

| 파일 | 테스트 수 | 커버리지 |
|---|---|---|
| `test_transform.py` | 18 | SE(3) 수학 |
| `test_projection.py` | 23 | Pinhole/fisheye 투영, STEP3 distortion/rotation 단위 테스트 |
| `test_projection_overlay.py` | 8 | STEP3 raw depth-colored 투영 sanity-check 오버레이 |
| `test_range_image.py` | 19 | STEP4 range image, ring 유도, LiDAR-native depth-discontinuity 추출 |
| `test_range_image_visualization.py` | 8 | STEP4 range image 렌더링, edge 하이라이트 |
| `test_deskew.py` | 16 | STEP5 motion deskew — 독립 수치 시뮬레이션 기반 ground-truth 검증 포함 |
| `test_deskew_comparison.py` | 7 | STEP5 deskew 전/후 BEV 오버레이 렌더링 |
| `test_dynamic_filter.py` | 14 | STEP8 multi-frame motion consistency 분류, overall vs static-only 비교 — 스펙 예시 시나리오(contamination) 직접 재현 |
| `test_root_cause.py` | 36 | STEP12 rule 기반 진단 — spec의 3개 예시 룰(temporal offset/yaw misalignment/dynamic contamination) 전부 직접 재현, 실제 다른 모듈 객체와의 통합 검증. STEP14: 🟢 confirmation(OK) 항목 — 문제뿐 아니라 확인된 정상 항목도 함께 보고 |
| `test_benchmark.py` | 20 | STEP15 — 6축(roll/pitch/yaw/tx/ty/tz) + timestamp monotonicity 벤치마크(스펙의 "0.1<0.2<0.5" 예시 재현), dynamic contamination/temporal offset **완전 실제 파이프라인**(mock 없음) known-cause→correct-diagnosis 검증 |
| `test_dynamic_filter_overlay.py` | 6 | STEP8 static/dynamic/unknown 오버레이 렌더링 |
| `test_spatial_analysis.py` | 17 | STEP9 depth bin/카메라 영역별 breakdown — 스펙 워크드 예제(0.8→1.0→1.8→3.9px) 직접 재현 |
| `test_spatial_analysis_plot.py` | 6 | STEP9 depth/수평/수직 3-패널 막대그래프 렌더링 |
| `test_uncertainty_plot.py` | 7 | STEP7 depth vs error 산점도, floor(Z) 곡선/GOOD·WARNING·BAD 밴드 |
| `test_camera.py` | 16 | Camera 로더 (rosbag 포함), `verify_image_shape` 해상도 검증 |
| `test_lidar.py` | 19 | PCD/PLY 파서, LiDAR 로더 (rosbag 포함) |
| `test_extrinsic.py` | 11 | 회전 포맷, T_CL/T_LC 방향 처리 |
| `test_dataset.py` | 17 | STEP2 Timestamp Sync — 후보 윈도우, 단조 매칭, Δt 오프셋 추정/보정, GOOD/WARNING/BAD/FAIL 분류, time_blocks() |
| `test_validation.py` | 28 | STEP1 Input Validation — camera/lidar/dataset 체크, INPUT_VALID/WARNING/INVALID 게이팅 |
| `test_noise_floor.py` | 26 | floor(Z) 유도 및 fallback 규칙, STEP7 compute_floor_array (per-point) |
| `test_normalization.py` | 21 | 점수 곡선, 500-sample property test 포함 |
| `test_edge_alignment.py` | 22 | M2, 합성 depth-step 장면, STEP7 per-point uncertainty 필드, STEP8 dynamic_mask |
| `test_edge_correspondence.py` | 21 | STEP6 M2 correspondence matching — candidate search/orientation/strength/local consistency, distractor 시나리오로 "가깝지만 틀린 edge 거부" 직접 검증 |
| `test_holdout_consistency.py` | 15 | M3, drift 검출 시나리오, STEP10 scene metadata + diagnose_instability("Long-range scenes" 등) |
| `test_multiframe_consistency.py` | 18 | M4, outlier 검출, STEP10 MAD/IQR/Hampel robust 통계, valid/failure/outlier ratio 분리 |
| `test_sanity_gate.py` | 10 | M0, occlusion-violation 검출 포함 |
| `test_plane_consistency.py` | 10 | 평면 피팅 + 경계 정합성 |
| `test_perturbation.py` | 20 | Local-minimum 검출, STEP11 delta grid/axis sensitivity/timestamp axis(motion.deskew 재사용) — 스펙 워크드 예제(Yaw/Tx HIGH 등) 시각 검증 |
| `test_sensitivity_plot.py` | 6 | STEP11 축별 sensitivity 막대그래프 렌더링 |
| `test_temporal_drift.py` | 9 | 추세 회귀, 유의성 게이팅 |
| `test_quality_score.py` | 14 | 카테고리 집계, 가중치 처리, partial-result WARNING 캡 |
| `test_confidence_coverage.py` | 16 | STEP13 Quality/Confidence/Coverage 분리 — 스펙의 두 대조 예시(같은 Quality, 다른 Confidence/Coverage) 정확히 재현 |
| `test_report.py` | 48 | JSON/HTML 생성, NaN 안전성, visuals 임베딩, STEP1/2/3/4/5/7/8/9/10/11/12/13/14 HTML 섹션 표시 여부 |
| `test_report_diff.py` | 7 | 리포트 diff, 회귀 판정 (classification/score 양쪽) |
| `test_markdown.py` | 10 | GitHub 코멘트 markdown, 전 classification UTF-8 인코딩 |
| `test_visualization.py` | 13 | Overlay/trajectory/histogram 렌더링 |
| `test_error_heatmap.py` | 13 | 오차 히트맵 그리드 집계 + 렌더링 |
| `test_colorized_pointcloud.py` | 10 | 컬러라이즈드 포인트클라우드, 해상도 불일치 오류, 깨진 3D 환경 대응 |
| `test_camera_frustum.py` | 15 | Frustum 기하 계산, `auto_frustum_depth`, 깨진 3D 환경 대응 |
| `test_bev_dual_panel.py` | 7 | BEV 듀얼 패널, edge point 재계산 정합성 |
| `test_interactive_viewer.py` | 9 | Plotly 씬 JSON 직렬화, 해상도 불일치 오류 |
| `test_sequence.py` | 7 | 시퀀스 GIF 프레임 샘플링 + 렌더링 |
| `test_m0_report_integration.py` | 2 | M0 → report end-to-end |
| `test_cli.py` | 64 | Demo 모드, config 로딩(rosbag 포함), 전체 파이프라인, exit code, `--validate-config`/`--compare-to`/`--format`/`--sequence-gif`/STEP5 `--deskew-*`/STEP8 `--dynamic-filter` |

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
- **다중 카메라 / 다중 LiDAR rig는 지원하지 않습니다** — `input/`, `geometry/`,
  `evaluation/`, `quality/`, `report/` 전 모듈이 "카메라 1대·LiDAR 1대"를
  전제로 짜여 있습니다. 지원하려면 패치 수준이 아니라 각 모듈의 데이터
  모델부터(현재 `EvaluationDataset`이 `camera: CameraModel` 단수 필드를
  가정) 다시 설계해야 하는 수준이라, 별도 설계 논의 없이는 진행하지
  않기로 했습니다.
