# Changelog

이 프로젝트의 주요 변경 사항은 모두 여기에 기록됩니다.
이 프로젝트는 [Semantic Versioning](https://semver.org/)을 따릅니다.

## [Unreleased]

### 성능 개선 (Performance)

- **`extract_lidar_edge_points` (M2의 핵심 연산 경로)를 두 단계에 걸쳐
  벡터화.** 포인트별 neighbor-depth 축약 연산이 파이썬에서 프로젝션된
  LiDAR 포인트 하나하나를 순회하면서 그 포인트의 (작은) neighbor-depth
  배열에 개별적으로 `.max()`/`.min()`을 호출하는 구조였습니다 — 포인트가
  약 3만 개인 합성 테스트 씬 기준, 이는 수만 번의 개별 numpy 축약 호출을
  의미하며, 각 호출은 실제로 하는 일에 비해 numpy의 고정 호출 오버헤드를
  과도하게 지불하고 있었습니다.
  - **1차**: 포인트별 루프를 `cKDTree.query_ball_point`의 포인트별
    neighbor 리스트로부터 만든 하나의 flatten된
    `(point_id, neighbor_depth)` 배열에 대한 `np.maximum.at` /
    `np.minimum.at` 벡터화 축약 연산으로 교체. **약 3배** 빨라짐.
  - **2차**: 1차에서 축약 연산 자체는 벡터화했지만, flatten 단계
    (모든 neighbor 인덱스를 하나씩 순회하는 파이썬 레벨 제너레이터)가
    새로운 병목이 되었음을 확인. `query_ball_point`(ragged 파이썬 리스트
    출력) 대신 `cKDTree.query_pairs(..., output_type="ndarray")`를 사용하도록
    교체 — 반경 내의 모든 포인트 쌍을 파이썬 레벨 순회 없이 C에서 하나의
    배열로 한 번에 계산합니다. 실제 테스트 씬 입력 기준 원본 대비
    **약 11배** 빨라짐 (1차 대비로도 약 3.6배 추가 개선).
  - 두 단계 모두, 교체하기 전에 100회 랜덤 파라미터 조합(포인트 수,
    radius, threshold, min_neighbors)과 `n=0`/`n=1` 엣지 케이스를 통해
    기존 루프 기반 구현과 동일한 결과를 내는지 검증했습니다.
  - 전체 테스트 스위트에 대한 종합적인 효과 (M3/M4/CLI/report 테스트가
    모두 M2를 반복 호출하므로 이 함수가 스위트 실행 시간을 지배함):
    20개 파일 / 253개 테스트 기준 전체 실행 시간이 약 12분에서
    **약 2.3분**으로 단축.
- `README.md`, `pyproject.toml`, `CONTRIBUTING.md`에 남아있던 플레이스홀더
  `YOUR_ORG` GitHub URL을 실제 저장소 주소로 수정하고, `git clone` 이후의
  `cd` 단계도 실제 클론 디렉토리명과 일치하도록 수정.

### 버그 수정 (Fixed)

- **핵심 카테고리 대부분이 완전히 FAIL해도 Overall Quality가 "GOOD"으로
  나올 수 있던 문제.** 점수화 대상 3개 카테고리 중 2개(예: M2 Geometry +
  M3 Generalization)가 FAIL해서 가중평균에서 제외되면, 살아남은 카테고리
  하나의 재정규화된 점수가 여전히 "GOOD"으로 반올림될 수 있었습니다 —
  이는 마치 calibration이 완전히 평가되어 신뢰할 만하다는 오해를 줄 수
  있었습니다. 이제 카테고리가 하나라도 유효하지 않으면
  `overall_classification`은 "WARNING"을 넘지 못하도록 캡됩니다.
  `overall_score` 자체는 캡하지 않아 실제 숫자는 여전히 확인 가능합니다.
  이 캡을 설명하는 새로운 warning이 `report["warnings"]`에 추가됩니다.
- CLI 콘솔 요약이 이제 결과가 부분적일 때 `OVERALL QUALITY` 줄에
  `(n/전체 categories)`를 덧붙입니다 — 이전에는 JSON 리포트의 warning
  목록에서만 확인할 수 있었습니다.

### 추가 (Added)

- **`--fail-on-partial` CLI 플래그**: 최종 classification과 무관하게,
  카테고리 중 하나라도 완전히 FAIL해서 Overall Quality 계산에서 제외됐다면
  exit code를 0이 아니게 만듭니다. `--fail-on-bad` 단독으로는 이 경우를
  잡지 못합니다 (부분 평가 결과는 이제 BAD/FAIL이 아니라 WARNING으로
  캡되므로) — 그래서 세 축(Geometry / Generalization / Stability) 모두가
  측정 가능해야 한다고 요구하는 CI 파이프라인을 위해 이 플래그가
  존재합니다. exit code `3`으로 종료되어 (`--fail-on-bad`의 `2`와는
  다름) CI 로그에서 두 실패 모드를 구분할 수 있습니다.

## [0.1.0] — 최초 릴리스

`evaluation_metric_spec.md`에 따른 MVP + Phase 5 advanced 진단 기능.

### 추가 (Added)

- **입력 로더** (`input/`): 카메라(image_dir), LiDAR(PCD ASCII/binary,
  PLY ASCII), extrinsic(rpy/quaternion/matrix, T_CL/T_LC 방향 자동
  처리), 타임스탬프 동기화 dataset 구성.
- **Geometry** (`geometry/`): SE(3) transform 유틸리티, pinhole/fisheye
  투영.
- **M0 Sanity Gate**: FOV coverage, depth-distribution, occlusion
  plausibility 검사 — 점수화되지 않으며, M2~M4가 의미 있는지를 게이트함.
- **MVP 점수화 metric**:
  - **M2 Edge Alignment** — 프로젝션된 LiDAR depth-discontinuity 포인트
    vs 이미지 edge (Canny + distance transform).
  - **M3 Hold-out Consistency** — 동일한 T_CL을 연속된 시간 블록들에
    걸쳐 평가.
  - **M4 Multi-frame Consistency** — 프레임별 에러 안정성 및 outlier
    검출.
- **센서 상대적 noise floor** (`quality/noise_floor.py`): LiDAR 각해상도,
  range 노이즈, edge detector 한계로부터 유도된 `floor(Z)` — 모든
  GOOD/WARNING/BAD threshold가 고정 픽셀 기준이 아니라 센서 성능에 맞춰
  스케일됨.
- **0-100 점수화** (`quality/normalization.py`): classification 경계에
  정확히 고정된 점수 곡선.
- **Quality Score 집계** (`quality/quality_score.py`): Geometry /
  Generalization / Stability → Overall Quality, 기본 동일 가중치,
  FAIL한 카테고리의 우아한 제외.
- **시각화** (`visualization/`): LiDAR-on-image overlay, M4 에러 추이
  차트, M2 에러 히스토그램.
- **리포트** (`report/`): 엄격한 NaN-safe JSON, 시각 자료가 임베드된
  단일 파일 다크 테마 HTML.
- **CLI** (`app/cli.py`): `--demo`(합성 씬) 및 `--config`(YAML을 통한
  실제 데이터) 진입점, CI용 `--fail-on-bad`.
- **Phase 5 advanced 진단** (`--advanced`로 opt-in, quality_score에
  절대 영향 주지 않음):
  - **Plane Consistency** — RANSAC을 통한 지배적 평면 경계 정합성.
  - **Perturbation Sensitivity** — T_CL이 local error minimum 근처에
    있는가?
  - **Temporal Drift** — M4의 프레임별 시퀀스에 대한 통계적으로 게이팅된
    선형 추세 검정.
- 20개 파일에 걸친 249개 테스트, 각각 독립적으로 실행 가능
  (`python3 tests/test_X.py`, pytest 불필요).
- GitHub Actions CI (`.github/workflows/ci.yaml`), Python 3.10-3.12.

### 알려진 한계 (Known limitations)

- rosbag / ROS topic 소스는 스텁 처리됨 (이 환경에 ROS 역직렬화
  의존성이 없음).
- Re-calibration repeatability, photometric consistency, GT
  (ground-truth) 모드는 이번 릴리스에서 명시적으로 스코프 밖 — README
  §8 참고.
