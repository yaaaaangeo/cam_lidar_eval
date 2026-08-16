# Changelog

이 프로젝트의 주요 변경 사항은 모두 여기에 기록됩니다.
이 프로젝트는 [Semantic Versioning](https://semver.org/)을 따릅니다.

## [Unreleased]

### 변경 (Changed)

- **Rig Geometry 섹션에 정적/인터랙티브 토글 추가, 인터랙티브 씬 용량
  최적화.** "Interactive 3D View"를 별도 섹션 대신 "Rig Geometry"
  섹션에 합치고, 정적 프러스텀 PNG와 인터랙티브 3D 뷰 사이를 전환하는
  Static/Interactive 토글 버튼을 추가했습니다.
  - 기본값은 Static(가벼운 PNG)이고, Interactive 탭을 처음 클릭할
    때에만 `Plotly.newPlot()`을 지연 실행합니다 -- 리포트를 열 때마다
    무조건 3D 씬을 렌더링하지 않아도 되므로 페이지 로드 체감이
    가벼워집니다(단, JSON 데이터 자체는 처음부터 페이지에 포함되어
    있으므로 파일 크기 자체는 줄지 않고, 렌더링 시점만 지연됩니다).
  - `visualization/interactive_viewer.py`의 좌표를 소수점 3자리로
    반올림하고, 포인트 색상을 `"rgb(r,g,b)"` 대신 `"#rrggbb"` 헥스
    문자열로 바꾸고, 인터랙티브 씬의 기본 포인트 수를 15,000 →
    6,000으로 낮췄습니다. `report/html.py`에서 씬 JSON도
    `json.dumps(..., separators=(",", ":"))`로 공백 없이 직렬화합니다.
    데모 리포트 기준 `report.html` 크기가 약 3.21MB &rarr; 2.48MB로
    (~23%) 줄었습니다. 여전히 가장 큰 비중은 vendored plotly.js
    gl3d 번들(~1.7MB)이며, 이는 인터랙티브 뷰를 오프라인에서 완전히
    동작시키기 위한 고정 비용입니다.

### 추가 (Added)

- **인터랙티브 3D 뷰어 (`visualization/interactive_viewer.py` +
  `report/vendor/`).** 컬러라이즈드 포인트클라우드와 카메라
  프러스텀을 하나의 회전/줌 가능한 3D 씬으로 결합해 `report.html`에
  내장했습니다.
  - `visualization.colorized_pointcloud.colorize_lidar_points()`와
    `visualization.camera_frustum.compute_frustum_geometry()`를
    재사용해 데이터를 만들고, 순수 Python dict(JSON 직렬화 가능)로
    반환합니다. HTML/JS 렌더링은 전적으로 `report/html.py`의 책임으로
    분리되어 있습니다.
  - plotly.js의 "gl3d" 파셜 번들(scatter3d/mesh3d만 포함, 전체
    번들 4.9MB 대비 1.7MB)을 npm에서 받아 `report/vendor/`에
    직접 vendoring했습니다. CDN 대신 vendoring한 이유는, 이 도구의
    핵심 사용처인 CI 파이프라인이 보통 외부 네트워크 접근이 막혀
    있어서 `report.html`이 오프라인에서도 완전히 동작해야 하기
    때문입니다. `pyproject.toml`의 `package-data`에도 등록해 `pip
    install` 배포판에도 포함됩니다.
  - `app/cli.py`의 `--no-visuals`/`--json-only`와 동일한 가드를
    받으므로, CI에서 게이트만 빠르게 돌리고 싶을 때는 이 무거운
    번들이 전혀 생성/포함되지 않습니다.

- **BEV + 카메라 이미지 듀얼 패널 (`visualization/bev_dual_panel.py`).**
  카메라 이미지(왼쪽)와 라이다 포인트를 위에서 내려다본 bird's-eye
  view(오른쪽, X vs 깊이 Z)를 나란히 배치하고, 같은 edge 포인트를 두
  뷰에서 동일한 GOOD/WARNING/BAD 색으로 강조하는 시각화를
  추가했습니다. 오차 히트맵이 "이미지 프레임의 어디"를 보여준다면,
  이 시각화는 "실제 3D 공간의 어디(거리/좌우)"에서 오차가
  집중되는지를 보여줘서, 오차가 거리에 비례해 커지는지(translation
  스케일 문제) 아니면 한쪽으로 치우치는지(회전/장착각 문제) 구분하는
  단서가 됩니다.
  - `EdgeAlignmentResult`가 2D 픽셀/오차만 보관하고 3D 위치는 버리기
    때문에, M2가 이미 수행한 것과 동일한 투영 + edge 추출
    (`evaluation.edge_alignment.extract_lidar_edge_points`)을
    똑같은 파라미터로 다시 실행해 각 edge 포인트의 라이다 프레임 3D
    좌표를 복원합니다. 파라미터가 M2 호출 때와 다르면 포인트 개수가
    어긋나므로 이를 감지해 `None`을 반환하도록 가드를 두었습니다
    (`render_bev_dual_panel_from_result`는 `edge_kwargs`를 그대로
    전달받아 이 문제를 원천적으로 피합니다).
  - 왼쪽 카메라 패널은 `overlay.py`의 `render_overlay()`를 그대로
    재사용해 포인트별 색상 기준이 다른 시각화들과 항상 일치합니다.
  - `app/cli.py`와 `report/html.py`의 M2 섹션에 연결되어,
    `--no-visuals`/`--json-only`가 아닌 한 `report.html`에 자동으로
    포함됩니다.

- **3D 카메라 프러스텀 오버레이 (`visualization/camera_frustum.py`).**
  LiDAR 좌표계 3D 뷰에 extrinsic 기준 카메라의 위치와 시야각(frustum)을
  그려서, translation/rotation 숫자 대신 두 센서의 물리적 배치를 한눈에
  보여주는 요약 이미지를 추가했습니다. 카메라가 라이다 뒤에 있거나
  엉뚱한 방향을 보고 있는 등의 명백한 문제는 YAML의 숫자 6개보다 이
  그림 한 장으로 훨씬 빨리 알아챌 수 있습니다.
  - `T_CL`(camera_from_lidar)의 역변환으로 카메라 원점/축을 라이다
    프레임으로 옮기고, intrinsics의 fx/fy로부터 근사한 수평/수직
    FOV로 피라미드형 frustum을 그립니다(왜곡까지 반영한 정밀한 형상이
    아니라 "대략 어디를, 어느 방향을" 전달하는 개략도이므로 fisheye도
    핀홀로 근사).
  - frustum 깊이는 기본적으로 대표 프레임 라이다 포인트 깊이의 75
    백분위수로 자동 조정되어(직접 `depth_m` 지정도 가능), 씬 스케일에
    맞춰 그려집니다.
  - 씬 감을 잡을 수 있도록 라이다 포인트를 옅게 컨텍스트로 함께
    뿌립니다(밀집 클라우드는 결정론적으로 서브샘플링).
  - `app/cli.py`와 `report/html.py`에 연결되어, `report.html` 상단
    (품질 점수 요약 바로 다음, M2/M3/M4보다 앞)에 "Rig Geometry"
    섹션으로 항상 포함됩니다(다른 시각화들과 달리 M2 성공 여부와
    무관하게 extrinsic만 있으면 항상 그려집니다).

- **오차 히트맵 오버레이 (`visualization/error_heatmap.py`).** 이미지를
  그리드로 나누고 M2의 픽셀 오차를 셀별로 공간적으로 집계해,
  GOOD/WARNING/BAD 색상을 반투명하게 이미지 위에 덧씌우는 시각화를
  추가했습니다. `overlay.py`가 포인트 하나하나의 오차를 보여준다면,
  이 시각화는 "프레임의 어느 영역에서 캘리브레이션이 약한가"를
  보여줍니다 — 가장자리/모서리에서만, 혹은 특정 방향에서만 오차가
  커지는 패턴은 균일한 오정렬이 아니라 특정 원인(왜곡 모델 부족,
  주점에서 먼 곳에서만 드러나는 작은 회전 오프셋 등)을 가리키는
  단서가 됩니다.
  - `overlay.py`의 `_COLOR_BGR` 색상 팔레트와 `quality.noise_floor`의
    floor(Z) 기준 GOOD/WARNING/BAD 임계값을 그대로 재사용해, 포인트별
    색과 셀별 색이 같은 기준으로 해석됩니다. 임계값 경계에서는 색이
    이산적으로 바뀌지 않고 연속적으로 보간됩니다.
  - 포인트 수가 `min_points_per_cell`(기본 3개) 미만인 셀은 색을
    칠하지 않고 그대로 둡니다 — 단일 포인트의 노이즈로 셀 전체를
    판정하지 않기 위함입니다.
  - `app/cli.py`의 파이프라인과 `report/html.py`의 M2 섹션에 연결되어,
    `--no-visuals`/`--json-only`가 아닌 한 `report.html`에 자동으로
    포함됩니다.

- **컬러라이즈드 포인트클라우드 시각화 (`visualization/colorized_pointcloud.py`).**
  LiDAR 포인트를 카메라 이미지에 투영해서 해당 픽셀의 RGB 색을 입힌
  카메라-LiDAR 퓨전 뷰를 추가했습니다. 캘리브레이션이 틀리면 물체
  경계에서 LiDAR 포인트가 반대쪽 색을 샘플링해 색이 번지거나 밀려
  보이는데, 이는 M2가 픽셀 단위로 측정하는 것과 같은 종류의 오차를
  한눈에 보여주는 시각적 근거가 됩니다.
  - 기존 M0/M2가 이미 쓰는 `geometry.projection.project_lidar_to_image`를
    그대로 재사용하므로, "어떤 포인트가 유효한가"의 기준이 도구 전체와
    항상 일치합니다.
  - 3D 씬(카메라 프레임 기준)과 위에서 내려다본 bird's-eye view를 한
    PNG에 나란히 렌더링합니다 — 3D 각도 하나만으로는 잘 안 보이는
    정렬 오차도 위에서 보면 바로 드러나기 때문입니다.
  - 밀집한 포인트클라우드는 `max_points`(기본 60,000)로 결정론적
    서브샘플링해 렌더링 속도와 PNG 용량을 관리합니다.
  - `app/cli.py`의 파이프라인과 `report/html.py`의 M2 섹션에 연결되어,
    `--no-visuals`/`--json-only`가 아닌 한 `report.html`에 자동으로
    포함됩니다.

- **rosbag / rosbag2 소스 지원.** `input/lidar.py`의
  `load_lidar_from_rosbag()`와 `input/camera.py`의
  `load_camera_from_rosbag()`가 더 이상 `NotImplementedError`를 내지
  않습니다. 순수 Python 라이브러리인 `rosbags`(+ 카메라용
  `rosbags-image`)를 이용해 rosbag1(`.bag`)과 rosbag2(디렉토리) 둘 다
  읽습니다 — **실제 ROS/rclpy 설치는 필요 없습니다**. 새 optional
  dependency 그룹으로 분리되어 있어(`pip install
  "cam-lidar-eval[rosbag]"`), `image_dir`/`pcd_dir`만 쓰는 기존
  사용자는 아무 영향이 없습니다.
  - LiDAR: `sensor_msgs/msg/PointCloud2` 메시지를 읽어 x/y/z(+intensity)
    필드를 추출합니다. `PointField`의 offset/datatype 정보로 구조를
    직접 파싱하므로(기존 PCD binary 리더와 같은 방식) `sensor_msgs_py`
    같은 ROS 전용 포인트클라우드 라이브러리가 필요 없습니다.
  - Camera: `sensor_msgs/msg/Image` / `CompressedImage` 메시지를
    `rosbags.image.message_to_cvimage()`로 BGR OpenCV 배열로 변환합니다.
  - 프레임 timestamp는 메시지의 `header.stamp`를 우선 사용하고,
    unstamped(0으로 채워진) 메시지는 bag의 자체 기록 시각으로 대체하며
    이 사실을 warning으로 명시적으로 남깁니다(조용히 근사치를 쓰지
    않음 — `quality/noise_floor.py`의 fallback-warning 관례와 동일한
    원칙).
  - Topic을 지정하지 않아도 해당 메시지 타입의 topic이 하나뿐이면
    자동으로 선택되고, 여러 개면 사용 가능한 topic 목록과 함께
    ValueError를 냅니다.
  - **`--config` YAML에 `camera.source`/`lidar.source: rosbag` 선택자
    추가.** `rosbag_path`(+선택적 `topic`)를 지정하면 됩니다. `camera`와
    `lidar`를 서로 다른 소스로 섞어 쓰는 것도 가능합니다(예: 카메라는
    rosbag, LiDAR는 `pcd_dir`). `source` 키를 생략하면 기존 동작
    (`image_dir`/`pcd_dir`)과 완전히 동일해서 기존 config 파일은 수정
    없이 그대로 동작합니다.
  - **여전히 지원하지 않는 것: 살아있는 `ros_topic` 구독.** bag 파일을
    읽는 것과 실행 중인 ROS 노드를 구독하는 것은 근본적으로 다른
    문제입니다 — 후자는 실시간 ROS2 미들웨어/DDS 연결이 필요하고,
    `rosbags` 라이브러리(오프라인 bag 리더)로는 해결할 수 없는
    영역이라 이 툴의 스코프 밖으로 남겨뒀습니다.
  - 검증: `rosbags.rosbag2.Writer`로 만든 합성 bag으로 왕복 테스트(포인트
    좌표/intensity/이미지 픽셀/타임스탬프 정확히 일치 확인), 그리고
    **동일한 합성 장면을 `pcd_dir`/`image_dir`로 읽은 결과와 `rosbag`으로
    읽은 결과가 완전히 동일한 Overall Quality를 내는지** 비교하는
    end-to-end 테스트로 두 로딩 경로의 동등성을 확인했습니다. CI의
    `test` job도 `pip install -e ".[rosbag]"`로 바꿔서 이 코드가 매
    PR마다 실제로 검증되도록 했습니다.
  - `tests/test_lidar.py` +8, `tests/test_camera.py` +4,
    `tests/test_cli.py` +5 (총 20개 테스트 파일, 276개 테스트로 증가).

### CI / 개발 도구 (CI / Tooling)

- **CI에 lint job 추가.** `.github/workflows/ci.yaml`에 `ruff check .`를
  돌리는 별도 `lint` job을 추가했습니다(Python 버전 매트릭스와 무관하게
  한 번만 실행). `[tool.ruff]` 설정은 의도적으로 pyflakes 동급 규칙셋
  (`F`)만 켰습니다 — import 정렬, `Optional[X]` → `X | None` 같은
  스타일 규칙까지 켜면 이 코드베이스의 기존 컨벤션과 충돌해서 175개
  가까이 걸리는데, 그건 아무도 안 읽는 노이즈가 될 뿐입니다. 이 작업
  과정에서 `tests/` 아래 미사용 import 8개(이전 세션에서 정리할 때
  `tests/`를 스캔 대상에서 빼먹었음)를 추가로 찾아 정리했습니다.
  `ruff>=0.6`을 `dev` extras에 추가.
- **CI Python 매트릭스에 3.13 추가.** `["3.10", "3.11", "3.12", "3.13"]`.
  numpy/scipy/matplotlib/PyYAML은 PyPI에 cp313 휠이 이미 있고,
  opencv-python은 `cp37-abi3`(stable ABI) 태그로 배포돼 있어 3.13에서도
  그대로 설치됩니다 — `pip download --python-version 313`으로 직접
  확인.
- **CI에 pip 의존성 캐싱 추가.** `actions/setup-python`의 `cache: pip`
  옵션을 lint/test 두 job 모두에 적용해 매 실행마다 numpy/opencv/scipy/
  matplotlib를 새로 받는 시간을 줄였습니다.
- **`--version` 플래그 추가.** `importlib.metadata`로 `pyproject.toml`의
  실제 설치된 버전을 동적으로 읽어오므로(하드코딩된 버전 문자열 없음)
  둘이 어긋날 일이 없습니다.
- **`--json-only` 플래그 추가.** `report.html` 생성(및 그 안에 들어갈
  overlay/trajectory/histogram 시각 자료 생성)을 완전히 건너뛰고
  `report.json`만 씁니다. `--no-visuals`는 이미지 임베드만 건너뛰고
  HTML 자체는 항상 만들었는데, `--fail-on-bad`/`--fail-on-partial`
  게이트로만 쓰는 CI 파이프라인에서는 아무도 열어보지 않는 HTML을
  만드는 시간 자체가 아깝기 때문입니다. `--fail-on-bad`/
  `--fail-on-partial`과 자유롭게 조합 가능합니다.
- `python -m app.cli --help`에 `--config` YAML 스키마 전체가 epilog로
  표시되도록 이미 지난 커밋에서 추가되어 있었는데, 여기에 `--version`/
  `--json-only` 플래그도 함께 반영되었습니다.

### 문서 (Documentation)

- **`evaluation_metric_spec.md` 복원.** 코드베이스 전체(10개 넘는 소스
  파일의 docstring)에서 "설계 근거"로 인용되지만 저장소에는 커밋된 적이
  없던 문서를, 그 인용들을 취합해 실제 구현과 정확히 일치하도록
  재구성해 추가했습니다. 문서에 적힌 모든 threshold/상수는 실제 코드
  값과 하나하나 대조 검증했습니다. 원본이 아니라 복원본이라는 점을
  문서 맨 위와 README에 명시했습니다.

### 개선 (Improved)

- **`--config` YAML에 스키마 검증 추가, 에러 메시지를 실행 가능하게
  개선.** 이전에는 필수 키가 하나만 빠져도 `error: failed to load
  dataset: 'camera'`처럼 raw `KeyError`가 그대로 노출되어, 무엇이
  문제인지는 알려줘도 어디를 봐야 하는지는 알려주지 않았습니다.
  이제 `load_dataset_from_config()`가 로더를 실행하기 전에 스키마
  전체를 한 번에 검사해서, **발견된 모든 문제를 한 번에 나열**하고
  (첫 번째 문제에서 멈추지 않음) 스키마 문서(`--help`의 epilog,
  `app/cli.py` docstring, `evaluation_metric_spec.md`) 위치를 안내하는
  `ConfigSchemaError`를 raise합니다. 검사 항목: 필수 top-level/nested
  키 존재 여부, 컨테이너 타입(mapping이어야 하는 곳에 list가 온 경우
  등), `rotation_format`/`parent`/`child` 같은 enum류 필드의 허용값
  검증.
  - `python -m app.cli --help`에 이제 `--config` YAML 스키마 전체가
    epilog로 표시됩니다(이전에는 모듈 docstring에만 있어서 `--help`
    로는 확인할 수 없었습니다).
  - 값 자체의 타당성(예: `image_dir`가 실제 존재하는 디렉토리인지,
    intrinsics가 물리적으로 그럴듯한지)까지는 검사하지 않습니다 —
    그건 각 로더가 이미 하고 있고, 이미 읽을 만한 에러를 냅니다.
    이 검증은 "필수 키가 있는가/구조가 맞는가"만 다룹니다.

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
