# 그림 출처 및 읽는 방법

| 파일 | 출처·내용 | 기준 |
|---|---|---|
|1-real-hand-rgb-depth.png|D435i 기록 세션의 RGB+정렬 깊이. 원본 SDK frame2571, 입력 source frame673로 대응 확인된 기존 자료|실제 촬영. 손 안정 HOLD 테이프 구간 안의 한 프레임|
|2-isaac-and-top-path-baseline-LOW-HIGH.png|기존 Isaac Sim 비교 녹화+로그 기반 위에서 본 경로 합성|위 step164/2.73s, 아래 step233/3.88s. 표의 최소값은 전체 A→B 구간|
|3a-path-comparison-left-3d.png|관절/EE 로그 재구성|시뮬레이터 스크린샷이 아닌3D 경로 그림|
|control-pipeline.svg|실제 로컬 코드의 연결 구조를 문서용 벡터로 작성|초록=기존 구성, 파랑=프로젝트 연결·추가 구성|
|clearance-and-arrival.png/.svg|scripts/build_figures.py가 evidence/R,LOW,HIGH의 로그에서 생성|ROI 관측면–buffer 제외 근사 구 거리. B 도달은 위치와 자세 동시 판정|
|nvblox-observation-diagnosis.png|기존 R-palm-geometry 분석의 figB-section-y-camera.png|구의 원래 반지름과20mm buffer, 실측 ESDF 관측 공백을 구분|

세 대표 이미지의 원본은 `Downloads/potfoliu`에 사용자가 정리한 파일과 동일하다. 원본 경로와 SHA256는 [provenance.json](data/provenance.json)에 있다. 이미지 변경 없이 해석 캡션을 문서에 추가했다.

## 수치 통일

사진 표시는 원래 충돌 구 기준이며 로그는20mm buffer를 포함한다. 본문 그래프는 사진과 같은 기준으로 재생성했고 CSV에는 두 기준을 모두 담았다. LOW는 원래 구 기준+8.80mm이지만 buffer 포함−11.20mm다.

HIGH 표시는 최종 p=1 민감도 실행, eta73.515982mm이다. p=0.5의42.87073mm 결과와 섞지 않는다. 두 질량값은 가정이며 캡션에서 활성 거리의 실험적 설정임을 명시한다.

근접 구간 전체·손 전체의 안전을 그림만으로 판단하지 않는다. ROI 밖 관측점은 회색으로 표시되지만 대상 전용 비용 입력에는 들어가지 않는다. 기존 nvblox 환경 비용에는 ROI 밖 관측도 반영될 수 있다.

비교 영상 `docs/media/isaac-comparison.mp4`는 기존 Isaac Sim 직접 녹화 합성본(실시간 속도)이며 세 조건의 관절·명령 궤적이 원래 결과와 동일했던 자료다. **영상 자막은 buffer 포함 값**이므로 사진과20mm 차이가 난다.
