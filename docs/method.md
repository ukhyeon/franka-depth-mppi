# 방법 및 구현 위치

## 비용 연결

`experiment/base_target_cost.py:TargetCostChannel`은 `mpc.solver.optimizers`의 실제 rollout `cost_fn`에 추가 비용을 연결한다. capture 전에 고정 크기의 대상 world/cache를 만들고, 패치 pose·enable flag·alpha를 갱신한다. 단순 전체 환경 weight 변경이 아니다. 보조 rollout만 바꾸는 구현도 아니다.

- R: 기존 환경·자체 충돌·목표·관절 bound·stop 비용만 사용.
- LOW/HIGH: 동일 기본 비용 + 별도 `WorldPrimitiveCollision`과 `PrimitiveCollisionCost`로 계산한 대상 비용.
- 모든 후보의 로봇 충돌 구를 평가한 뒤 비용을 합산하고 MPPI 총비용·가중치·BEST 선택에 반영.
- 기존 깊이 지도 유지. 대상 패치 팽창 없음. LOW/HIGH는 같은1mm cuboid 관측 패치에 서로 다른 활성 거리를 적용.
- `alpha=1`, 대상 내부 `weight=100000`. `eta`는 근접 비용의 활성 범위와 값·기울기에 영향을 준다.

## 유효질량 가정

`policy.py`의 기본은 p=0.5다. 대표 HIGH는 `hand_experiment.py --eta-exponent 1.0`으로 같은 activation tensor를 capture 전에 갱신한 민감도 실험이다. `scripts/replay.sh HIGH`가 이 옵션을 명시한다.

| 조건 | m_r(kg) | m_h(kg) | 환산질량 mu(kg) | p | eta(m) |
|---|---:|---:|---:|---:|---:|
| LOW |5|0.6|0.5357142857|기본0.5 (p=1도 같은 값)|0.025|
| HIGH 대표|5|2.3|1.5753424658|1|0.073515981735|
| 초기 HIGH 참고|5|2.3|1.5753424658|0.5|0.042870730614|

이는 충돌 모델의 질량 관계에서 착안해 선택한 휴리스틱이다. 후보별 실제 충돌력 계산, Franka 관성행렬 기반 유효질량 산출, 물리적으로 검증된 안전거리 계산을 수행하지 않는다.

## 관측 표현

`target_margin.py:TargetMargin.update`:

1. 깊이 m·카메라 내부 파라미터로 점을 역투영하고 카메라→로봇 pose를 적용한다.
2. `target.json`의 고정 ROI로 점을 선택한다. 사람/손 분류 모델은 없다.
3. yz 평면20mm bin마다 최소12점이 있어야 후보를 만든다. 중앙값에 가장 가까운 실제 관측점을 anchor로 선택한다.
4. nvblox raw query가 유효하고 절대값≤0.06m인 anchor를 남긴다.
5. 최대64슬롯에1mm 정육면체를 배치한다. 관측에서 사라진 패치는 해당 갱신에 비활성화한다. 관측 누락과 실제 제거를 완벽히 구분하지 않는다.

`TargetMargin`의 기존 형상 확대 margin은0이며, 본 조건에는30mm 확대 형상을 사용하지 않는다. 이웃 패치의 활성 범위가 겹칠 수 있고 비용이 합산된다. 전체 손의 닫힌 형상을 추정한 것이 아니다.

## 지도 및 상태 흐름

`hand_experiment.py`는 기록된 map_step마다 occupancy decay→깊이 적분→ESDF→동기화→hash 갱신을 수행한다. 물리 dt와 모델 base dt는1/60s, horizon30의 비균일 dt 합은0.675s다. MPPI400후보·기본5회 반복·seed0을 유지한다. 현재 물체 관측을 사용하며 물체의 미래 이동 예측 모델은 없다.

측정 q/v는 관절 이름으로 정렬해 제출한다. 기존 cuRobo 필터(위치·속도0.1)를 거쳐 optimizer에 들어가므로 raw 측정 상태와 필터 시작 상태가 동일하다는 뜻은 아니다. 명령은 위치 기반이며 기존 velocity target=zero를 유지한다.

`goal_progress.py`: 실제 EE 위치≤10mm 및 자세≤0.05rad 도달 이벤트 후 다음 스텝에 반대 목표를 지령한다. 대표 결과의 `--goal-plan AB`는 B 도달 뒤 유지한다. `shuttle`은 연속 왕복이고 대표 결과와 별도 모드다.

## 거리와 시간

`original_surface_clearance_m` = min(관측 ROI 점–로봇 구 중심 거리 − 버퍼 포함 반지름). 계획 구의 buffer는20mm다.

대표 사진·새 그래프의 거리 = 위 로그값 +20mm. 모든 구에서 공통 buffer를 제거했으므로 동일 점 집합·구 집합의 최솟값도 정확히20mm 차이 난다. 평가 점은 실행 중 약2,000점으로 솎은 ROI 관측점이다. 전 해상도·ROI 밖·mesh 기준 평가는 다른 지표다.

A→B 시간은 **B 지령 시작→실제 B 도달**이다. 시뮬레이션 시작 시각과 구분한다. 조건별 최솟값 시점은 R163/LOW169/HIGH165로 다르다. 대표 이미지의 같은 시점 화면과 전체 구간 최솟값을 혼동하지 않는다.

## 기여 범위

cuRobo 비용 함수와 MPPI·운동학·충돌 커널은 재사용했다. 추가 기여는 대상 표현·질량 조건 연결·실제 optimizer 삽입·기록/재생·실제 도달 판정·진단 및 평가다. 미관측 ESDF 처리나 native kernel 자체를 수정해 문제를 해결한 결과는 아니다.
