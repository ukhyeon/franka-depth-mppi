# 재현 및 실행 환경

## 공개 묶음에서 바로 가능한 것

- `scripts/build_figures.py`: 포함된 실제 실행 로그로 표·그림 생성. Python3, numpy, matplotlib 필요.
- `scripts/verify_bundle.py`: 원시 깊이 해시, 제어 코드 보존, 세 조건의 공통 설정·실제 도달 수치 검증.
- 기존 Isaac Sim/cuRobo/ABI=0 nvblox 설치가 있으면 `scripts/replay.sh`로 카메라 없이 기록 재생.

이 저장소는 Isaac Sim, cuRobo 전체 소스·로봇 자산, GPU 바이너리를 포함하지 않는다. 새 머신에서 설치부터 자동 완료하는 환경 패키지는 아니다.

## 확인된 버전 및 소스

| 항목 | 실행에 사용한 값 |
|---|---|
|Isaac Sim / Python|4.5.0 / 3.10.15|
|PyTorch|2.5.1+cu118, `_GLIBCXX_USE_CXX11_ABI=0`|
|CUDA / C++ 빌드 컴파일러|11.8 / GCC11|
|GPU|RTX4060Ti8GB, sm89|
|cuRobo|0.7.7, `0a50de1ba72db304195d59d9d0b1ed269696047f`|
|native nvblox|valtsblukis/nvblox `7bda93e993594ca69c479dde64213ac35835eb84`|
|nvblox_torch|NVlabs/nvblox_torch `e121121c100669848c75e1be97df94a91307e23f`|

native와 래퍼를 모두 ABI=0으로 별도 prefix에 빌드한 기존 환경이다. ABI=1 native와 ABI=0 래퍼를 섞으면 로딩이 되더라도 Mapper 해제 시 메모리 문제가 발생했던 이력이 있다. 이번 정리 작업에서 설치·재빌드는 하지 않았다.

기존 빌드 조건:

- native/래퍼 CXX 및 CUDA flags: `-D_GLIBCXX_USE_CXX11_ABI=0`.
- CUDA compiler/host: CUDA11.8 nvcc / g++11. 당시 `CMAKE_CUDA_ARCHITECTURES=89`.
- 래퍼는 Isaac Sim의 Torch CMake 경로와 새 native install prefix를 사용.
- 기존 ABI=0 glog0.6.0, gflags2.2.2, SQLite3.39.4 static archive, Eigen3.4.0, stdgpu `e10f6f3ccc9902d693af4380c3bcd188ec34a2e6` 사용.
- [래퍼 CMake 변경 기록](environment/nvblox_torch-abi0.patch): 새 native exported target 사용 및 시스템 glog 혼입 방지. 패치는 해당 commit 기준의 기록이며 자동 적용하지 않는다.
- native 소스 변경 없음. cuRobo core의 로컬 변경은 `particle_opt_base.py`의 설명 주석뿐임을 정리 시 확인했다. 수정된 Isaac 예제 helper는 이 저장소 `experiment/helper.py`에 보존했다.

상세 라이프사이클 검증은 기존 환경에서 Mapper 생성→저장지도 로딩→거리 질의→객체 해제→정상 종료를 독립 프로세스3회 확인한 것이다. 장기 안정성·다른 GPU 호환성 보장은 아니다.

## 실행 경로 지정

```bash
cp .env.example .env.local
```

파일의 `ISAACSIM_ROOT`, `CUROBO_ROOT`, `ABI0_ROOT`, `CUDA_HOME`을 기존 설치 경로로 맞춘다. ABI0_ROOT 아래 예상 구조:

```text
install/lib/libnvblox_lib.so
nvblox_torch/src/nvblox_torch/bin/libpy_nvblox.so
torch-cache/
```

실행 스크립트는 자신의 프로세스에서만 PYTHONPATH·LD_LIBRARY_PATH를 지정한다. ROS 컨테이너나 MoveIt은 사용하지 않으며 전역 셸 설정을 바꾸지 않는다.

## 최종 세 조건 재생

```bash
bash scripts/replay.sh R
bash scripts/replay.sh LOW
bash scripts/replay.sh HIGH
```

공통: 포함 깊이236프레임, 총472스텝, seed0, measured q/v, model base dt1/60, reached 전환, AB 후 B 유지, 기존 filter/cost 유지, 최종 infeasible이면 measured-position hold.

R은 추가 대상 비용을 연결하지 않는다. HIGH는 p=1 옵션을 명시한다. `hand_experiment.py`의 기본 HIGH(p=0.5)와 혼동하지 않는다. 각 호출은 새로운 프로세스·지도·solver로 시작한다.

GUI를 보려면 headless 옵션을 넣지 않고 다음 명령을 사용한다:

```bash
bash scripts/run_hand.sh --mode replay --condition HIGH --eta-exponent 1.0 \
  --input ../data/hand-tape-20261012-000004-472steps-registered \
  --goal-mode reached --goal-plan AB --state-input measured \
  --model-dt 0.016666666666666666 --failure-action hold-measured --audit-failures --light-render
```

기본 재생 wrapper는 headless다. GUI에서 상자의 목표 위치나 ROI를 바꾸면 공개 비교와 다른 실험이 된다. 원본 검증 기록은 `evidence/`, 새 결과는 `experiment/results/` 아래 timestamp 디렉터리다.

## 새 실측 입력 기록(대표 결과 재현과 별개)

D435i SDK `pyrealsense2`가 기존 Isaac Sim Python에 필요하다. 실제 물체 조작은 사용자가 수행한다.

```bash
bash scripts/run_hand.sh --mode record --condition R --hand-capture --hand-gui \
  --state-input measured --model-dt 0.016666666666666666 --failure-action hold-measured \
  --record-label new-hand --check-steps 20000
```

BACKGROUND(빈 장면)→PLACING(손 배치)→HOLD(고정)→REMOVED(손·팔 제거). 기록 중 가상 로봇은 고정 자세이며 회피 실험이 아니다. RGB는 `--portfolio-snapshots`를 명시할 때만 저장한다. 새 입력의 정합·품질 확인은 별도이며 이 저장소의 결과 수치를 자동으로 재현한다고 가정하지 않는다.

## 배포용 변경과 검증 범위

제어 Python 파일15개는 원본과 동일하다. 스크립트의 경로 지정·런타임 디렉터리 생성만 새로 작성했다. 배포 메타데이터에서는 장치 serial과 로컬 홈 절대경로만 제거/치환했다. 깊이 배열과 프레임 manifest, 관절/명령/비용 수치는 보존했다.

정리 과정의 검증은 포함 자료 무결성·공통 설정·실제 목표 도달·문서 링크·스크립트 구문 및 CLI 검사다. 새 제어 실험이나 정책 튜닝은 수행하지 않는다. 다른 머신에서 full GPU replay를 완료했다고 주장하지 않는다.
