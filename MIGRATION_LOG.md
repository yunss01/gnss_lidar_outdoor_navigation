# CARLA 마이그레이션 기록

## 2026-07-16 - Task 1: Gazebo 제거

- 수정 범위는 사용자 요청에 따라 `simulation_pkg`와 `launch_pkg`로 제한했다.
- `simulation_pkg`의 월드, 모델, RViz 설정, 스폰 노드, 장애물 이동 노드,
  학습 모델 및 Gazebo 의존성을 제거했다.
- 기존 `MotionCommand` 제어 변환식은 Task 4에서 CARLA 제어 노드로 옮길 때까지
  `simulation_pkg/simulation_sender_node.py`에 보존했다.
- 다른 인식, 판단, 디버그, 시리얼 패키지는 수정하거나 빌드에서 제외하지 않았다.
- 전체 9개 ROS 2 패키지의 `colcon build --symlink-install`이 성공했다.
- `src/`에서 `gazebo` 및 `ros_gz` 검색 결과는 0건이다.
- `simulation_pkg` 테스트 결과는 2개 성공, 저작권 검사 1개 건너뜀이다.
- `launch_pkg`의 Flake8, CMake lint, PEP257, XML 검사가 모두 성공했다.
- 전체 테스트 결과는 16개 중 실패 0개, 건너뜀 1개다.

## 2026-07-16 - Task 2: CARLA 0.10.0 설치 및 검사

### 설치 정보

- 설치 시각: `2026-07-16T16:22:57+09:00`
- 설치 경로: `/home/sukja/carla/0.10.0`
- 배포 파일: 공식 Ubuntu 22 패키지
  `Carla-0.10.0-Linux-Shipping.tar.gz`
- 다운로드 주소:
  `https://carla-releases.b-cdn.net/Linux/Carla-0.10.0-Linux-Shipping.tar.gz`
- 배포 파일 크기: `10,422,621,784 bytes`
- 설치본 크기: 약 `20 GiB`
- CARLA 커밋: `ada75f920642e18cace0a9f85ecf9d4077ddb531`
- 콘텐츠 커밋: `518f45bd010c0dea7239b55eed27d01ba132934e`
- Unreal Engine 커밋: `6cf20383302899026a5e777219ba4206afe47d38`
- Python: `3.10.12`
- Python 클라이언트: 패키지 포함
  `carla-0.10.0-cp310-cp310-linux_x86_64.whl`
- Python 설치 위치: `/home/sukja/.local/lib/python3.10/site-packages`

### 시험 환경

- 운영체제: Ubuntu 22.04, 커널 `6.8.0-87-generic`
- GPU: NVIDIA GeForce RTX 3060, VRAM `12,288 MiB`
- NVIDIA 드라이버: `580.159.03`
- CPU: AMD Ryzen 7 8700G, 8코어 16스레드
- RAM: `60 GiB`

### 실행 및 결과

서버 실행 명령:

```bash
cd /home/sukja/carla/0.10.0
./CarlaUnreal.sh --ros2 -RenderOffScreen -quality-level=Low \
  -ResX=640 -ResY=360 -nosound \
  '-ExecCmds=g.TimeoutForBlockOnRenderFence 600'
```

- 서버 버전: `0.10.0`
- 클라이언트 버전: `0.10.0`
- 제공 맵: `Town10HD_Opt`, `Mine_01`
- `Mine_01` 로드: 성공
- `Mine_01` 300프레임 시뮬레이션 FPS: `87.76`
- `Mine_01` 300프레임 실제 처리 FPS: `87.78`
- 측정 시간: `3.42초`
- 측정 중 GPU 메모리: `4,652 MiB / 12,288 MiB`
- 측정 중 GPU 사용률: `96%`
- 측정 중 GPU 온도: `64도`
- 추가 안정성 확인: `Mine_01`에서 4,250프레임까지 진행 확인

### 확인된 주의사항

- 기본 60초 렌더 펜스 제한에서는 `Mine_01` 동적 로드 중
  `GameThread timed out waiting for RenderThread` 오류로 서버가 종료됐다.
- `g.TimeoutForBlockOnRenderFence 600`을 적용한 재시도에서는 맵 로드와 FPS 측정이
  성공했다. 이후 0.10.0 실행 명령에 이 설정을 유지한다.
- 공식 권장 VRAM 16GB보다 적은 12GB GPU이므로 위 수치는 저품질·오프스크린
  조건이다. 센서 추가 후 FPS를 다시 측정해야 한다.
- 설치 후 홈 파티션 여유 공간은 약 `7.5 GiB`다. UE 캐시와 rosbag을 저장하기
  전에 추가 공간을 확보해야 한다.
- 0.9.16 폴백은 현재 필요하지 않다.
