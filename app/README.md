# App Module

PyQt5 기반 실시간 조립 가이드 애플리케이션입니다.  
카메라 입력, 모델 추론 결과, 단계별 텍스트/이미지 가이드를 하나의 사용자 인터페이스로 통합하여 보여줍니다.

## 프로젝트 설명

`app/` 폴더는 사용자가 직접 실행하는 애플리케이션 영역입니다.  
이 모듈은 카메라 프레임을 받아 Detect/Classify 모델 추론 결과를 반영하고, 현재 제품/부품/단계에 맞는 가이드를 실시간으로 화면에 표시합니다.

즉, 이 폴더는 전체 프로젝트에서 **실행/UI 계층**을 담당합니다.

## 핵심 기능

- 실시간 카메라 화면 표시
- 제품 자동 감지 및 가이드 진입
- 부품 체크리스트 표시
- 단계별 텍스트 가이드 표시
- 단계별 이미지 가이드 표시
- 신뢰도 출력
- 조건 충족 시 자동 다음 단계 전환
- 혼합 감지 / 모델 누락 / 예측 실패 등 예외 처리

## 기술 스택

- Python
- PyQt5
- OpenCV
- Ultralytics YOLO
- PyTorch

## 자동 전환 로직 개요

앱은 단일 프레임 결과만으로 단계를 즉시 넘기지 않습니다.
현재 기대 단계와 분류 결과가 일치하고, 신뢰도가 기준 이상이며, 일정 횟수 이상 연속 확인된 경우에만 다음 단계로 전환합니다.

이 구조의 장점:

순간 오탐 방지

흔들림/가림에 대한 안정성 향상

미완성 상태에서 성급한 진행 억제

## 모델 관리

Detect 모델
사용 위치: 제품 자동 감지, 부품 체크리스트

Classify 모델
사용 위치: 단계 판단 및 자동 진행

## 폴더 구조

```bash
app/
├─ README.md
├─ Realtime_Assembly_Guide.py
├─ assets/
│  ├─ foosball/
│  │  ├─ foosball.png
│  │  ├─ parts-1.png
│  │  ├─ step 1-1.png
│  │  └─ ...
│  ├─ plates_tool/
│  └─ raspberry_pi/
├─ YOLO11_model/
│  ├─ detect_all/
│  │  └─ weights/
│  │  │  └─ best.pt
│  └─ classify_foosball/
│  │  └─ weights/
│  │  │  └─ best.pt
│  └─ classify_plates/
│  │  └─ weights/
│  │  │  └─ best.pt
│  └─ classify_raspberry/
│  │  └─ weights/
│  │  │  └─ best.pt
```