# Realtime_Assembly_Guide
Realtime_Assembly_Guide

각 소스의 원본 파일만 보존하고 코드 진행 후 나온 실행 파일은 업로드X

📁 부품 정리 가이드
├── 📂 dataset_detect_parts # 학습할 데이터셋 (부품만)
│   ├── 📂 images_all       # 부품 이미지
│   │   ├── ...
│   ├── 📂 labels_all       # 라벨 txt
│   │   ├── ...
│   ├── 📂 _prepared        # 코드로 생성
│   │   ├── data.yaml
│   │   ├── ...             # 폴더 (images & labels)
├── 📂 runs_detect          # 부품 학습
│   ├── 📂 total_detect     # 1차 학습
│   │   ├── 📂 weights
│   │   │   ├── best.pt      # 단일 부품
│   │   ├── ...
│   ├── 📂 total_detect2     # 2차 학습
│   │   ├── 📂 weights
│   │   │   ├── best.pt      # 복합 부품
│   │   ├── ...
│
├── 📂 dataset_ready        # 학습할 데이터_단계 (코드로 생성)
│   ├── 📂 step1            # 학습 (단계)
│   ├── ...
├── 📂 runs_classify        # 학습 후 데이터 (분류)
│   ├── 📂 foosball_steps   # 학습
│   │   ├── 📂 weights
│   │   │   ├── best.pt
│   │   ├── ...
│
├── 3t_project_UI.py