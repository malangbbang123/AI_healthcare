🏥 Health Checkup ML Pipeline
Prediction of Stroke & Angina Risk using Health Check-up Data

본 프로젝트는 건강검진 데이터를 기반으로 뇌졸중, 심장병(심근경색·협심증) 발생 위험을 예측하는 머신러닝 파이프라인입니다.
데이터 전처리, 머지, 스케일링, 모델 로딩, 추론까지 전체 자동화 파이프라인을 제공합니다.

📁 Project Structure

프로젝트 전체 디렉토리 구조는 다음과 같습니다:

/workspace/source/code_je/251104/
│
├── utils.py                         # 공통 함수 모듈 (파일명 정리, 스케일러 로딩 등)
├── Final_Preprocessing.py           # 결과 + 설문 데이터를 합친 전처리(최종본)
├── Merged_Preprocessing.py          # 설문 + 검사결과 merge 스크립트
├── Survey_Preprocessing.py          # 설문지 전처리
├── Result_Preprocessing.py          # 검사결과 전처리
├── Inference.py                     # 모델 로딩 및 추론 파이프라인
│
├── clean_data/
│   ├── Merged.csv                   # merge된 중간 데이터
│   ├── Prep_251104.csv              # 최종 학습/추론용 데이터
│
├── scaler/
│   ├── z_score_뇌졸중.pkl
│   ├── z_score_심장병(심근경색및협심증).pkl
│
├── json/
│   ├── features.json                # 질병별 feature 리스트 & 라벨
│
└── structure.txt                    # 전체 구조 문서

⚙️ Pipeline Overview
1️⃣ 데이터 전처리 (Preprocessing)

전처리는 크게 네 단계로 구성됩니다:

✓ 검사결과(Result) 전처리

코드북 기반 변수명 변경

필요 없는 검사 항목 제거

음성/양성 변수 수치화

극단값 및 결측치 처리
📄 코드: Result_Preprocessing.py


Result_Preprocessing

✓ 설문(Survey) 전처리

800+ 설문 항목 코드 → 한글/의미 있는 이름으로 치환

가족력/과거력/흡연/음주/운동 항목 통합

mode/mean 기반 결측치 처리
📄 코드: Survey_Preprocessing.py


Survey_Preprocessing

✓ Merge

S_PID + ORDDD(검진일자) 기준으로 병합

중복/불필요 열 제거
📄 코드: Merged_Preprocessing.py


Merged_Preprocessing

✓ 최종 통합 & Cleaning

복잡한 가족력/흡연/음주/운동 feature 엔지니어링

단위 통합(잔/병/cc → g)

통합 feature 생성
📄 코드: Final_Preprocessing.py


Final_Preprocessing

📊 2️⃣ Feature Scaling

질병별로 StandardScaler를 별도로 fit 또는 로드:

📄 코드: utils.py -> load_scaler_or_fit()


Inference

기존 스케일러 존재 시 로드

없으면 자동 학습 후 저장

features.json 기반으로 스케일할 컬럼 자동 선택

🤖 3️⃣ Modeling & Inference

Inference 파이프라인은 다음을 포함합니다:

각 질병별 모델(catboost, xgboost, lightgbm) 로드

특징 불일치 방지 위한 feature 이름 검증

소프트보팅(Soft Voting) ensemble

ROC-AUC 계산

추론 결과 CSV로 저장

📄 코드: Inference.py


Inference

🔍 4️⃣ Run Inference
python Inference.py


실행 시:

뇌졸중 / 심장병 모델 각각 실행

test_size=0.2로 내부 검증

추론 결과 저장 위치:

/workspace/source/test/{오늘날짜}/Results/Inference_뇌졸중/inference_result.csv

🎯 5️⃣ Output Format

저장되는 결과 파일:

S_PID	Pred_Ensemble	True_Label
12345	0.842	1
98721	0.103	0
🧪 6️⃣ Evaluation Metrics

ROC-AUC

F1-score

Sensitivity (Recall)

Specificity

Confusion Matrix

📄 계산 방식은 evaluation() 함수 참고


utils

🚀 Requirements
Python 3.10+
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
joblib
pickle
json

✨ Author

홍지은 / Jieun Hong
Machine Learning Researcher – Neurodigm
(Healthcare AI, Disease Prediction, Medical ML Pipeline)
