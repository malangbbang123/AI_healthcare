# 🏥 Health Checkup Machine Learning Pipeline
**Prediction of Stroke & Angina Risk using Health Check-up Data**  
건강검진 데이터를 기반으로 **뇌졸중**, **심장병(심근경색·협심증)** 위험도를 예측하는 머신러닝 파이프라인입니다.  
데이터 전처리 → 병합 → 스케일링 → 모델 로딩 → 추론까지 전체 프로세스를 자동화합니다.

---

## 📁 Project Structure

/workspace/source/code_je/251104/│
├── utils.py # 공통 유틸 함수 (정규화, 스케일러 로딩 등)
├── Final_Preprocessing.py # 최종 전처리 스크립트
├── Survey_Preprocessing.py # 설문 데이터 정리
├── Result_Preprocessing.py # 검사결과 데이터 전처리
├── Merged_Preprocessing.py # 설문 + 검사결과 merge
├── Inference.py # 모델 불러오기 및 추론 실행
│
├── clean_data/
│ ├── Merged.csv # 병합된 중간 데이터
│ ├── Prep_251104.csv # 최종 학습/추론용 데이터
│
├── scaler/
│ ├── z_score_뇌졸중.pkl
│ ├── z_score_심장병(심근경색및협심증).pkl
│
├── json/
│ ├── features.json # 질병별 피처 및 라벨 목록
│
└── structure.txt # 전체 구조 문서



---

## ⚙️ Pipeline Overview

### 1️⃣ **전처리 단계**

#### 🔹 Result_Preprocessing.py  
- 검사결과 변수명 정리  
- 필요 없는 항목 제거 (시력, CT, MRI 등)  
- 결측치 처리  
- 일부 항목 수치화  

#### 🔹 Survey_Preprocessing.py  
- 복잡한 설문 항목 코드 → 의미 있는 이름으로 변환  
- 흡연·음주·운동 관련 항목 통합  
- mode/mean 기반 결측치 보정  

#### 🔹 Merged_Preprocessing.py  
- `S_PID` + `ORDDD(검진일자)` 로 join  
- 중복 제거 및 필요없는 컬럼 drop  

#### 🔹 Final_Preprocessing.py  
- 단위 통일 (예: 소주 잔→g, 병→g)  
- 가족력/과거력/운동/음주 feature 엔지니어링  
- 최종 분석용 테이블 생성  

---

## ✨ 2️⃣ Feature Scaling

`utils.py` 의 `load_scaler_or_fit()` 함수로 수행합니다.

- 기존 스케일러 존재 시 → 로드  
- 없으면 자동 학습 후 저장  
- 질환별 스케일러 파일 생성:
  - `z_score_뇌졸중.pkl`
  - `z_score_심장병(심근경색및협심증).pkl`

---

## 🤖 3️⃣ Modeling & Inference

`Inference.py`에서 아래 과정을 자동 수행합니다:

- 질환별 모델(catboost, xgboost, lgbm) 로드  
- Feature mismatch 검증  
- Soft Voting Ensemble  
- ROC-AUC 계산  
- 예측 결과 CSV 저장  

결과 저장 경로:

/workspace/source/test/{today}/Results/Inference_{질환명}/inference_result.csv



---

## 📈 4️⃣ Evaluation Metrics

- **ROC-AUC**
- **F1-score**
- **Sensitivity (Recall)**
- **Specificity**
- **Confusion Matrix**
- **MCC**

지표 계산은 `evaluation()` 함수에서 수행됩니다.

---

## 🧪 5️⃣ Run Inference

```bash
python Inference.py



Running inference for [뇌졸중]
ROC-AUC: 0.8421
Saved results → /workspace/source/test/20250101/Results/Inference_뇌졸중/inference_result.csv


| S_PID | Pred_Ensemble | True_Label |
| ----- | ------------- | ---------- |
| 12345 | 0.842         | 1          |
| 98721 | 0.103         | 0          |



🚀 Requirements
Python 3.10+
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
joblib
json
pickle

👩‍💻 Author

홍지은 (Jieun Hong)
Machine Learning Researcher – Neurodigm
Healthcare AI • Disease Prediction • ML Pipeline Engineering

