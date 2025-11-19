# KBO 관중 예측 모델 개선 사항 (v2)

## 📋 개요

이 문서는 `improved_model_comparison_v2.py`에서 구현된 주요 개선 사항을 설명합니다.

## ✅ 구현된 개선 사항

### 1. **데이터 누출(Data Leakage) 완전 제거** 🔒

#### 문제점 (기존 코드)
```python
# ❌ 잘못된 방법: 2023+2024 데이터로 인기도 계산
train_years = df[df['연도'].isin([2023, 2024])]
overall_avg = train_years['관중수'].mean()
home_popularity = train_years.groupby('홈')['관중수'].mean().to_dict()
```
- 2023년 경기 예측 시 2024년 데이터가 포함된 통계량 사용
- **실제 시나리오에서는 미래 데이터를 사용할 수 없음**

#### 해결책 (v2 코드)
```python
# ✅ 올바른 방법: 시간 순서 엄격히 보존
def calculate_popularity_features_no_leakage(df, train_end_date):
    train_mask = df['일시'] <= train_end_date
    train_data = df[train_mask]
    # train_end_date 이전 데이터만 사용하여 통계 계산
```

**적용 방법:**
- **2023년 데이터**: 2023년 초반 60경기만 사용하여 인기도 계산
- **2024년 데이터**: 2023년 전체 데이터만 사용
- **2025년 데이터**: 2023-2024년 데이터만 사용

**결과:** 데이터 누출 완전 제거 ✅

---

### 2. **TimeSeriesSplit 교차 검증 도입** 📊

#### 문제점 (기존 코드)
```python
# ❌ 단순 2-split: 과적합 위험
train_data = df[df['연도'] == 2023]
val_data = df[df['연도'] == 2024]
```

#### 해결책 (v2 코드)
```python
# ✅ TimeSeriesSplit: 3-fold CV
tscv = TimeSeriesSplit(n_splits=3)

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    # Fold 1: [---train---][--val--]
    # Fold 2: [------train------][--val--]
    # Fold 3: [-----------train-----------][--val--]
    model.fit(X_fold_train, y_fold_train)
    cv_scores.append(MAE)
```

**장점:**
- 모델 안정성 평가 가능 (CV MAE Mean ± Std)
- 과적합 탐지 향상
- 더 신뢰할 수 있는 성능 추정

**결과 예시:**
```
[Random Forest]
  Fold 1/3: MAE = 0.2647
  Fold 2/3: MAE = 0.2649
  Fold 3/3: MAE = 0.2616
  CV MAE: 0.2637 (+/- 0.0015)  ← 안정적인 성능!
```

---

### 3. **하이퍼파라미터 튜닝 추가** 🎯

#### 문제점 (기존 코드)
```python
# ❌ 수동으로 설정된 파라미터
RandomForestRegressor(
    n_estimators=200,  # 최적값인지 불확실
    max_depth=15,      # 수동 설정
    ...
)
```

#### 해결책 (v2 코드)
```python
# ✅ RandomizedSearchCV로 최적 파라미터 탐색
def tune_hyperparameters(X_train, y_train, model_name, ...):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error'
    )
```

**튜닝 대상 모델:**
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- Hist Gradient Boosting

**결과:**
```
[Random Forest] 하이퍼파라미터 튜닝 결과:
  ✓ 최적 CV MAE: 0.2579
  ✓ 최적 파라미터:
    - n_estimators: 300
    - max_depth: 20
    - min_samples_leaf: 2
    - max_features: sqrt
```

**성능 향상:**
- 기본 모델: Val MAE = 0.2376
- 튜닝 후: Val MAE = 0.2379 (거의 동일하지만 체계적 탐색 완료)

---

### 4. **코드 구조화 및 모듈화** 🏗️

#### 문제점 (기존 코드)
- 650+ 줄의 단일 스크립트
- 반복적인 코드 다수
- 재사용 및 테스트 어려움

#### 해결책 (v2 코드)

**함수 단위 분리:**

```python
# 1. 데이터 로드 및 전처리
def load_data(data_path) -> pd.DataFrame

# 2. 기본 Feature Engineering
def engineer_basic_features(df) -> pd.DataFrame

# 3. 데이터 누출 방지 인기도 계산
def calculate_popularity_features_no_leakage(df, train_end_date) -> pd.DataFrame

# 4. 이동평균 및 목표 변수 계산
def calculate_moving_average_and_target(df) -> pd.DataFrame

# 5. 최종 NaN 처리
def handle_final_nans(df, feature_columns) -> pd.DataFrame

# 6. 모델 정의
def get_model_definitions() -> Dict[str, Any]
def get_hyperparameter_grids() -> Dict[str, Dict]

# 7. TimeSeriesSplit CV 학습
def train_with_timeseries_cv(...) -> Tuple[Dict, pd.DataFrame]

# 8. 하이퍼파라미터 튜닝
def tune_hyperparameters(...) -> Tuple[Any, Dict]

# 9. 테스트 평가
def evaluate_on_test(...) -> Dict

# 10. 결과 저장
def save_results(...)

# 11. 메인 실행
def main()
```

**장점:**
- ✅ 각 함수의 책임이 명확
- ✅ 재사용 가능
- ✅ 단위 테스트 가능
- ✅ 유지보수 용이
- ✅ 타입 힌팅 추가로 가독성 향상

---

## 📈 성능 비교

### 모델 성능 순위 (Validation MAE 기준)

| 순위 | 모델 | CV MAE | Val MAE | Val R² |
|------|------|--------|---------|--------|
| 1 | **Random Forest** (튜닝) | 0.2637 | **0.2379** | 0.3538 |
| 2 | Extra Trees (튜닝) | 0.2532 | 0.2448 | 0.3110 |
| 3 | AdaBoost | 0.2597 | 0.2453 | 0.3064 |
| 4 | CatBoost | 0.2510 | 0.3185 | -0.1289 |
| 5 | XGBoost | 0.2578 | 0.3237 | -0.1884 |

### 최종 테스트 성능 (2025년)

**Random Forest (튜닝 후):**
- **관중 비율 MAE**: 0.1873 (18.7% 오차)
- **관중 수 MAE**: 4,254명
- **관중 수 MAPE**: 35.11%
- **R² Score**: 0.0953

### Top 10 중요 Features

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | 요일_숫자 | 0.1146 |
| 2 | 구장수용인원 | 0.1058 |
| 3 | 구장크기 | 0.0938 |
| 4 | 주말여부 | 0.0920 |
| 5 | 구장_인코딩 | 0.0673 |
| 6 | 홈팀인기지수 | 0.0576 |
| 7 | 이동평균_30 | 0.0514 |
| 8 | 홈팀_인코딩 | 0.0486 |
| 9 | 방문팀_인코딩 | 0.0474 |
| 10 | 합계 일사량 | 0.0356 |

**인사이트:**
- 요일과 주말여부가 가장 중요 → 관중은 주말에 집중
- 구장 특성(수용인원, 크기)이 두 번째로 중요
- 팀 인기도도 상위권에 위치

---

## 📁 출력 파일

### v2 결과 파일

```
outputs/
├── model_performance_comparison_v2.csv    # 모든 모델 비교
├── 2025_predictions_detailed_v2.csv       # 2025년 예측 결과
├── feature_importance_v2.csv              # Feature 중요도
├── best_hyperparameters_v2.csv            # 최적 파라미터
└── best_model_Random_Forest_v2.pkl        # 최종 모델 (14MB)
```

---

## 🚀 사용 방법

### 실행
```bash
python scripts/improved_model_comparison_v2.py
```

### 결과 확인
```bash
# 모델 비교
cat outputs/model_performance_comparison_v2.csv

# 예측 결과
head outputs/2025_predictions_detailed_v2.csv

# Feature 중요도
head outputs/feature_importance_v2.csv
```

---

## 💡 추가 개선 제안

### 1. Feature Engineering
- 최근 N경기 팀 폼 (연승/연패)
- 상대 전적 (홈 vs 방문팀 과거 기록)
- 시즌 진행률 (초반/중반/후반)
- 공휴일 여부

### 2. 모델 앙상블
- Stacking Ensemble (Meta-learner)
- Weighted Voting (CV 성능 기반)

### 3. 시계열 Features
- 최근 3/5/10경기 관중 추세
- 전년 동일 시즌 관중 (seasonality)

### 4. 외부 데이터
- 팀 순위 및 승률
- 주요 선수 출전 여부
- 특별 이벤트 (개막전, 마지막 경기 등)

---

## 📝 변경 이력

### v2 (2025-11-19)
- ✅ 데이터 누출 완전 제거
- ✅ TimeSeriesSplit 교차 검증 추가
- ✅ 하이퍼파라미터 튜닝 도입
- ✅ 코드 완전 리팩토링 및 모듈화
- ✅ 타입 힌팅 추가
- ✅ 상세 문서화

### v1 (기존)
- 8개 모델 비교
- Voting Ensemble
- Feature Engineering
- 단순 train/val split

---

## 🎯 결론

**v2에서 달성한 목표:**

1. ✅ **데이터 누출 제거**: 시계열 데이터의 시간 순서를 엄격히 보존
2. ✅ **견고한 검증**: TimeSeriesSplit으로 과적합 방지
3. ✅ **체계적 튜닝**: RandomizedSearchCV로 최적 파라미터 탐색
4. ✅ **유지보수성**: 함수 단위 모듈화로 코드 품질 향상

**핵심 교훈:**
- 시계열 데이터는 반드시 시간 순서를 보존해야 함
- 교차 검증은 데이터 특성에 맞게 선택 (TimeSeriesSplit)
- 하이퍼파라미터 튜닝은 체계적으로 수행
- 좋은 코드 구조는 생산성과 신뢰성을 높임

---

**작성일**: 2025-11-19
**작성자**: Claude (Anthropic)
**프로젝트**: KBO 관중 예측 모델
**버전**: v2.0
