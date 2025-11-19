"""
KBO 정규시즌 관중 수 예측 모델 구축
3조 - 모델 구축 담당: 임혜린, 윤태영

핵심 방법론: 비율 기반 시계열 예측
- 목표: 관중 비율 = 실제 관중 수 / 해당 일 제외 직전 30경기 이동평균
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("KBO 관중 수 예측 모델 구축 프로젝트")
print("=" * 80)

# ====== 1. 데이터 로드 및 기본 탐색 ======
print("\n[Phase 1] 데이터 로드 및 탐색")
print("-" * 80)

data_path = Path(__file__).resolve().parent / "df_refined.csv"
df = pd.read_csv(data_path, encoding='utf-8-sig')

print(f"데이터 shape: {df.shape}")
print(f"\n컬럼 목록:\n{df.columns.tolist()}")
print(f"\n데이터 타입:\n{df.dtypes}")
print(f"\n결측치 현황:\n{df.isnull().sum()}")
print(f"\n기초 통계:\n{df.describe()}")

# 상위 5개 데이터 확인
print(f"\n데이터 샘플:\n{df.head()}")

# ====== 2. 데이터 전처리 ======
print("\n[Phase 2] 데이터 전처리")
print("-" * 80)

# 2.1 결측치 처리
print("2.1 결측치 처리 중...")
missing_cols = df.columns[df.isnull().any()].tolist()
print(f"결측치가 있는 컬럼: {missing_cols}")

for col in missing_cols:
    if df[col].dtype in ['float64', 'int64']:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"  - {col}: 평균값({mean_value:.2f})으로 대체")

print(f"결측치 처리 후: {df.isnull().sum().sum()}개")

# 2.2 일시 변환
print("\n2.2 일시 데이터 변환...")
df['일시'] = pd.to_datetime(df['일시'], format='%Y년%m월%d일')
df = df.sort_values('일시').reset_index(drop=True)
print(f"일시 범위: {df['일시'].min()} ~ {df['일시'].max()}")

# 2.3 연도, 월, 일 추출
df['연도'] = df['일시'].dt.year
df['월'] = df['일시'].dt.month
df['일'] = df['일시'].dt.day
df['주차'] = df['일시'].dt.isocalendar().week

print(f"연도별 경기 수:\n{df['연도'].value_counts().sort_index()}")

# ====== 3. Feature Engineering ======
print("\n[Phase 3] Feature Engineering")
print("-" * 80)

# 3.1 시간 관련 Features
print("3.1 시간 관련 Features 생성...")

# 요일 인코딩 (월=0, 일=6)
weekday_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
df['요일_숫자'] = df['요일'].map(weekday_map)

# 주중/주말 구분
df['주말여부'] = df['요일_숫자'].apply(lambda x: 1 if x >= 4 else 0)

# 계절성 인코딩 (sin/cos 변환)
df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)

print(f"  - 요일 인코딩 완료")
print(f"  - 주말 경기: {df['주말여부'].sum()}경기")
print(f"  - 주중 경기: {len(df) - df['주말여부'].sum()}경기")

# 3.2 날씨 관련 Features
print("\n3.2 날씨 관련 Features 생성...")

# 평균 기온
df['평균기온'] = (df['최저기온(°C)'] + df['최고기온(°C)']) / 2

# 비 여부
df['비여부'] = (df['일강수량(mm)'] > 0).astype(int)

# 강수 강도 범주화
def categorize_rain(rain):
    if rain == 0:
        return 0  # 맑음
    elif rain < 5:
        return 1  # 약한 비
    elif rain < 20:
        return 2  # 비
    else:
        return 3  # 강한 비

df['강수강도'] = df['일강수량(mm)'].apply(categorize_rain)

# 날씨 상태 (전운량 기반)
def categorize_weather(cloud):
    if cloud <= 3:
        return 0  # 맑음
    elif cloud <= 7:
        return 1  # 구름 많음
    else:
        return 2  # 흐림

df['날씨상태'] = df['평균 전운량(1/10)'].apply(categorize_weather)

print(f"  - 평균 기온: {df['평균기온'].mean():.2f}°C")
print(f"  - 비 오는 날: {df['비여부'].sum()}일 ({df['비여부'].sum()/len(df)*100:.1f}%)")
print(f"  - 강수강도 분포:\n{df['강수강도'].value_counts().sort_index()}")

# 3.3 구장 관련 Features
print("\n3.3 구장 관련 Features 생성...")

# 구장별 최대 수용 인원 (공식 자료 기준)
stadium_capacity = {
    '잠실': 25000,
    '수원': 20000,
    '문학': 20500,
    '고척': 16500,
    '광주': 11000,
    '사직': 23000,
    '창원': 20000,
    '대구': 10000,
    '한밭': 11500,  # 2023-2024년 한화 홈구장
    '대전': 13000,  # 2025년 한화 새 구장
    '울산': 11000,
    '포항': 11000,
    '청주': 10000
}

df['구장수용인원'] = df['구장'].map(stadium_capacity)

# 구장 크기 범주
df['구장크기'] = pd.cut(df['구장수용인원'], 
                       bins=[0, 12000, 20000, 30000], 
                       labels=[0, 1, 2]).astype(float)  # 숫자로 변환

print(f"  - 구장 수: {df['구장'].nunique()}개")
print(f"  - 구장별 경기 수:\n{df['구장'].value_counts()}")

# 3.4 팀 인기도 Features (핵심!)
print("\n3.4 팀 인기도 Features 생성...")

# 2023-2024년 데이터로 팀별 평균 관중 계산
train_years = df[df['연도'].isin([2023, 2024])]

# 전체 평균 관중 (먼저 계산)
overall_avg = train_years['관중수'].mean()

# 홈팀 평균 관중
home_popularity = train_years.groupby('홈')['관중수'].mean().to_dict()
df['홈팀평균관중'] = df['홈'].map(home_popularity)
# 2025년 데이터에 없는 팀은 평균값으로 처리
df['홈팀평균관중'].fillna(overall_avg, inplace=True)

# 방문팀 평균 관중
away_popularity = train_years.groupby('방문')['관중수'].mean().to_dict()
df['방문팀평균관중'] = df['방문'].map(away_popularity)
df['방문팀평균관중'].fillna(overall_avg, inplace=True)

# 팀 인기 지수 (평균 대비 비율)
df['홈팀인기지수'] = df['홈팀평균관중'] / overall_avg
df['방문팀인기지수'] = df['방문팀평균관중'] / overall_avg

# 대진 조합 인기도
matchup = train_years.groupby(['홈', '방문'])['관중수'].mean().to_dict()
df['대진평균관중'] = df.apply(lambda row: matchup.get((row['홈'], row['방문']), overall_avg), axis=1)
df['대진인기지수'] = df['대진평균관중'] / overall_avg

print(f"  - 홈팀별 평균 관중:\n{pd.Series(home_popularity).sort_values(ascending=False)}")
print(f"\n  - 전체 평균 관중: {overall_avg:.0f}명")

# 3.5 이동평균 및 목표 변수 계산 (핵심!)
print("\n3.5 목표 변수(관중 비율) 계산...")

# 각 경기 이전 30경기의 평균 관중 수 (이동평균)
df['이동평균_30'] = df['관중수'].rolling(window=30, min_periods=1).mean().shift(1)

# 처음 30경기는 전체 평균으로 초기화
initial_avg = df.iloc[:30]['관중수'].mean()
df.loc[df['이동평균_30'].isna(), '이동평균_30'] = initial_avg

# 관중 비율 = 실제 관중수 / 이동평균
df['관중비율'] = df['관중수'] / df['이동평균_30']

print(f"  - 이동평균 범위: {df['이동평균_30'].min():.0f} ~ {df['이동평균_30'].max():.0f}")
print(f"  - 관중비율 평균: {df['관중비율'].mean():.3f}")
print(f"  - 관중비율 범위: {df['관중비율'].min():.3f} ~ {df['관중비율'].max():.3f}")

# 매진율 계산
df['매진율'] = df['관중수'] / df['구장수용인원']
print(f"  - 평균 매진율: {df['매진율'].mean():.1%}")

# 3.6 범주형 변수 인코딩
print("\n3.6 범주형 변수 인코딩...")

# Label Encoding
le_stadium = LabelEncoder()
le_home = LabelEncoder()
le_away = LabelEncoder()

df['구장_인코딩'] = le_stadium.fit_transform(df['구장'])
df['홈팀_인코딩'] = le_home.fit_transform(df['홈'])
df['방문팀_인코딩'] = le_away.fit_transform(df['방문'])

print(f"  - 구장: {df['구장'].nunique()}개 -> {df['구장_인코딩'].nunique()}개 코드")
print(f"  - 홈팀: {df['홈'].nunique()}개 -> {df['홈팀_인코딩'].nunique()}개 코드")
print(f"  - 방문팀: {df['방문'].nunique()}개 -> {df['방문팀_인코딩'].nunique()}개 코드")

# ====== 4. 데이터 분할 (시계열 특성 고려) ======
print("\n[Phase 4] 데이터 분할")
print("-" * 80)

# Phase 1: 2023년 Train, 2024년 Validation
train_data = df[df['연도'] == 2023].copy()
val_data = df[df['연도'] == 2024].copy()
test_data = df[df['연도'] == 2025].copy()

print(f"Train (2023): {len(train_data)}경기")
print(f"Validation (2024): {len(val_data)}경기")
print(f"Test (2025): {len(test_data)}경기")

# Feature 선택
feature_columns = [
    # 시간 Features
    '월', '요일_숫자', '주말여부', '월_sin', '월_cos',
    
    # 날씨 Features
    '최저기온(°C)', '최고기온(°C)', '평균기온',
    '일강수량(mm)', '비여부', '강수강도',
    '평균 상대습도(%)', '합계 일사량(MJ/m2)',
    '평균 전운량(1/10)', '날씨상태',
    '평균 지면온도(°C)',
    
    # 구장 Features
    '구장_인코딩', '구장수용인원', '구장크기',
    
    # 팀 인기도 Features
    '홈팀_인코딩', '방문팀_인코딩',
    '홈팀인기지수', '방문팀인기지수', '대진인기지수',
    
    # 이동평균
    '이동평균_30'
]

print(f"\n사용할 Feature 수: {len(feature_columns)}개")
print(f"Feature 목록:\n{feature_columns}")

# 데이터 준비
X_train = train_data[feature_columns]
y_train = train_data['관중비율']

X_val = val_data[feature_columns]
y_val = val_data['관중비율']

X_test = test_data[feature_columns]
y_test = test_data['관중비율']

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ====== 5. 모델 학습 ======
print("\n[Phase 5] 모델 학습")
print("-" * 80)

# 5.1 Decision Tree Regressor
print("\n5.1 Decision Tree Regressor 학습...")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)

y_val_pred_dt = dt_model.predict(X_val)
mae_dt = mean_absolute_error(y_val, y_val_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_val, y_val_pred_dt))
r2_dt = r2_score(y_val, y_val_pred_dt)

print(f"Validation MAE: {mae_dt:.4f}")
print(f"Validation RMSE: {rmse_dt:.4f}")
print(f"Validation R²: {r2_dt:.4f}")

# 5.2 Random Forest Regressor (추천!)
print("\n5.2 Random Forest Regressor 학습...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_val_pred_rf = rf_model.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
r2_rf = r2_score(y_val, y_val_pred_rf)

print(f"Validation MAE: {mae_rf:.4f}")
print(f"Validation RMSE: {rmse_rf:.4f}")
print(f"Validation R²: {r2_rf:.4f}")

# 5.3 Hist Gradient Boosting Regressor (NaN 처리 가능)
print("\n5.3 Hist Gradient Boosting Regressor 학습...")
gb_model = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

y_val_pred_gb = gb_model.predict(X_val)
mae_gb = mean_absolute_error(y_val, y_val_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_val, y_val_pred_gb))
r2_gb = r2_score(y_val, y_val_pred_gb)

print(f"Validation MAE: {mae_gb:.4f}")
print(f"Validation RMSE: {rmse_gb:.4f}")
print(f"Validation R²: {r2_gb:.4f}")

# 최적 모델 선택
print("\n모델 성능 비교:")
models_performance = {
    'Decision Tree': {'MAE': mae_dt, 'RMSE': rmse_dt, 'R²': r2_dt},
    'Random Forest': {'MAE': mae_rf, 'RMSE': rmse_rf, 'R²': r2_rf},
    'Gradient Boosting': {'MAE': mae_gb, 'RMSE': rmse_gb, 'R²': r2_gb}
}

for model_name, metrics in models_performance.items():
    print(f"{model_name:20s} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R²']:.4f}")

# 최적 모델 선택 (MAE 기준)
best_model_name = min(models_performance, key=lambda x: models_performance[x]['MAE'])
print(f"\n최적 모델: {best_model_name}")

# 최적 모델 설정
if best_model_name == 'Decision Tree':
    best_model = dt_model
    y_val_pred_best = y_val_pred_dt
elif best_model_name == 'Random Forest':
    best_model = rf_model
    y_val_pred_best = y_val_pred_rf
else:
    best_model = gb_model
    y_val_pred_best = y_val_pred_gb

# ====== 6. 하이퍼파라미터 튜닝 (Random Forest) ======
print("\n[Phase 6] 하이퍼파라미터 튜닝 (Random Forest)")
print("-" * 80)

print("GridSearchCV를 사용한 하이퍼파라미터 최적화...")
param_grid = {
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'n_estimators': [200, 300]
}

tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

# 2023년 데이터만으로 Grid Search
grid_search.fit(X_train, y_train)

print(f"\n최적 파라미터: {grid_search.best_params_}")
print(f"최적 MAE: {-grid_search.best_score_:.4f}")

# 최적 모델로 재학습
tuned_model = grid_search.best_estimator_
y_val_pred_tuned = tuned_model.predict(X_val)

mae_tuned = mean_absolute_error(y_val, y_val_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))
r2_tuned = r2_score(y_val, y_val_pred_tuned)

print(f"\n튜닝 후 Validation MAE: {mae_tuned:.4f}")
print(f"튜닝 후 Validation RMSE: {rmse_tuned:.4f}")
print(f"튜닝 후 Validation R²: {r2_tuned:.4f}")

# 튜닝된 모델이 더 나으면 교체
if mae_tuned < models_performance[best_model_name]['MAE']:
    print("✓ 튜닝된 모델이 더 우수합니다!")
    best_model = tuned_model
    y_val_pred_best = y_val_pred_tuned
else:
    print("기존 모델을 유지합니다.")

# ====== 7. 최종 모델 학습 (2023 + 2024) ======
print("\n[Phase 7] 최종 모델 학습 (2023 + 2024 데이터)")
print("-" * 80)

X_final_train = pd.concat([X_train, X_val])
y_final_train = pd.concat([y_train, y_val])

print(f"최종 학습 데이터: {X_final_train.shape}")

# 최적 파라미터로 최종 모델 학습
final_model = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_final_train, y_final_train)
print("최종 모델 학습 완료!")

# ====== 8. 2025년 예측 ======
print("\n[Phase 8] 2025년 예측")
print("-" * 80)

# 8.1 2025년 트렌드 추정
print("8.1 2025년 트렌드 추정...")

avg_2023 = train_data['관중수'].mean()
avg_2024 = val_data['관중수'].mean()
growth_rate = avg_2024 / avg_2023

print(f"2023년 평균 관중: {avg_2023:.0f}명")
print(f"2024년 평균 관중: {avg_2024:.0f}명")
print(f"증가율: {(growth_rate-1)*100:.1f}%")

# 2025년 추정 기본값 (성장 추세 반영)
baseline_2025_estimated = avg_2024 * growth_rate
print(f"2025년 추정 기본값: {baseline_2025_estimated:.0f}명")

# 실제 2025년 트렌드 (오차 분석용)
baseline_2025_actual = test_data['이동평균_30'].mean()
print(f"2025년 실제 기본값: {baseline_2025_actual:.0f}명")

# 8.2 비율 예측
print("\n8.2 관중 비율 예측...")
y_test_pred_ratio = final_model.predict(X_test)

print(f"예측 비율 범위: {y_test_pred_ratio.min():.3f} ~ {y_test_pred_ratio.max():.3f}")
print(f"예측 비율 평균: {y_test_pred_ratio.mean():.3f}")

# 8.3 실제 관중 수 변환
print("\n8.3 실제 관중 수 변환...")

# 방법 1: 추정 트렌드 사용 (제출용)
test_data['예측관중_추정트렌드'] = y_test_pred_ratio * baseline_2025_estimated

# 방법 2: 실제 이동평균 사용 (오차 분석용)
test_data['예측관중_실제트렌드'] = y_test_pred_ratio * test_data['이동평균_30']

print(f"추정 트렌드 기반 예측 평균: {test_data['예측관중_추정트렌드'].mean():.0f}명")
print(f"실제 트렌드 기반 예측 평균: {test_data['예측관중_실제트렌드'].mean():.0f}명")
print(f"실제 관중 평균: {test_data['관중수'].mean():.0f}명")

# ====== 9. 평가 및 오차 분석 ======
print("\n[Phase 9] 평가 및 오차 분석")
print("-" * 80)

# 9.1 기본 평가 지표
print("9.1 기본 평가 지표 (비율 기준)")

mae_ratio = mean_absolute_error(y_test, y_test_pred_ratio)
rmse_ratio = np.sqrt(mean_squared_error(y_test, y_test_pred_ratio))
r2_ratio = r2_score(y_test, y_test_pred_ratio)
mape_ratio = np.mean(np.abs((y_test - y_test_pred_ratio) / y_test)) * 100

print(f"MAE (비율): {mae_ratio:.4f} (평균 {mae_ratio*100:.1f}% 오차)")
print(f"RMSE (비율): {rmse_ratio:.4f}")
print(f"R² Score: {r2_ratio:.4f}")
print(f"MAPE: {mape_ratio:.2f}%")

# 9.2 관중 수 기준 평가
print("\n9.2 관중 수 기준 평가 (추정 트렌드)")

A = test_data['관중수'].values  # 실제 관중 수
B = test_data['예측관중_추정트렌드'].values  # 추정 트렌드 기반 예측
C = test_data['예측관중_실제트렌드'].values  # 실제 트렌드 기반 예측

mae_attendance = mean_absolute_error(A, B)
rmse_attendance = np.sqrt(mean_squared_error(A, B))
r2_attendance = r2_score(A, B)
mape_attendance = np.mean(np.abs((A - B) / A)) * 100

print(f"MAE (관중수): {mae_attendance:.0f}명")
print(f"RMSE (관중수): {rmse_attendance:.0f}명")
print(f"R² Score: {r2_attendance:.4f}")
print(f"MAPE: {mape_attendance:.2f}%")

# 9.3 오차 분해 분석 (핵심!)
print("\n9.3 오차 분해 분석")
print("-" * 40)

total_error = A - B  # 전체 오차
trend_error = C - B  # 트렌드 추정 오차
model_error = A - C  # 모델 패턴 오차

mae_total = np.mean(np.abs(total_error))
mae_trend = np.mean(np.abs(trend_error))
mae_model = np.mean(np.abs(model_error))

print(f"총 오차 (MAE): {mae_total:.0f}명")
print(f"  ├─ 트렌드 추정 오차: {mae_trend:.0f}명 ({mae_trend/mae_total*100:.1f}%)")
print(f"  └─ 모델 패턴 오차: {mae_model:.0f}명 ({mae_model/mae_total*100:.1f}%)")

print(f"\n트렌드 추정 정확도: {baseline_2025_estimated/baseline_2025_actual*100:.1f}%")
print(f"  (추정: {baseline_2025_estimated:.0f}명, 실제: {baseline_2025_actual:.0f}명)")

# ====== 10. Feature Importance 분석 ======
print("\n[Phase 10] Feature Importance 분석")
print("-" * 80)

# Feature Importance 추출
importances = final_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 15 중요 Features:")
print(feature_importance_df.head(15).to_string(index=False))

# ====== 11. 결과 요약 ======
print("\n" + "=" * 80)
print("최종 결과 요약")
print("=" * 80)

print(f"\n[모델 정보]")
print(f"  - 최종 모델: Random Forest Regressor")
print(f"  - 학습 데이터: 2023-2024년 ({len(X_final_train)}경기)")
print(f"  - 테스트 데이터: 2025년 ({len(X_test)}경기)")
print(f"  - Feature 수: {len(feature_columns)}개")

print(f"\n[성능 지표 - 관중 비율]")
print(f"  - MAE: {mae_ratio:.4f} ({mae_ratio*100:.1f}% 오차)")
print(f"  - RMSE: {rmse_ratio:.4f}")
print(f"  - R²: {r2_ratio:.4f}")
print(f"  - MAPE: {mape_ratio:.2f}%")

print(f"\n[성능 지표 - 관중 수]")
print(f"  - MAE: {mae_attendance:.0f}명")
print(f"  - RMSE: {rmse_attendance:.0f}명")
print(f"  - R²: {r2_attendance:.4f}")
print(f"  - MAPE: {mape_attendance:.2f}%")

print(f"\n[오차 분석]")
print(f"  - 총 오차: {mae_total:.0f}명")
print(f"    ├─ 트렌드 추정 오차: {mae_trend:.0f}명 ({mae_trend/mae_total*100:.1f}%)")
print(f"    └─ 모델 패턴 오차: {mae_model:.0f}명 ({mae_model/mae_total*100:.1f}%)")

print(f"\n[Top 5 중요 Features]")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)

# 저장할 데이터 준비
results = {
    'test_data': test_data,
    'feature_importance': feature_importance_df,
    'final_model': final_model,
    'metrics': {
        'mae_ratio': mae_ratio,
        'rmse_ratio': rmse_ratio,
        'r2_ratio': r2_ratio,
        'mape_ratio': mape_ratio,
        'mae_attendance': mae_attendance,
        'rmse_attendance': rmse_attendance,
        'r2_attendance': r2_attendance,
        'mape_attendance': mape_attendance,
        'mae_total': mae_total,
        'mae_trend': mae_trend,
        'mae_model': mae_model
    }
}

print("\n결과 데이터 준비 완료. 시각화를 위해 results 딕셔너리를 사용하세요.")
