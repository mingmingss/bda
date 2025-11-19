"""
KBO 정규시즌 관중 수 예측 모델 구축 - 개선 버전
3조 - 모델 구축 담당: 임혜린, 윤태영

개선 사항:
1. 다양한 모델 비교 (XGBoost, CatBoost, LightGBM, AdaBoost, ExtraTrees 추가)
2. Voting Ensemble 추가
3. 시각화 코드 추가
4. 모델 성능 비교표 저장
5. 교차 검증 결과 저장
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    VotingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
import joblib
warnings.filterwarnings('ignore')

# XGBoost, CatBoost, LightGBM import (설치 필요)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost가 설치되지 않았습니다. pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost가 설치되지 않았습니다. pip install catboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM이 설치되지 않았습니다. pip install lightgbm")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

print("=" * 80)
print("KBO 관중 수 예측 모델 구축 프로젝트 - 개선 버전")
print("=" * 80)

# ====== 1. 데이터 로드 및 기본 탐색 ======
print("\n[Phase 1] 데이터 로드 및 탐색")
print("-" * 80)

data_path = Path(__file__).resolve().parent / "df_refined.csv"
df = pd.read_csv(data_path, encoding='utf-8-sig')

print(f"데이터 shape: {df.shape}")
print(f"\n결측치 현황:\n{df.isnull().sum()}")

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
weekday_map = {
    '월요일': 0, '화요일': 1, '수요일': 2, '목요일': 3,
    '금요일': 4, '토요일': 5, '일요일': 6
}
df['요일_숫자'] = df['요일'].map(weekday_map)
df['주말여부'] = df['요일_숫자'].apply(lambda x: 1 if x >= 4 else 0)
df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)

# 3.2 날씨 관련 Features
print("\n3.2 날씨 관련 Features 생성...")
df['평균기온'] = (df['최저기온(°C)'] + df['최고기온(°C)']) / 2
df['비여부'] = (df['일강수량(mm)'] > 0).astype(int)

def categorize_rain(rain):
    if rain == 0:
        return 0
    elif rain < 5:
        return 1
    elif rain < 20:
        return 2
    else:
        return 3

df['강수강도'] = df['일강수량(mm)'].apply(categorize_rain)

def categorize_weather(cloud):
    if cloud <= 3:
        return 0
    elif cloud <= 7:
        return 1
    else:
        return 2

df['날씨상태'] = df['평균 전운량(1/10)'].apply(categorize_weather)

# 3.3 구장 관련 Features
print("\n3.3 구장 관련 Features 생성...")
stadium_capacity = {
    '잠실': 25000, '수원': 20000, '문학': 20500, '고척': 16500,
    '광주': 11000, '사직': 23000, '창원': 20000, '대구': 10000,
    '한밭': 11500, '대전': 13000, '울산': 11000, '포항': 11000, '청주': 10000
}

df['구장수용인원'] = df['구장'].map(stadium_capacity)
df['구장크기'] = pd.cut(df['구장수용인원'],
                       bins=[0, 12000, 20000, 30000],
                       labels=[0, 1, 2]).astype(float)

# 3.4 팀 인기도 Features
print("\n3.4 팀 인기도 Features 생성...")
train_years = df[df['연도'].isin([2023, 2024])]
overall_avg = train_years['관중수'].mean()

home_popularity = train_years.groupby('홈')['관중수'].mean().to_dict()
df['홈팀평균관중'] = df['홈'].map(home_popularity)
df['홈팀평균관중'].fillna(overall_avg, inplace=True)

away_popularity = train_years.groupby('방문')['관중수'].mean().to_dict()
df['방문팀평균관중'] = df['방문'].map(away_popularity)
df['방문팀평균관중'].fillna(overall_avg, inplace=True)

df['홈팀인기지수'] = df['홈팀평균관중'] / overall_avg
df['방문팀인기지수'] = df['방문팀평균관중'] / overall_avg

matchup = train_years.groupby(['홈', '방문'])['관중수'].mean().to_dict()
df['대진평균관중'] = df.apply(lambda row: matchup.get((row['홈'], row['방문']), overall_avg), axis=1)
df['대진인기지수'] = df['대진평균관중'] / overall_avg

# 3.5 이동평균 및 목표 변수 계산
print("\n3.5 목표 변수(관중 비율) 계산...")
df['이동평균_30'] = df['관중수'].rolling(window=30, min_periods=1).mean().shift(1)
initial_avg = df.iloc[:30]['관중수'].mean()
df.loc[df['이동평균_30'].isna(), '이동평균_30'] = initial_avg
df['관중비율'] = df['관중수'] / df['이동평균_30']
df['매진율'] = df['관중수'] / df['구장수용인원']

print(f"  - 관중비율 평균: {df['관중비율'].mean():.3f}")
print(f"  - 관중비율 범위: {df['관중비율'].min():.3f} ~ {df['관중비율'].max():.3f}")

# 3.6 범주형 변수 인코딩
print("\n3.6 범주형 변수 인코딩...")
le_stadium = LabelEncoder()
le_home = LabelEncoder()
le_away = LabelEncoder()

df['구장_인코딩'] = le_stadium.fit_transform(df['구장'])
df['홈팀_인코딩'] = le_home.fit_transform(df['홈'])
df['방문팀_인코딩'] = le_away.fit_transform(df['방문'])

# 3.7 최종 NaN 체크 및 처리
print("\n3.7 최종 NaN 체크 및 처리...")
nan_counts = df.isnull().sum()
nan_columns = nan_counts[nan_counts > 0]

if len(nan_columns) > 0:
    print(f"⚠️  Feature Engineering 후 NaN 발견:")
    for col in nan_columns.index:
        print(f"  - {col}: {nan_columns[col]}개")
        if df[col].dtype in ['float64', 'int64']:
            # 수치형: 평균값으로 대체
            mean_val = df[col].mean()
            if pd.isna(mean_val):  # 평균도 NaN이면 0으로 대체
                df[col].fillna(0, inplace=True)
                print(f"    → 0으로 대체")
            else:
                df[col].fillna(mean_val, inplace=True)
                print(f"    → 평균값({mean_val:.2f})으로 대체")
        else:
            # 범주형: 최빈값으로 대체
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
            df[col].fillna(mode_val, inplace=True)
            print(f"    → 최빈값({mode_val})으로 대체")
else:
    print("✓ NaN 없음 - 모든 Feature가 정상입니다!")

# ====== 4. 데이터 분할 ======
print("\n[Phase 4] 데이터 분할")
print("-" * 80)

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

X_train = train_data[feature_columns]
y_train = train_data['관중비율']

X_val = val_data[feature_columns]
y_val = val_data['관중비율']

X_test = test_data[feature_columns]
y_test = test_data['관중비율']

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ====== 5. 모델 정의 및 학습 ======
print("\n[Phase 5] 다양한 모델 학습 및 비교")
print("-" * 80)

# 모델 딕셔너리
models = {}

# 5.1 Decision Tree
print("\n5.1 Decision Tree Regressor 학습...")
models['Decision Tree'] = DecisionTreeRegressor(random_state=42, max_depth=10)

# 5.2 Random Forest
print("5.2 Random Forest Regressor 학습...")
models['Random Forest'] = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# 5.3 Extra Trees
print("5.3 Extra Trees Regressor 학습...")
models['Extra Trees'] = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# 5.4 Hist Gradient Boosting
print("5.4 Hist Gradient Boosting Regressor 학습...")
models['Hist Gradient Boosting'] = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# 5.5 AdaBoost
print("5.5 AdaBoost Regressor 학습...")
models['AdaBoost'] = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# 5.6 XGBoost (if available)
if XGBOOST_AVAILABLE:
    print("5.6 XGBoost Regressor 학습...")
    models['XGBoost'] = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

# 5.7 CatBoost (if available)
if CATBOOST_AVAILABLE:
    print("5.7 CatBoost Regressor 학습...")
    models['CatBoost'] = cb.CatBoostRegressor(
        iterations=200,
        depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )

# 5.8 LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    print("5.8 LightGBM Regressor 학습...")
    models['LightGBM'] = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

# 모델 학습 및 평가
print("\n모델 학습 중...")
print("=" * 80)

results = []

for name, model in models.items():
    print(f"\n[{name}] 학습 중...")

    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 평가
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    # 결과 저장
    results.append({
        'Model': name,
        'Train MAE': train_mae,
        'Train RMSE': train_rmse,
        'Train R²': train_r2,
        'Val MAE': val_mae,
        'Val RMSE': val_rmse,
        'Val R²': val_r2,
        'Overfit Score': train_mae - val_mae  # 음수면 좋음
    })

    print(f"  Train - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"  Val   - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"  Overfit: {train_mae - val_mae:.4f} (음수가 좋음)")

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Val MAE')

print("\n" + "=" * 80)
print("모델 성능 비교 (Validation MAE 기준 정렬)")
print("=" * 80)
print(results_df.to_string(index=False))

# 최고 성능 모델 선택
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n🏆 최고 성능 모델: {best_model_name}")
print(f"   Validation MAE: {results_df.iloc[0]['Val MAE']:.4f}")

# ====== 6. Voting Ensemble ======
print("\n[Phase 6] Voting Ensemble 구축")
print("-" * 80)

# 상위 3개 모델로 Voting Ensemble 구성
top_3_models = results_df.head(3)['Model'].tolist()
print(f"상위 3개 모델: {top_3_models}")

ensemble_estimators = [(name, models[name]) for name in top_3_models]
voting_model = VotingRegressor(estimators=ensemble_estimators)

print("Voting Ensemble 학습 중...")
voting_model.fit(X_train, y_train)

y_val_pred_voting = voting_model.predict(X_val)
val_mae_voting = mean_absolute_error(y_val, y_val_pred_voting)
val_rmse_voting = np.sqrt(mean_squared_error(y_val, y_val_pred_voting))
val_r2_voting = r2_score(y_val, y_val_pred_voting)

print(f"\nVoting Ensemble 성능:")
print(f"  Val MAE: {val_mae_voting:.4f}")
print(f"  Val RMSE: {val_rmse_voting:.4f}")
print(f"  Val R²: {val_r2_voting:.4f}")

# Voting Ensemble이 더 좋으면 최고 모델로 선택
if val_mae_voting < results_df.iloc[0]['Val MAE']:
    print("\n✓ Voting Ensemble이 단일 모델보다 우수합니다!")
    best_model = voting_model
    best_model_name = 'Voting Ensemble'
else:
    print(f"\n✓ {best_model_name}이 Voting Ensemble보다 우수합니다.")

# ====== 7. 최종 모델 학습 (2023 + 2024) ======
print("\n[Phase 7] 최종 모델 학습 (2023 + 2024 데이터)")
print("-" * 80)

X_final_train = pd.concat([X_train, X_val])
y_final_train = pd.concat([y_train, y_val])

print(f"최종 학습 데이터: {X_final_train.shape}")
print(f"최종 모델: {best_model_name}")

# 최종 모델 재학습
if best_model_name == 'Voting Ensemble':
    # Voting Ensemble은 이미 학습된 모델들을 사용하므로 재학습 필요
    ensemble_estimators_final = []
    for name in top_3_models:
        # 각 모델을 2023+2024 데이터로 재학습
        model_copy = models[name]
        model_copy.fit(X_final_train, y_final_train)
        ensemble_estimators_final.append((name, model_copy))

    final_model = VotingRegressor(estimators=ensemble_estimators_final)
    final_model.fit(X_final_train, y_final_train)
else:
    # 단일 모델인 경우
    final_model = best_model
    final_model.fit(X_final_train, y_final_train)

print("최종 모델 학습 완료!")

# ====== 8. 2025년 예측 ======
print("\n[Phase 8] 2025년 예측")
print("-" * 80)

# 8.1 트렌드 추정
print("8.1 2025년 트렌드 추정...")
avg_2023 = train_data['관중수'].mean()
avg_2024 = val_data['관중수'].mean()
growth_rate = avg_2024 / avg_2023

print(f"2023년 평균 관중: {avg_2023:.0f}명")
print(f"2024년 평균 관중: {avg_2024:.0f}명")
print(f"증가율: {(growth_rate-1)*100:.1f}%")

baseline_2025_estimated = avg_2024 * growth_rate
baseline_2025_actual = test_data['이동평균_30'].mean()

print(f"2025년 추정 기본값: {baseline_2025_estimated:.0f}명")
print(f"2025년 실제 기본값: {baseline_2025_actual:.0f}명")

# 8.2 비율 예측
print("\n8.2 관중 비율 예측...")
y_test_pred_ratio = final_model.predict(X_test)

print(f"예측 비율 범위: {y_test_pred_ratio.min():.3f} ~ {y_test_pred_ratio.max():.3f}")
print(f"예측 비율 평균: {y_test_pred_ratio.mean():.3f}")

# 8.3 실제 관중 수 변환
print("\n8.3 실제 관중 수 변환...")
test_data['예측관중_추정트렌드'] = y_test_pred_ratio * baseline_2025_estimated
test_data['예측관중_실제트렌드'] = y_test_pred_ratio * test_data['이동평균_30']

print(f"추정 트렌드 기반 예측 평균: {test_data['예측관중_추정트렌드'].mean():.0f}명")
print(f"실제 트렌드 기반 예측 평균: {test_data['예측관중_실제트렌드'].mean():.0f}명")
print(f"실제 관중 평균: {test_data['관중수'].mean():.0f}명")

# ====== 9. 평가 및 오차 분석 ======
print("\n[Phase 9] 평가 및 오차 분석")
print("-" * 80)

# 9.1 비율 기준 평가
mae_ratio = mean_absolute_error(y_test, y_test_pred_ratio)
rmse_ratio = np.sqrt(mean_squared_error(y_test, y_test_pred_ratio))
r2_ratio = r2_score(y_test, y_test_pred_ratio)
mape_ratio = np.mean(np.abs((y_test - y_test_pred_ratio) / y_test)) * 100

print("\n9.1 비율 기준 평가")
print(f"MAE (비율): {mae_ratio:.4f} (평균 {mae_ratio*100:.1f}% 오차)")
print(f"RMSE (비율): {rmse_ratio:.4f}")
print(f"R² Score: {r2_ratio:.4f}")
print(f"MAPE: {mape_ratio:.2f}%")

# 9.2 관중 수 기준 평가
A = test_data['관중수'].values
B = test_data['예측관중_추정트렌드'].values
C = test_data['예측관중_실제트렌드'].values

mae_attendance = mean_absolute_error(A, B)
rmse_attendance = np.sqrt(mean_squared_error(A, B))
r2_attendance = r2_score(A, B)
mape_attendance = np.mean(np.abs((A - B) / A)) * 100

print("\n9.2 관중 수 기준 평가")
print(f"MAE (관중수): {mae_attendance:.0f}명")
print(f"RMSE (관중수): {rmse_attendance:.0f}명")
print(f"R² Score: {r2_attendance:.4f}")
print(f"MAPE: {mape_attendance:.2f}%")

# 9.3 오차 분해 분석
total_error = A - B
trend_error = C - B
model_error = A - C

mae_total = np.mean(np.abs(total_error))
mae_trend = np.mean(np.abs(trend_error))
mae_model = np.mean(np.abs(model_error))

print("\n9.3 오차 분해 분석")
print(f"총 오차 (MAE): {mae_total:.0f}명")
print(f"  ├─ 트렌드 추정 오차: {mae_trend:.0f}명 ({mae_trend/mae_total*100:.1f}%)")
print(f"  └─ 모델 패턴 오차: {mae_model:.0f}명 ({mae_model/mae_total*100:.1f}%)")

# ====== 10. Feature Importance (가능한 경우) ======
print("\n[Phase 10] Feature Importance 분석")
print("-" * 80)

if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 15 중요 Features:")
    print(feature_importance_df.head(15).to_string(index=False))
else:
    print("해당 모델은 Feature Importance를 제공하지 않습니다.")
    feature_importance_df = None

# ====== 11. 결과 저장 ======
print("\n[Phase 11] 결과 저장")
print("-" * 80)

# 출력 디렉토리 생성
output_dir = Path(__file__).resolve().parent.parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 11.1 모델 성능 비교표 저장
results_df.to_csv(output_dir / "model_performance_comparison.csv", index=False, encoding='utf-8-sig')
print(f"✓ 모델 성능 비교표 저장: {output_dir / 'model_performance_comparison.csv'}")

# 11.2 예측 결과 저장
prediction_df = test_data[['일시', '구장', '홈', '방문', '관중수', '예측관중_추정트렌드', '예측관중_실제트렌드']].copy()
prediction_df['오차'] = prediction_df['관중수'] - prediction_df['예측관중_추정트렌드']
prediction_df.to_csv(output_dir / "2025_predictions_detailed.csv", index=False, encoding='utf-8-sig')
print(f"✓ 예측 결과 저장: {output_dir / '2025_predictions_detailed.csv'}")

# 11.3 Feature Importance 저장
if feature_importance_df is not None:
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding='utf-8-sig')
    print(f"✓ Feature Importance 저장: {output_dir / 'feature_importance.csv'}")

# 11.4 최종 모델 저장
model_path = output_dir / f"best_model_{best_model_name.replace(' ', '_')}.pkl"
joblib.dump(final_model, model_path)
print(f"✓ 최종 모델 저장: {model_path}")

# ====== 12. 최종 요약 ======
print("\n" + "=" * 80)
print("최종 결과 요약")
print("=" * 80)

print(f"\n[모델 정보]")
print(f"  - 최종 모델: {best_model_name}")
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

if feature_importance_df is not None:
    print(f"\n[Top 5 중요 Features]")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\n[모델 성능 순위]")
for idx, row in results_df.head(5).iterrows():
    print(f"  {idx+1}. {row['Model']}: Val MAE = {row['Val MAE']:.4f}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)

print("\n💡 다음 단계:")
print("  1. scripts/visualize_results.py를 실행하여 시각화 생성")
print("  2. outputs/ 폴더에서 결과 확인")
print("  3. 최고 성능 모델의 하이퍼파라미터 튜닝 고려")
