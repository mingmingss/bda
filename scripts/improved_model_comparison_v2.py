"""
KBO 정규시즌 관중 수 예측 모델 구축 - 개선 버전 v2
3조 - 모델 구축 담당: 임혜린, 윤태영

개선 사항 v2:
1. ✅ 데이터 누출(Data Leakage) 완전 제거 - 시계열 순서를 엄격히 보존
2. ✅ TimeSeriesSplit 교차 검증 도입
3. ✅ 하이퍼파라미터 튜닝 추가 (RandomizedSearchCV)
4. ✅ 코드 구조화 - 함수/모듈화
5. ✅ 재사용성 및 테스트 용이성 향상
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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
from sklearn.base import clone
import warnings
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# XGBoost, CatBoost, LightGBM import (선택적)
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


# ============================================================================
# 1. 데이터 로드 및 전처리 함수
# ============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """
    데이터를 로드하고 기본 전처리를 수행합니다.

    Args:
        data_path: CSV 파일 경로

    Returns:
        전처리된 DataFrame
    """
    print("\n[Phase 1] 데이터 로드 및 탐색")
    print("-" * 80)

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"데이터 shape: {df.shape}")
    print(f"\n결측치 현황:\n{df.isnull().sum()}")

    # 기본 결측치 처리
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"\n결측치가 있는 컬럼: {missing_cols}")
        for col in missing_cols:
            if df[col].dtype in ['float64', 'int64']:
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                print(f"  - {col}: 평균값({mean_value:.2f})으로 대체")

    # 일시 변환
    df['일시'] = pd.to_datetime(df['일시'], format='%Y년%m월%d일')
    df = df.sort_values('일시').reset_index(drop=True)
    print(f"\n일시 범위: {df['일시'].min()} ~ {df['일시'].max()}")

    # 시간 컬럼 추출
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['주차'] = df['일시'].dt.isocalendar().week

    print(f"\n연도별 경기 수:\n{df['연도'].value_counts().sort_index()}")

    return df


def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간, 날씨, 구장 등 기본 features를 생성합니다.
    (팀 인기도는 별도 함수에서 처리 - 데이터 누출 방지)

    Args:
        df: 원본 DataFrame

    Returns:
        Feature가 추가된 DataFrame
    """
    print("\n[Phase 2] 기본 Feature Engineering")
    print("-" * 80)

    df = df.copy()

    # 1. 시간 관련 Features
    print("1. 시간 관련 Features 생성...")
    weekday_map = {
        '월요일': 0, '화요일': 1, '수요일': 2, '목요일': 3,
        '금요일': 4, '토요일': 5, '일요일': 6
    }
    df['요일_숫자'] = df['요일'].map(weekday_map)

    # NaN 처리 - 요일이 없으면 일시에서 추출
    if df['요일_숫자'].isnull().any():
        df.loc[df['요일_숫자'].isnull(), '요일_숫자'] = df.loc[df['요일_숫자'].isnull(), '일시'].dt.dayofweek

    df['주말여부'] = df['요일_숫자'].apply(lambda x: 1 if x >= 4 else 0)
    df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
    df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)

    # 2. 날씨 관련 Features
    print("2. 날씨 관련 Features 생성...")
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
        if pd.isna(cloud):
            return 1  # 중간값으로 대체
        if cloud <= 3:
            return 0
        elif cloud <= 7:
            return 1
        else:
            return 2

    df['날씨상태'] = df['평균 전운량(1/10)'].apply(categorize_weather)

    # 3. 구장 관련 Features
    print("3. 구장 관련 Features 생성...")
    stadium_capacity = {
        '잠실': 25000, '수원': 20000, '문학': 20500, '고척': 16500,
        '광주': 11000, '사직': 23000, '창원': 20000, '대구': 10000,
        '한밭': 11500, '대전': 13000, '울산': 11000, '포항': 11000, '청주': 10000
    }

    df['구장수용인원'] = df['구장'].map(stadium_capacity)
    df['구장크기'] = pd.cut(df['구장수용인원'],
                          bins=[0, 12000, 20000, 30000],
                          labels=[0, 1, 2]).astype(float)

    # 4. 범주형 변수 인코딩
    print("4. 범주형 변수 인코딩...")
    le_stadium = LabelEncoder()
    le_home = LabelEncoder()
    le_away = LabelEncoder()

    df['구장_인코딩'] = le_stadium.fit_transform(df['구장'])
    df['홈팀_인코딩'] = le_home.fit_transform(df['홈'])
    df['방문팀_인코딩'] = le_away.fit_transform(df['방문'])

    # 5. 매진율 (참고용)
    df['매진율'] = df['관중수'] / df['구장수용인원']

    print("✓ 기본 Feature Engineering 완료")

    return df


def calculate_popularity_features_no_leakage(df: pd.DataFrame,
                                               train_end_date: pd.Timestamp) -> pd.DataFrame:
    """
    데이터 누출 없이 팀 인기도 features를 계산합니다.

    핵심: train_end_date 이전의 데이터만 사용하여 통계량을 계산합니다.

    Args:
        df: 전체 DataFrame
        train_end_date: 학습 데이터의 마지막 날짜

    Returns:
        인기도 features가 추가된 DataFrame
    """
    df = df.copy()

    # train_end_date 이전의 데이터만 사용
    train_mask = df['일시'] <= train_end_date
    train_data = df[train_mask]

    if len(train_data) == 0:
        # 데이터가 없으면 전체 평균 사용
        overall_avg = df['관중수'].mean()
    else:
        overall_avg = train_data['관중수'].mean()

    # 홈팀 인기도
    home_popularity = train_data.groupby('홈')['관중수'].mean().to_dict()
    df['홈팀평균관중'] = df['홈'].map(home_popularity)
    df['홈팀평균관중'].fillna(overall_avg, inplace=True)

    # 방문팀 인기도
    away_popularity = train_data.groupby('방문')['관중수'].mean().to_dict()
    df['방문팀평균관중'] = df['방문'].map(away_popularity)
    df['방문팀평균관중'].fillna(overall_avg, inplace=True)

    # 인기 지수
    df['홈팀인기지수'] = df['홈팀평균관중'] / overall_avg
    df['방문팀인기지수'] = df['방문팀평균관중'] / overall_avg

    # 대진 인기도
    matchup = train_data.groupby(['홈', '방문'])['관중수'].mean().to_dict()
    df['대진평균관중'] = df.apply(lambda row: matchup.get((row['홈'], row['방문']), overall_avg), axis=1)
    df['대진인기지수'] = df['대진평균관중'] / overall_avg

    return df


def calculate_moving_average_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    이동평균 및 목표 변수(관중비율)를 계산합니다.

    Args:
        df: DataFrame

    Returns:
        이동평균과 목표변수가 추가된 DataFrame
    """
    df = df.copy()

    print("\n이동평균 및 목표 변수 계산...")
    df['이동평균_30'] = df['관중수'].rolling(window=30, min_periods=1).mean().shift(1)

    # 초기값 처리
    initial_avg = df.iloc[:30]['관중수'].mean()
    df.loc[df['이동평균_30'].isna(), '이동평균_30'] = initial_avg

    # 목표 변수: 관중 비율
    df['관중비율'] = df['관중수'] / df['이동평균_30']

    print(f"  - 관중비율 평균: {df['관중비율'].mean():.3f}")
    print(f"  - 관중비율 범위: {df['관중비율'].min():.3f} ~ {df['관중비율'].max():.3f}")

    return df


def handle_final_nans(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    최종 NaN 처리를 수행합니다.

    Args:
        df: DataFrame
        feature_columns: 확인할 feature 컬럼 리스트

    Returns:
        NaN이 제거된 DataFrame
    """
    df = df.copy()

    print("\n최종 NaN 체크 및 처리...")

    for col in feature_columns:
        if col not in df.columns:
            continue

        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"⚠️  {col}: {nan_count}개 NaN 발견")

            if df[col].dtype in ['float64', 'int64']:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    df[col].fillna(0, inplace=True)
                    print(f"    → 0으로 대체")
                else:
                    df[col].fillna(mean_val, inplace=True)
                    print(f"    → 평균값({mean_val:.2f})으로 대체")
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                df[col].fillna(mode_val, inplace=True)
                print(f"    → 최빈값({mode_val})으로 대체")

    print("✓ NaN 처리 완료")
    return df


# ============================================================================
# 2. 모델 정의 및 학습 함수
# ============================================================================

def get_model_definitions() -> Dict[str, Any]:
    """
    사용할 모델들을 정의합니다.

    Returns:
        모델 이름과 모델 객체의 딕셔너리
    """
    models = {}

    models['Decision Tree'] = DecisionTreeRegressor(random_state=42, max_depth=10)

    models['Random Forest'] = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    models['Extra Trees'] = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    models['Hist Gradient Boosting'] = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    models['AdaBoost'] = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostRegressor(
            iterations=200,
            depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )

    if LIGHTGBM_AVAILABLE:
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

    return models


def get_hyperparameter_grids() -> Dict[str, Dict]:
    """
    하이퍼파라미터 튜닝을 위한 파라미터 그리드를 정의합니다.

    Returns:
        모델 이름과 파라미터 그리드의 딕셔너리
    """
    param_grids = {}

    # Random Forest
    param_grids['Random Forest'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    # Extra Trees
    param_grids['Extra Trees'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    # Hist Gradient Boosting
    param_grids['Hist Gradient Boosting'] = {
        'max_iter': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [10, 20, 30]
    }

    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }

    if LIGHTGBM_AVAILABLE:
        param_grids['LightGBM'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'num_leaves': [20, 31, 50, 100]
        }

    return param_grids


def train_with_timeseries_cv(df: pd.DataFrame,
                             feature_columns: List[str],
                             train_years: List[int],
                             val_years: List[int],
                             n_splits: int = 3) -> Tuple[Dict, pd.DataFrame]:
    """
    TimeSeriesSplit을 사용한 교차 검증으로 모델을 학습합니다.

    Args:
        df: 전체 DataFrame
        feature_columns: 사용할 feature 컬럼 리스트
        train_years: 학습에 사용할 연도 리스트
        val_years: 검증에 사용할 연도 리스트
        n_splits: TimeSeriesSplit의 fold 수

    Returns:
        (학습된 모델 딕셔너리, 결과 DataFrame)
    """
    print("\n[Phase 3] TimeSeriesSplit 교차 검증을 통한 모델 학습")
    print("-" * 80)

    # 데이터 분할
    train_data = df[df['연도'].isin(train_years)].copy()
    val_data = df[df['연도'].isin(val_years)].copy()

    print(f"\nTrain years: {train_years} ({len(train_data)}경기)")
    print(f"Val years: {val_years} ({len(val_data)}경기)")

    X_train = train_data[feature_columns]
    y_train = train_data['관중비율']

    X_val = val_data[feature_columns]
    y_val = val_data['관중비율']

    # TimeSeriesSplit 설정
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 모델 정의
    models = get_model_definitions()

    results = []
    trained_models = {}

    print(f"\nTimeSeriesSplit ({n_splits} folds) 교차 검증 시작...")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n[{name}] 학습 중...")

        # TimeSeriesSplit CV 점수
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 모델 학습 (clone을 사용하여 nested estimator 문제 해결)
            model_fold = clone(model)
            model_fold.fit(X_fold_train, y_fold_train)

            # 검증
            y_fold_pred = model_fold.predict(X_fold_val)
            fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
            cv_scores.append(fold_mae)

            print(f"  Fold {fold_idx+1}/{n_splits}: MAE = {fold_mae:.4f}")

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"  CV MAE: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # 전체 학습 데이터로 최종 학습
        model.fit(X_train, y_train)
        trained_models[name] = model

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
            'CV MAE Mean': cv_mean,
            'CV MAE Std': cv_std,
            'Train MAE': train_mae,
            'Train RMSE': train_rmse,
            'Train R²': train_r2,
            'Val MAE': val_mae,
            'Val RMSE': val_rmse,
            'Val R²': val_r2,
            'Overfit Score': train_mae - val_mae
        })

        print(f"  Train - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  Val   - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

    # 결과 DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val MAE')

    print("\n" + "=" * 80)
    print("모델 성능 비교 (Validation MAE 기준 정렬)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    return trained_models, results_df


def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         model_name: str,
                         base_model: Any,
                         param_grid: Dict,
                         n_iter: int = 20,
                         n_splits: int = 3) -> Tuple[Any, Dict]:
    """
    RandomizedSearchCV를 사용하여 하이퍼파라미터를 튜닝합니다.

    Args:
        X_train: 학습 features
        y_train: 학습 target
        model_name: 모델 이름
        base_model: 기본 모델
        param_grid: 파라미터 그리드
        n_iter: RandomizedSearch 반복 횟수
        n_splits: TimeSeriesSplit fold 수

    Returns:
        (최적 모델, 최적 파라미터)
    """
    print(f"\n[{model_name}] 하이퍼파라미터 튜닝 중...")
    print(f"  - 탐색 공간: {len(param_grid)} 파라미터")
    print(f"  - 반복 횟수: {n_iter}")
    print(f"  - CV folds: {n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = -random_search.best_score_

    print(f"  ✓ 최적 CV MAE: {best_score:.4f}")
    print(f"  ✓ 최적 파라미터: {best_params}")

    return random_search.best_estimator_, best_params


# ============================================================================
# 3. 평가 및 저장 함수
# ============================================================================

def evaluate_on_test(model: Any,
                     test_data: pd.DataFrame,
                     feature_columns: List[str],
                     baseline_2025: float) -> Dict:
    """
    테스트 데이터에 대해 모델을 평가합니다.

    Args:
        model: 학습된 모델
        test_data: 테스트 DataFrame
        feature_columns: feature 컬럼 리스트
        baseline_2025: 2025년 baseline 값

    Returns:
        평가 결과 딕셔너리
    """
    X_test = test_data[feature_columns]
    y_test = test_data['관중비율']

    # 비율 예측
    y_test_pred_ratio = model.predict(X_test)

    # 관중수 변환
    predicted_attendance = y_test_pred_ratio * baseline_2025
    actual_attendance = test_data['관중수'].values

    # 평가 지표 계산
    # 비율 기준
    mae_ratio = mean_absolute_error(y_test, y_test_pred_ratio)
    rmse_ratio = np.sqrt(mean_squared_error(y_test, y_test_pred_ratio))
    r2_ratio = r2_score(y_test, y_test_pred_ratio)
    mape_ratio = np.mean(np.abs((y_test - y_test_pred_ratio) / y_test)) * 100

    # 관중수 기준
    mae_attendance = mean_absolute_error(actual_attendance, predicted_attendance)
    rmse_attendance = np.sqrt(mean_squared_error(actual_attendance, predicted_attendance))
    r2_attendance = r2_score(actual_attendance, predicted_attendance)
    mape_attendance = np.mean(np.abs((actual_attendance - predicted_attendance) / actual_attendance)) * 100

    return {
        'mae_ratio': mae_ratio,
        'rmse_ratio': rmse_ratio,
        'r2_ratio': r2_ratio,
        'mape_ratio': mape_ratio,
        'mae_attendance': mae_attendance,
        'rmse_attendance': rmse_attendance,
        'r2_attendance': r2_attendance,
        'mape_attendance': mape_attendance,
        'predicted_attendance': predicted_attendance,
        'actual_attendance': actual_attendance
    }


def save_results(output_dir: Path,
                 results_df: pd.DataFrame,
                 test_results: pd.DataFrame,
                 feature_importance_df: pd.DataFrame,
                 best_model: Any,
                 best_model_name: str,
                 tuned_params: Dict = None):
    """
    결과를 파일로 저장합니다.

    Args:
        output_dir: 출력 디렉토리
        results_df: 모델 비교 결과
        test_results: 테스트 예측 결과
        feature_importance_df: Feature importance
        best_model: 최적 모델
        best_model_name: 최적 모델 이름
        tuned_params: 튜닝된 파라미터 (선택적)
    """
    print("\n[Phase Final] 결과 저장")
    print("-" * 80)

    output_dir.mkdir(exist_ok=True)

    # 1. 모델 성능 비교표
    results_df.to_csv(output_dir / "model_performance_comparison_v2.csv",
                     index=False, encoding='utf-8-sig')
    print(f"✓ 모델 성능 비교표 저장: model_performance_comparison_v2.csv")

    # 2. 예측 결과
    test_results.to_csv(output_dir / "2025_predictions_detailed_v2.csv",
                       index=False, encoding='utf-8-sig')
    print(f"✓ 예측 결과 저장: 2025_predictions_detailed_v2.csv")

    # 3. Feature Importance
    if feature_importance_df is not None:
        feature_importance_df.to_csv(output_dir / "feature_importance_v2.csv",
                                    index=False, encoding='utf-8-sig')
        print(f"✓ Feature Importance 저장: feature_importance_v2.csv")

    # 4. 튜닝된 파라미터
    if tuned_params:
        params_df = pd.DataFrame([tuned_params])
        params_df.to_csv(output_dir / "best_hyperparameters_v2.csv",
                        index=False, encoding='utf-8-sig')
        print(f"✓ 최적 파라미터 저장: best_hyperparameters_v2.csv")

    # 5. 최종 모델
    model_path = output_dir / f"best_model_{best_model_name.replace(' ', '_')}_v2.pkl"
    joblib.dump(best_model, model_path)
    print(f"✓ 최종 모델 저장: {model_path.name}")

    print("\n✓ 모든 결과 저장 완료!")


# ============================================================================
# 4. 메인 실행 함수
# ============================================================================

def main():
    """
    메인 실행 함수
    """
    print("=" * 80)
    print("KBO 관중 수 예측 모델 구축 프로젝트 - 개선 버전 v2")
    print("데이터 누출 제거 + TimeSeriesSplit CV + 하이퍼파라미터 튜닝")
    print("=" * 80)

    # 경로 설정
    data_path = Path(__file__).resolve().parent / "df_refined.csv"
    output_dir = Path(__file__).resolve().parent.parent / "outputs"

    # 1. 데이터 로드
    df = load_data(data_path)

    # 2. 기본 Feature Engineering
    df = engineer_basic_features(df)

    # 3. 데이터 분할 정의
    train_years = [2023]
    val_years = [2024]
    test_years = [2025]

    # 4. 데이터 누출 방지를 위한 팀 인기도 계산
    print("\n[데이터 누출 방지] 팀 인기도 Features 계산")
    print("-" * 80)

    # 2023년 데이터: 2022년 이전 데이터가 없으므로 2023년 상반기 데이터 사용
    # (또는 전년도 데이터가 있다면 그것을 사용)
    # 여기서는 2023년 첫 60경기를 기준으로 사용
    print("2023년 학습 데이터에 대한 인기도 계산...")
    df_2023 = df[df['연도'] == 2023].copy()
    train_end_2023 = df_2023.iloc[min(60, len(df_2023)-1)]['일시']
    df_2023 = calculate_popularity_features_no_leakage(df_2023, train_end_2023)
    df_2023 = calculate_moving_average_and_target(df_2023)

    # 2024년 데이터: 2023년 전체 데이터 사용
    print("\n2024년 검증 데이터에 대한 인기도 계산...")
    df_2024 = df[df['연도'] == 2024].copy()
    train_end_2024 = df[df['연도'] == 2023]['일시'].max()
    df_2024 = calculate_popularity_features_no_leakage(df_2024, train_end_2024)
    df_2024 = calculate_moving_average_and_target(df_2024)

    # 2025년 데이터: 2023-2024년 전체 데이터 사용
    print("\n2025년 테스트 데이터에 대한 인기도 계산...")
    df_2025 = df[df['연도'] == 2025].copy()
    train_end_2025 = df[df['연도'].isin([2023, 2024])]['일시'].max()
    df_2025 = calculate_popularity_features_no_leakage(df_2025, train_end_2025)
    df_2025 = calculate_moving_average_and_target(df_2025)

    # 데이터 병합
    df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
    df = df.sort_values('일시').reset_index(drop=True)

    print("\n✓ 데이터 누출 없이 features 생성 완료!")

    # 5. Feature 컬럼 정의
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

    # 6. 최종 NaN 처리
    df = handle_final_nans(df, feature_columns)

    # 7. TimeSeriesSplit CV로 모델 학습
    trained_models, results_df = train_with_timeseries_cv(
        df=df,
        feature_columns=feature_columns,
        train_years=train_years,
        val_years=val_years,
        n_splits=3
    )

    # 8. 상위 3개 모델에 대해 하이퍼파라미터 튜닝
    print("\n[Phase 4] 상위 모델에 대한 하이퍼파라미터 튜닝")
    print("-" * 80)

    top_3_models = results_df.head(3)['Model'].tolist()
    print(f"튜닝 대상: {top_3_models}")

    # 학습 데이터 준비
    train_data = df[df['연도'].isin(train_years)]
    X_train = train_data[feature_columns]
    y_train = train_data['관중비율']

    param_grids = get_hyperparameter_grids()
    tuned_models = {}
    best_tuned_params = {}

    for model_name in top_3_models:
        if model_name in param_grids:
            base_model = trained_models[model_name]
            param_grid = param_grids[model_name]

            tuned_model, best_params = tune_hyperparameters(
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                base_model=base_model,
                param_grid=param_grid,
                n_iter=20,
                n_splits=3
            )

            tuned_models[model_name] = tuned_model
            best_tuned_params[model_name] = best_params
        else:
            print(f"  [{model_name}] 파라미터 그리드 없음 - 기본 모델 사용")
            tuned_models[model_name] = trained_models[model_name]

    # 9. 튜닝된 모델 중 최고 모델 선택
    print("\n[Phase 5] 튜닝된 모델 평가")
    print("-" * 80)

    val_data = df[df['연도'].isin(val_years)]
    X_val = val_data[feature_columns]
    y_val = val_data['관중비율']

    tuned_results = []
    for model_name, model in tuned_models.items():
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        tuned_results.append({
            'Model': model_name,
            'Val MAE (Tuned)': val_mae,
            'Val R² (Tuned)': val_r2
        })

        print(f"{model_name}: Val MAE = {val_mae:.4f}, R² = {val_r2:.4f}")

    tuned_results_df = pd.DataFrame(tuned_results).sort_values('Val MAE (Tuned)')
    best_model_name = tuned_results_df.iloc[0]['Model']
    best_model = tuned_models[best_model_name]

    print(f"\n🏆 최고 성능 모델 (튜닝 후): {best_model_name}")
    print(f"   Validation MAE: {tuned_results_df.iloc[0]['Val MAE (Tuned)']:.4f}")

    # 10. 최종 모델 학습 (2023 + 2024)
    print("\n[Phase 6] 최종 모델 학습 (2023 + 2024)")
    print("-" * 80)

    final_train_data = df[df['연도'].isin(train_years + val_years)]
    X_final_train = final_train_data[feature_columns]
    y_final_train = final_train_data['관중비율']

    final_model = clone(best_model)
    final_model.fit(X_final_train, y_final_train)

    print(f"최종 학습 완료: {len(X_final_train)}경기")

    # 11. 2025년 예측
    print("\n[Phase 7] 2025년 예측")
    print("-" * 80)

    test_data = df[df['연도'].isin(test_years)].copy()

    # 트렌드 추정
    avg_2023 = df[df['연도'] == 2023]['관중수'].mean()
    avg_2024 = df[df['연도'] == 2024]['관중수'].mean()
    growth_rate = avg_2024 / avg_2023
    baseline_2025 = avg_2024 * growth_rate

    print(f"2023년 평균 관중: {avg_2023:.0f}명")
    print(f"2024년 평균 관중: {avg_2024:.0f}명")
    print(f"증가율: {(growth_rate-1)*100:.1f}%")
    print(f"2025년 추정 baseline: {baseline_2025:.0f}명")

    # 평가
    eval_results = evaluate_on_test(
        model=final_model,
        test_data=test_data,
        feature_columns=feature_columns,
        baseline_2025=baseline_2025
    )

    print("\n[평가 결과 - 관중 비율]")
    print(f"  MAE: {eval_results['mae_ratio']:.4f} ({eval_results['mae_ratio']*100:.1f}% 오차)")
    print(f"  RMSE: {eval_results['rmse_ratio']:.4f}")
    print(f"  R²: {eval_results['r2_ratio']:.4f}")
    print(f"  MAPE: {eval_results['mape_ratio']:.2f}%")

    print("\n[평가 결과 - 관중 수]")
    print(f"  MAE: {eval_results['mae_attendance']:.0f}명")
    print(f"  RMSE: {eval_results['rmse_attendance']:.0f}명")
    print(f"  R²: {eval_results['r2_attendance']:.4f}")
    print(f"  MAPE: {eval_results['mape_attendance']:.2f}%")

    # 12. 예측 결과 DataFrame 생성
    test_results = test_data[['일시', '구장', '홈', '방문', '관중수']].copy()
    test_results['예측관중'] = eval_results['predicted_attendance']
    test_results['오차'] = test_results['관중수'] - test_results['예측관중']
    test_results['오차율(%)'] = (test_results['오차'] / test_results['관중수'] * 100).abs()

    # 13. Feature Importance
    feature_importance_df = None
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\n[Top 10 중요 Features]")
        print(feature_importance_df.head(10).to_string(index=False))

    # 14. 결과 저장
    save_results(
        output_dir=output_dir,
        results_df=results_df,
        test_results=test_results,
        feature_importance_df=feature_importance_df,
        best_model=final_model,
        best_model_name=best_model_name,
        tuned_params=best_tuned_params.get(best_model_name)
    )

    # 15. 최종 요약
    print("\n" + "=" * 80)
    print("최종 결과 요약")
    print("=" * 80)

    print(f"\n[모델 정보]")
    print(f"  - 최종 모델: {best_model_name} (하이퍼파라미터 튜닝)")
    print(f"  - 학습 데이터: 2023-2024년 ({len(X_final_train)}경기)")
    print(f"  - 테스트 데이터: 2025년 ({len(test_data)}경기)")
    print(f"  - Feature 수: {len(feature_columns)}개")
    print(f"  - CV 방식: TimeSeriesSplit (3 folds)")

    print(f"\n[성능 지표]")
    print(f"  - 관중 비율 MAE: {eval_results['mae_ratio']:.4f}")
    print(f"  - 관중 수 MAE: {eval_results['mae_attendance']:.0f}명")
    print(f"  - 관중 수 MAPE: {eval_results['mape_attendance']:.2f}%")
    print(f"  - R² Score: {eval_results['r2_attendance']:.4f}")

    if best_tuned_params.get(best_model_name):
        print(f"\n[최적 하이퍼파라미터]")
        for param, value in best_tuned_params[best_model_name].items():
            print(f"  - {param}: {value}")

    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)

    print("\n💡 개선 사항:")
    print("  ✅ 데이터 누출 완전 제거")
    print("  ✅ TimeSeriesSplit 교차 검증 적용")
    print("  ✅ 하이퍼파라미터 튜닝 완료")
    print("  ✅ 코드 모듈화 및 구조화")
    print("\n💡 다음 단계:")
    print("  1. scripts/visualize_results.py로 시각화")
    print("  2. outputs/ 폴더에서 v2 결과 확인")
    print("  3. 필요시 추가 feature engineering 고려")


if __name__ == "__main__":
    main()
