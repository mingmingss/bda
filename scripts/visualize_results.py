"""
KBO 관중 수 예측 - 결과 시각화
3조 - 모델 구축 담당: 임혜린, 윤태영

시각화 내용:
1. 모델 성능 비교 (Bar Chart)
2. 예측 vs 실제 산점도
3. 시계열 예측 결과
4. Feature Importance
5. 오차 분포 히스토그램
6. 월별/요일별/구장별 예측 정확도
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
sns.set_palette('husl')

print("=" * 80)
print("KBO 관중 수 예측 - 결과 시각화")
print("=" * 80)

# 출력 디렉토리
output_dir = Path(__file__).resolve().parent.parent / "outputs"
viz_dir = output_dir / "visualizations"
viz_dir.mkdir(exist_ok=True)

# ====== 데이터 로드 ======
print("\n[1] 데이터 로드 중...")

# 모델 성능 비교
performance_df = pd.read_csv(output_dir / "model_performance.csv", encoding='utf-8-sig')
print(f"✓ 모델 성능 비교 데이터 로드: {len(performance_df)}개 모델")

# 예측 결과
predictions_df = pd.read_csv(output_dir / "predictions_2025.csv", encoding='utf-8-sig')
predictions_df['일시'] = pd.to_datetime(predictions_df['일시'])
print(f"✓ 예측 결과 데이터 로드: {len(predictions_df)}개 경기")

# Feature Importance (있는 경우)
feature_importance_path = output_dir / "feature_importance.csv"
if feature_importance_path.exists():
    feature_importance_df = pd.read_csv(feature_importance_path, encoding='utf-8-sig')
    print(f"✓ Feature Importance 데이터 로드: {len(feature_importance_df)}개 Features")
else:
    feature_importance_df = None
    print("⚠️  Feature Importance 파일이 없습니다.")

# ====== 1. 모델 성능 비교 (Bar Chart) ======
print("\n[2] 시각화 생성 중...")
print("  1/7 모델 성능 비교 차트...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MAE 비교
ax1 = axes[0]
performance_sorted = performance_df.sort_values('Val MAE')
colors = ['#d62728' if i == 0 else '#1f77b4' for i in range(len(performance_sorted))]
ax1.barh(performance_sorted['Model'], performance_sorted['Val MAE'], color=colors)
ax1.set_xlabel('Validation MAE', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance - MAE (Lower is Better)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(performance_sorted['Val MAE']):
    ax1.text(v + 0.001, i, f'{v:.4f}', va='center')

# RMSE 비교
ax2 = axes[1]
performance_sorted_rmse = performance_df.sort_values('Val RMSE')
colors = ['#d62728' if i == 0 else '#1f77b4' for i in range(len(performance_sorted_rmse))]
ax2.barh(performance_sorted_rmse['Model'], performance_sorted_rmse['Val RMSE'], color=colors)
ax2.set_xlabel('Validation RMSE', fontsize=12, fontweight='bold')
ax2.set_title('Model Performance - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(performance_sorted_rmse['Val RMSE']):
    ax2.text(v + 0.001, i, f'{v:.4f}', va='center')

# R² 비교
ax3 = axes[2]
performance_sorted_r2 = performance_df.sort_values('Val R²', ascending=False)
colors = ['#2ca02c' if i == 0 else '#1f77b4' for i in range(len(performance_sorted_r2))]
ax3.barh(performance_sorted_r2['Model'], performance_sorted_r2['Val R²'], color=colors)
ax3.set_xlabel('Validation R²', fontsize=12, fontweight='bold')
ax3.set_title('Model Performance - R² (Higher is Better)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
for i, v in enumerate(performance_sorted_r2['Val R²']):
    ax3.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig(viz_dir / '01_model_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {viz_dir / '01_model_performance_comparison.png'}")
plt.close()

# ====== 2. 예측 vs 실제 산점도 ======
print("  2/7 예측 vs 실제 산점도...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 2.1 관중 수 기준
ax1 = axes[0]
ax1.scatter(predictions_df['관중수'], predictions_df['예측관중'],
            alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

# 완벽한 예측선
min_val = min(predictions_df['관중수'].min(), predictions_df['예측관중'].min())
max_val = max(predictions_df['관중수'].max(), predictions_df['예측관중'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax1.set_xlabel('Actual Attendance', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Attendance', fontsize=12, fontweight='bold')
ax1.set_title('Prediction vs Actual - Attendance (2025)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 통계 정보 추가
mae = np.mean(np.abs(predictions_df['관중수'] - predictions_df['예측관중']))
rmse = np.sqrt(np.mean((predictions_df['관중수'] - predictions_df['예측관중'])**2))
r2 = 1 - np.sum((predictions_df['관중수'] - predictions_df['예측관중'])**2) / \
    np.sum((predictions_df['관중수'] - predictions_df['관중수'].mean())**2)

textstr = f'MAE: {mae:.0f}\nRMSE: {rmse:.0f}\nR²: {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# 2.2 오차 분포
ax2 = axes[1]
errors = predictions_df['오차']
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
           label=f'Mean Error: {errors.mean():.0f}')

ax2.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(viz_dir / '02_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {viz_dir / '02_prediction_vs_actual.png'}")
plt.close()

# ====== 3. 시계열 예측 결과 ======
print("  3/7 시계열 예측 결과...")

fig, ax = plt.subplots(figsize=(20, 6))

# 날짜별 정렬
predictions_sorted = predictions_df.sort_values('일시')

# 실제 관중 수
ax.plot(predictions_sorted['일시'], predictions_sorted['관중수'],
       label='Actual Attendance', linewidth=2, alpha=0.7, color='blue')

# 예측 관중 수
ax.plot(predictions_sorted['일시'], predictions_sorted['예측관중'],
       label='Predicted Attendance', linewidth=2, alpha=0.7, color='red')

# 이동평균 (트렌드)
ax.plot(predictions_sorted['일시'],
       predictions_sorted['관중수'].rolling(window=30).mean(),
       label='30-Day Moving Average', linewidth=2, linestyle='--',
       alpha=0.5, color='green')

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Attendance', fontsize=12, fontweight='bold')
ax.set_title('Time Series Prediction - KBO Attendance 2025', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# x축 회전
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(viz_dir / '03_time_series_prediction.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {viz_dir / '03_time_series_prediction.png'}")
plt.close()

# ====== 4. Feature Importance ======
if feature_importance_df is not None:
    print("  4/7 Feature Importance...")

    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = feature_importance_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for i, v in enumerate(top_features['Importance']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / '04_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 저장: {viz_dir / '04_feature_importance.png'}")
    plt.close()
else:
    print("  4/7 Feature Importance... (스킵 - 데이터 없음)")

# ====== 5. 월별 예측 정확도 ======
print("  5/7 월별 예측 정확도...")

predictions_df['월'] = predictions_df['일시'].dt.month

monthly_stats = predictions_df.groupby('월').agg({
    '관중수': 'mean',
    '예측관중': 'mean',
    '오차': lambda x: np.mean(np.abs(x))
}).reset_index()

monthly_stats.columns = ['월', '평균_실제관중', '평균_예측관중', 'MAE']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5.1 월별 평균 관중
ax1 = axes[0]
x = monthly_stats['월']
width = 0.35

ax1.bar(x - width/2, monthly_stats['평균_실제관중'], width,
       label='Actual', alpha=0.8, color='blue', edgecolor='black')
ax1.bar(x + width/2, monthly_stats['평균_예측관중'], width,
       label='Predicted', alpha=0.8, color='red', edgecolor='black')

ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Attendance', fontsize=12, fontweight='bold')
ax1.set_title('Monthly Average Attendance - Actual vs Predicted', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 5.2 월별 MAE
ax2 = axes[1]
colors_mae = plt.cm.RdYlGn_r(monthly_stats['MAE'] / monthly_stats['MAE'].max())
ax2.bar(monthly_stats['월'], monthly_stats['MAE'], color=colors_mae, edgecolor='black')

ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('Monthly Prediction Error (MAE)', fontsize=14, fontweight='bold')
ax2.set_xticks(monthly_stats['월'])
ax2.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(monthly_stats['MAE']):
    ax2.text(monthly_stats['월'].iloc[i], v + 50, f'{v:.0f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / '05_monthly_accuracy.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {viz_dir / '05_monthly_accuracy.png'}")
plt.close()

# ====== 6. 요일별 예측 정확도 ======
print("  6/7 요일별 예측 정확도...")

weekday_map_kor = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
predictions_df['요일_숫자'] = predictions_df['일시'].dt.dayofweek
predictions_df['요일'] = predictions_df['요일_숫자'].map(weekday_map_kor)

weekday_stats = predictions_df.groupby('요일').agg({
    '관중수': 'mean',
    '예측관중': 'mean',
    '오차': lambda x: np.mean(np.abs(x))
}).reset_index()

# 요일 순서 정렬
weekday_order = ['월', '화', '수', '목', '금', '토', '일']
weekday_stats['요일'] = pd.Categorical(weekday_stats['요일'], categories=weekday_order, ordered=True)
weekday_stats = weekday_stats.sort_values('요일')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 6.1 요일별 평균 관중
ax1 = axes[0]
x = np.arange(len(weekday_stats))
width = 0.35

ax1.bar(x - width/2, weekday_stats['관중수'], width,
       label='Actual', alpha=0.8, color='blue', edgecolor='black')
ax1.bar(x + width/2, weekday_stats['예측관중'], width,
       label='Predicted', alpha=0.8, color='red', edgecolor='black')

ax1.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Attendance', fontsize=12, fontweight='bold')
ax1.set_title('Weekly Average Attendance - Actual vs Predicted', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(weekday_stats['요일'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 6.2 요일별 MAE
ax2 = axes[1]
colors_mae = plt.cm.RdYlGn_r(weekday_stats['오차'] / weekday_stats['오차'].max())
ax2.bar(weekday_stats['요일'], weekday_stats['오차'], color=colors_mae, edgecolor='black')

ax2.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('Weekly Prediction Error (MAE)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(weekday_stats['오차']):
    ax2.text(i, v + 50, f'{v:.0f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / '06_weekly_accuracy.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {viz_dir / '06_weekly_accuracy.png'}")
plt.close()

# ====== 7. 구장별 예측 정확도 ======
print("  7/7 구장별 예측 정확도...")

stadium_stats = predictions_df.groupby('구장').agg({
    '관중수': ['mean', 'count'],
    '예측관중': 'mean',
    '오차': lambda x: np.mean(np.abs(x))
}).reset_index()

stadium_stats.columns = ['구장', '평균_실제관중', '경기수', '평균_예측관중', 'MAE']
stadium_stats = stadium_stats.sort_values('MAE')

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# 7.1 구장별 평균 관중
ax1 = axes[0]
x = np.arange(len(stadium_stats))
width = 0.35

ax1.bar(x - width/2, stadium_stats['평균_실제관중'], width,
       label='Actual', alpha=0.8, color='blue', edgecolor='black')
ax1.bar(x + width/2, stadium_stats['평균_예측관중'], width,
       label='Predicted', alpha=0.8, color='red', edgecolor='black')

ax1.set_xlabel('Stadium', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Attendance', fontsize=12, fontweight='bold')
ax1.set_title('Stadium Average Attendance - Actual vs Predicted', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(stadium_stats['구장'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 경기 수 표시
for i, count in enumerate(stadium_stats['경기수']):
    ax1.text(i, stadium_stats['평균_실제관중'].iloc[i] + 500,
            f'n={count}', ha='center', va='bottom', fontsize=9)

# 7.2 구장별 MAE
ax2 = axes[1]
colors_mae = plt.cm.RdYlGn_r(stadium_stats['MAE'] / stadium_stats['MAE'].max())
bars = ax2.barh(stadium_stats['구장'], stadium_stats['MAE'], color=colors_mae, edgecolor='black')

ax2.set_xlabel('MAE', fontsize=12, fontweight='bold')
ax2.set_ylabel('Stadium', fontsize=12, fontweight='bold')
ax2.set_title('Stadium Prediction Error (MAE) - Sorted', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

for i, v in enumerate(stadium_stats['MAE']):
    ax2.text(v + 50, i, f'{v:.0f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / '07_stadium_accuracy.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {viz_dir / '07_stadium_accuracy.png'}")
plt.close()

# ====== 최종 요약 ======
print("\n" + "=" * 80)
print("시각화 완료!")
print("=" * 80)
print(f"\n저장 위치: {viz_dir}")
print("\n생성된 시각화 목록:")
print("  1. 01_model_performance_comparison.png - 모델 성능 비교")
print("  2. 02_prediction_vs_actual.png - 예측 vs 실제 산점도 & 오차 분포")
print("  3. 03_time_series_prediction.png - 시계열 예측 결과")
if feature_importance_df is not None:
    print("  4. 04_feature_importance.png - Feature Importance")
print("  5. 05_monthly_accuracy.png - 월별 예측 정확도")
print("  6. 06_weekly_accuracy.png - 요일별 예측 정확도")
print("  7. 07_stadium_accuracy.png - 구장별 예측 정확도")
print("\n" + "=" * 80)
