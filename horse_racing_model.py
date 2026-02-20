"""
승마 경주 결과 예측 모델
- 목표: 경주마의 우승(1위) 여부 이진 분류
- 알고리즘: XGBoost + LightGBM 앙상블
- 피처: 말 정보, 기수, 경주 조건, 최근 성적 등
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. 샘플 데이터 생성
# ─────────────────────────────────────────────

def generate_sample_data(n_races: int = 500, horses_per_race: int = 8, seed: int = 42) -> pd.DataFrame:
    """
    실제 경주 데이터가 없을 때 사용하는 샘플 데이터 생성기.
    각 row = 경주에 출전한 한 마리의 기록.
    """
    np.random.seed(seed)
    rows = []

    for race_id in range(1, n_races + 1):
        track = np.random.choice(["서울", "부산", "제주"], p=[0.5, 0.3, 0.2])
        distance = np.random.choice([1000, 1200, 1400, 1600, 1800, 2000])
        surface = np.random.choice(["잔디", "모래"], p=[0.6, 0.4])
        weather = np.random.choice(["맑음", "흐림", "비"], p=[0.6, 0.3, 0.1])
        grade = np.random.choice(["G1", "G2", "G3", "일반"], p=[0.05, 0.1, 0.2, 0.65])
        purse = np.random.randint(5000, 100000)  # 만원 단위

        # 각 마리 능력치 (승자 결정에 영향)
        abilities = np.random.normal(50, 15, horses_per_race).clip(10, 100)
        winner_idx = np.argmax(abilities + np.random.normal(0, 10, horses_per_race))

        for horse_idx in range(horses_per_race):
            ability = abilities[horse_idx]
            rows.append({
                "race_id": race_id,
                "horse_no": horse_idx + 1,
                "track": track,
                "distance": distance,
                "surface": surface,
                "weather": weather,
                "grade": grade,
                "purse": purse,

                # 말 특성
                "horse_age": np.random.randint(3, 8),
                "horse_weight": np.random.normal(480, 20),
                "weight_change": np.random.normal(0, 3),   # 직전 대비 체중 변화(kg)

                # 기수
                "jockey_win_rate": np.random.beta(2, 8),   # 기수 승률 (0~1)
                "jockey_exp_years": np.random.randint(1, 20),

                # 최근 성적 (최근 5경기)
                "recent_avg_rank": np.random.uniform(1, horses_per_race),
                "recent_win_count": np.random.randint(0, 5),
                "recent_top3_count": np.random.randint(0, 5),
                "days_since_last_race": np.random.randint(7, 90),

                # 이 경주 조건 적합도
                "distance_preference": np.random.uniform(0, 1),  # 거리 선호도
                "surface_win_rate": np.random.beta(2, 8),         # 해당 주로 승률

                # 능력치 (실제에서는 없음, 참고용)
                "_ability": ability,

                # 정답 레이블
                "is_winner": int(horse_idx == winner_idx),
            })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────
# 2. 피처 엔지니어링
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 카테고리 인코딩
    le = LabelEncoder()
    for col in ["track", "surface", "weather", "grade"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # 파생 피처
    df["rank_improvement"] = df["recent_avg_rank"].apply(lambda x: max(0, 5 - x))  # 최근 성적 향상도
    df["fitness_score"] = (
        df["jockey_win_rate"] * 0.3 +
        df["surface_win_rate"] * 0.2 +
        df["distance_preference"] * 0.2 +
        (df["recent_win_count"] / 5) * 0.3
    )
    df["weight_penalty"] = df["weight_change"].abs()  # 체중 변화 절대값 (클수록 불리)
    df["experience_factor"] = df["jockey_exp_years"] / 20  # 기수 경험 정규화

    return df


FEATURE_COLS = [
    "horse_no", "distance", "purse", "horse_age", "horse_weight",
    "weight_change", "jockey_win_rate", "jockey_exp_years",
    "recent_avg_rank", "recent_win_count", "recent_top3_count",
    "days_since_last_race", "distance_preference", "surface_win_rate",
    "track_enc", "surface_enc", "weather_enc", "grade_enc",
    "rank_improvement", "fitness_score", "weight_penalty", "experience_factor",
]
TARGET_COL = "is_winner"


# ─────────────────────────────────────────────
# 3. 모델 학습
# ─────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# 4. 앙상블 예측
# ─────────────────────────────────────────────

def ensemble_predict_proba(models: list, X) -> np.ndarray:
    """여러 모델의 예측 확률 평균 앙상블"""
    probas = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probas


# ─────────────────────────────────────────────
# 5. 평가
# ─────────────────────────────────────────────

def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 6. 경주 우승마 추천
# ─────────────────────────────────────────────

def recommend_winner(df_race: pd.DataFrame, models: list) -> pd.DataFrame:
    """
    단일 경주 데이터(여러 말)를 받아 우승 확률 순위 반환.
    df_race: 한 경주에 출전하는 모든 말의 피처 데이터
    """
    df_feat = engineer_features(df_race)
    X = df_feat[FEATURE_COLS]
    proba = ensemble_predict_proba(models, X)

    result = df_race[["horse_no"]].copy()
    result["win_probability"] = proba
    result["rank"] = result["win_probability"].rank(ascending=False).astype(int)
    return result.sort_values("rank")


# ─────────────────────────────────────────────
# 7. 메인 파이프라인
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  승마 경주 결과 예측 모델")
    print("=" * 50)

    # 데이터 생성
    print("\n[1] 샘플 데이터 생성 중...")
    df = generate_sample_data(n_races=500, horses_per_race=8)
    df = engineer_features(df)
    print(f"    총 {len(df):,}개 행 생성 완료 (경주 {df['race_id'].nunique()}개)")

    # 학습/테스트 분리
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    학습: {len(X_train):,}개 / 테스트: {len(X_test):,}개")

    # 모델 학습
    print("\n[2] 모델 학습 중...")
    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)
    models = [xgb_model, lgb_model]
    print("    XGBoost, LightGBM 학습 완료")

    # 예측 및 평가
    print("\n[3] 모델 평가")
    y_proba = ensemble_predict_proba(models, X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate(y_test, y_pred, y_proba)

    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    F1 Score : {metrics['f1_score']:.4f}")
    print(f"    ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\n    Classification Report:")
    print(metrics["classification_report"])

    # 피처 중요도
    print("\n[4] 피처 중요도 (XGBoost Top 5)")
    fi = get_feature_importance(xgb_model, FEATURE_COLS)
    print(fi.head(5).to_string(index=False))

    # 모델 저장
    output_dir = "models/prediction"
    import os
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(xgb_model, f"{output_dir}/horse_xgb.pkl")
    joblib.dump(lgb_model, f"{output_dir}/horse_lgb.pkl")
    print(f"\n[5] 모델 저장 완료: {output_dir}/")

    return models, df, metrics, fi


if __name__ == "__main__":
    main()
