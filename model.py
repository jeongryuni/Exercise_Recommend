# 라이브러리
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib


# CSV 로드 (체력측정 및 운동처방 종합 데이터) https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=b3924850-aa65-11ec-8ee4-95f65f846b27
files = sorted(glob.glob("data/KS_NFA_FTNESS_MESURE_MVN_PRSCRPTN_GNRLZ_INFO_*.csv"))
print("로드 파일:", files)

df_list = []
for file in files:
    print("로드 중:", file)
    temp = pd.read_csv(file, encoding='utf-8-sig')
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print("원본 shape:", df.shape)



# 수치형 컬럼 변환
numeric_cols = [
    "MESURE_AGE_CO",
    "MESURE_IEM_001_VALUE", "MESURE_IEM_002_VALUE", "MESURE_IEM_003_VALUE",
    "MESURE_IEM_004_VALUE", "MESURE_IEM_005_VALUE", "MESURE_IEM_006_VALUE",
    "MESURE_IEM_007_VALUE", "MESURE_IEM_008_VALUE", "MESURE_IEM_009_VALUE",
    "MESURE_IEM_010_VALUE", "MESURE_IEM_012_VALUE", "MESURE_IEM_013_VALUE",
    "MESURE_IEM_014_VALUE", "MESURE_IEM_015_VALUE", "MESURE_IEM_016_VALUE",
    "MESURE_IEM_017_VALUE", "MESURE_IEM_018_VALUE", "MESURE_IEM_019_VALUE",
    "MESURE_IEM_020_VALUE", "MESURE_IEM_021_VALUE", "MESURE_IEM_022_VALUE",
    "MESURE_IEM_023_VALUE", "MESURE_IEM_024_VALUE", "MESURE_IEM_025_VALUE",
    "MESURE_IEM_026_VALUE", "MESURE_IEM_027_VALUE", "MESURE_IEM_028_VALUE",
    "MESURE_IEM_029_VALUE", "MESURE_IEM_030_VALUE", "MESURE_IEM_031_VALUE",
    "MESURE_IEM_032_VALUE", "MESURE_IEM_033_VALUE", "MESURE_IEM_034_VALUE",
    "MESURE_IEM_035_VALUE", "MESURE_IEM_036_VALUE", "MESURE_IEM_037_VALUE",
    "MESURE_IEM_038_VALUE", "MESURE_IEM_039_VALUE", "MESURE_IEM_040_VALUE",
    "MESURE_IEM_041_VALUE", "MESURE_IEM_043_VALUE", "MESURE_IEM_044_VALUE",
    "MESURE_IEM_050_VALUE", "MESURE_IEM_051_VALUE", "MESURE_IEM_052_VALUE"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["MESURE_AGE_CO"].between(10, 100)]
df["group"] = (df["MESURE_AGE_CO"] // 10) * 10



# 3. 성별 인코딩
df["F"] = (df["SEXDSTN_FLAG_CD"] == "F").astype(int)
df["M"] = (df["SEXDSTN_FLAG_CD"] == "M").astype(int)



# 15개 운동 라벨 + 난이도 자동 분류
def map_label(text):
    if pd.isna(text):
        return "기타"

    t = text.lower()

    # 난이도
    if "초급" in t:
        level = "초급"
    elif "중급" in t:
        level = "중급"
    elif "고급" in t or "상급" in t:
        level = "고급"
    else:
        level = "일반"

    # 라벨 15종
    if "스트레칭" in t:
        return f"스트레칭({level})"

    if "걷기" in t or "워킹" in t:
        return f"유산소-걷기({level})"

    if "달리기" in t or "조깅" in t or "러닝" in t:
        return f"유산소-달리기({level})"

    if "자전거" in t or "사이클" in t:
        return f"유산소-자전거({level})"

    if any(k in t for k in ["버피", "점프", "전신"]):
        return f"전신유산소({level})"

    if any(k in t for k in ["스쿼트", "런지", "하체"]):
        return f"하체근력({level})"

    if any(k in t for k in ["푸시업", "팔굽", "가슴", "어깨"]):
        return f"상체근력({level})"

    if any(k in t for k in ["플랭크", "코어", "복근"]):
        return f"코어({level})"

    if "요가" in t or "필라테스" in t:
        return f"요가필라테스({level})"

    if "홈트" in t or "full body" in t:
        return f"전신근력({level})"

    return "기타"


df["label"] = df["MVM_PRSCRPTN_CN"].apply(map_label)
df = df[df["label"] != "기타"]

print("라벨 종류:", df["label"].unique())



# Feature 구성
feature_cols = [
    "group",
    "MESURE_AGE_CO",
    "F", "M",

    # 체력측정 주요지표
    "MESURE_IEM_001_VALUE",  # 키
    "MESURE_IEM_002_VALUE",  # 체중
    "MESURE_IEM_018_VALUE",  # BMI

    "MESURE_IEM_007_VALUE",  # 악력 좌
    "MESURE_IEM_008_VALUE",  # 악력 우
    "MESURE_IEM_009_VALUE",  # 윗몸말아올리기
    "MESURE_IEM_030_VALUE",  # VO₂max
]

df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df["label"]



# 라벨 인코딩
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)



# 데이터 스플릿
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42
)



# 모델 학습
model = RandomForestClassifier(
    n_estimators=350,
    max_depth=22,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Macro F1:", f1_score(y_test, pred, average="macro"))
print("Micro F1:", f1_score(y_test, pred, average="micro"))



# 모델 저장
joblib.dump(model, "exercise_recommender.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("모델 저장 완료")


# 정확도 / F1-score 그래프
import matplotlib.pyplot as plt

# 정확도 & F1-score 저장
acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
micro_f1 = f1_score(y_test, pred, average="micro")

plt.figure(figsize=(7,5))
plt.bar(["Accuracy", "Macro F1", "Micro F1"], [acc, macro_f1, micro_f1])
plt.ylim(0, 1)
plt.title("Model Accuracy & F1 Score")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("metric_score.png")
plt.show()

# 모델 예측 혼동 행렬
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("confusion_matrix.png")
plt.show()


# 변수 중요도 그래프
import numpy as np

importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,6))
indices = np.argsort(importances)[::-1]
plt.bar(np.array(feature_names)[indices], importances[indices])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()
