# recommend.py

import warnings
import random
import pandas as pd
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from fastapi import FastAPI
import joblib
import traceback
from db import get_connection
import os

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = FastAPI()

# ---------------------------------------------------------
# 1. 모델 로드
# ---------------------------------------------------------
try:
    model = joblib.load("exercise_recommender.pkl")
    encoder = joblib.load("label_encoder.pkl")
    print("모델 로드 완료")
except Exception as e:
    print("모델 로드 실패:", e)
    traceback.print_exc()
    model = None
    encoder = None


# ---------------------------------------------------------
# 2. 본운동 텍스트 기반 카테고리 자동 분류
# ---------------------------------------------------------
def classify_category(text):
    if not text:
        return "기타"

    t = text.lower()

    # 하체
    if any(k in t for k in ["스쿼트", "런지", "무릎", "허벅", "엉덩", "넙다리", "점프", "앉았다 일어서기", "다리"]):
        return "하체근력"

    # 상체
    if any(k in t for k in ["푸시업", "팔굽", "전완", "어깨", "가슴", "팔", "로우"]):
        return "상체근력"

    # 코어
    if any(k in t for k in ["플랭크", "버티기", "버드독", "크런치", "브릿지", "사이드", "힙힌지"]):
        return "코어"

    # 걷기
    if any(k in t for k in ["걷기", "파워워킹", "보행", "트레드밀"]):
        return "걷기"

    # 달리기
    if any(k in t for k in ["달리기", "조깅", "러닝", "왕복달리기", "인터벌"]):
        return "달리기"

    # 자전거
    if "자전거" in t:
        return "자전거"

    # 유연성
    if any(k in t for k in ["스트레칭", "유연", "필라테스", "요가", "아기자세"]):
        return "유연성"

    return "기타"


# ---------------------------------------------------------
# 3. CSV 파싱 → 카테고리별 운동 리스트 자동 생성
# ---------------------------------------------------------
CSV_PATH = "exercise_routine_clean.csv"

def extract_routines():
    if not os.path.exists(CSV_PATH):
        print("CSV 없음:", CSV_PATH)
        return {}

    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    col = "MVM_PRSCRPTN_CN"

    category_routines = {}

    for row in df[col].dropna():
        parts = [p.strip() for p in row.split("/")]

        prep = []
        main = []
        cool = []

        # 각 파트에서 운동 추출
        for p in parts:
            if p.startswith("준비운동"):
                items = p.split(":", 1)[1]
                prep = [x.strip() for x in items.split(",") if x.strip()]
            elif p.startswith("본운동"):
                items = p.split(":", 1)[1]
                main = [x.strip() for x in items.split(",") if x.strip()]
            elif p.startswith("정리운동"):
                items = p.split(":", 1)[1]
                cool = [x.strip() for x in items.split(",") if x.strip()]

        # 본운동 텍스트로 카테고리 분류
        if main:
            full_text = ",".join(main)
            category = classify_category(full_text)

            if category not in category_routines:
                category_routines[category] = {
                    "준비운동": [],
                    "본운동": [],
                    "정리운동": []
                }

            category_routines[category]["준비운동"].extend(prep)
            category_routines[category]["본운동"].extend(main)
            category_routines[category]["정리운동"].extend(cool)

    print("CSV 파싱 완료")
    return category_routines


CATEGORY_ROUTINES = extract_routines()
print("카테고리별 루틴 자동 구성 완료")


# ---------------------------------------------------------
# 4. 추천 API
# ---------------------------------------------------------
@app.get("/recommend/{user_id}")
def recommend(user_id: int):

    try:
        # DB 조회
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        sql = """
            SELECT AGE, GENDER, HEIGHT, CURRENT_WEIGHT, BMI
            FROM users
            WHERE USER_ID = %s;
        """
        cursor.execute(sql, (user_id,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user:
            return {"error": "user not found"}

        # 사용자 정보
        age = user["AGE"]
        gender = user["GENDER"]
        height = user["HEIGHT"]
        weight = user["CURRENT_WEIGHT"]
        bmi = user["BMI"]

        # 예측 입력 벡터
        group = (age // 10) * 10
        F = 1 if gender and gender.upper() == "F" else 0
        M = 1 if gender and gender.upper() == "M" else 0

        x = np.array([[group, age, F, M, height, weight, bmi]])

        # 모델 예측
        pred = model.predict(x)[0]
        predicted_category = encoder.inverse_transform([pred])[0]

        print("예측 카테고리:", predicted_category)

        routines = CATEGORY_ROUTINES.get(predicted_category)

        if not routines:
            return {
                "user_id": user_id,
                "predicted_category": predicted_category,
                "routine": None
            }

        # 준비운동 1개 선택
        prep_list = routines["준비운동"]
        prep = random.choice(list(set(prep_list))) if prep_list else None

        # 본운동 2개 선택
        main_list = list(set(routines["본운동"]))
        random.shuffle(main_list)
        main_selected = main_list[:2] if len(main_list) >= 2 else main_list

        # 정리운동 1개 선택
        cool_list = routines["정리운동"]
        cool = random.choice(list(set(cool_list))) if cool_list else None

        return {
            "user_id": user_id,
            "predicted_category": predicted_category,
            "routine": {
                "준비운동": prep,
                "본운동": main_selected,
                "정리운동": cool
            }
        }

    except Exception as e:
        print("오류:", e)
        traceback.print_exc()
        return {"error": str(e)}
