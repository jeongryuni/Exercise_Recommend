# recommend.py
import pandas as pd
import numpy as np
import random
from fastapi import FastAPI
import joblib
import pymysql
from db import get_connection

app = FastAPI()

# 1) 모델 로드
model = joblib.load("exercise_recommender.pkl")
encoder = joblib.load("label_encoder.pkl")

# 2) 영상 매핑 로드
video_df = pd.read_csv("exercise_video_mapping.csv")

video_map = {
    row["운동명"]: {
        "url": row.get("영상URL"),
        "title": row.get("운동명"),
        "thumb": row.get("썸네일")
    }
    for _, row in video_df.iterrows()
}

def get_video(ex):
    return video_map.get(ex, {"url": None, "title": ex, "thumb": "/img/no_video.png"})


# 3) ROUTINE (일부 카테고리만 정의됨)
ROUTINE = {
    "하체근력": {
        "준비": ["전신 스트레칭"],
        "본": ["스쿼트", "런지"],
        "정리": ["전신 스트레칭"]
    },
    "상체근력": {
        "준비": ["전신 스트레칭"],
        "본": ["푸시업"],
        "정리": ["전신 스트레칭"]
    },
    "코어": {
        "준비": ["브릿지"],
        "본": ["플랭크", "크런치"],
        "정리": ["전신 스트레칭"]
    },
    "유산소-걷기": {
        "준비": ["전신 스트레칭"],
        "본": ["빠르게 걷기"],
        "정리": ["전신 스트레칭"]
    },
    "유산소-달리기": {
        "준비": ["전신 스트레칭"],
        "본": ["조깅", "인터벌 러닝"],
        "정리": ["전신 스트레칭"]
    },
    "전신유산소": {
        "준비": ["전신 스트레칭"],
        "본": ["버피", "점핑잭"],
        "정리": ["전신 스트레칭"]
    },
    "요가필라테스": {
        "준비": ["전신 스트레칭"],
        "본": ["요가 루틴", "전굴", "고양이-소"],
        "정리": ["비둘기 자세"]
    }
}


# 4) feature 구성 (총 11개)
def build_features(age, gender, height, weight, bmi):
    group = (age // 10) * 10
    F = 1 if gender == "F" else 0
    M = 1 if gender == "M" else 0

    return np.array([[
        group,     # 1. 연령대 그룹
        age,       # 2. 나이
        F,         # 3. 여자
        M,         # 4. 남자
        height,    # 5. 키
        weight,    # 6. 체중
        bmi,       # 7. BMI

        20,        # 8. 악력 좌 (기본값)
        22,        # 9. 악력 우 (기본값)
        10,        # 10. 윗몸말아올리기 (기본값)
        30         # 11. VO2max (기본값)
    ]])


# 5) 추천 API
@app.get("/recommend/{user_id}")
def recommend(user_id: int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT AGE, GENDER, HEIGHT, CURRENT_WEIGHT, BMI
        FROM USERS 
        WHERE USER_ID=%s
    """, (user_id,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if not user:
        return {"error": "User not found"}

    X = build_features(
        user["AGE"],
        user["GENDER"],
        user["HEIGHT"],
        user["CURRENT_WEIGHT"],
        user["BMI"]
    )

    pred = model.predict(X)[0]
    category = encoder.inverse_transform([pred])[0]

    # base_category 추출 (예: "상체근력(중급)" → "상체근력")
    base_category = category.split("(")[0]

    # ROUTINE에 없으면 기본 루틴 적용
    routine = ROUTINE.get(base_category)
    if routine is None:
        routine = {
            "준비": ["전신 스트레칭"],
            "본": ["빠르게 걷기"],
            "정리": ["전신 스트레칭"]
        }

    prep = routine["준비"][0]
    main = random.choice(routine["본"])
    cool = routine["정리"][0]

    return {
        "user_id": user_id,
        "predicted_category": category,
        "routine": {
            "준비운동": {"name": prep, **get_video(prep)},
            "본운동": {"name": main, **get_video(main)},
            "정리운동": {"name": cool, **get_video(cool)}
        }
    }
