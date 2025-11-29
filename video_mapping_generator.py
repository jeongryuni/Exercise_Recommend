# video_mapping_generator.py
import pandas as pd
import re
import chardet

SRC = "서울올림픽기념국민체육진흥공단_국민체력100 운동처방 동영상주소 정보_20210727.csv"

# 인코딩 자동 감지
with open(SRC, 'rb') as f:
    raw = f.read()
    encoding = chardet.detect(raw)['encoding']
df = pd.read_csv(SRC, encoding=encoding)

print("원본 로우:", df.shape)

# 운동 > 카테고리 매핑
CATEGORY_MAP = {
    "스쿼트": "근력",
    "런지": "근력",
    "힙브릿지": "근력",
    "푸시업": "근력",
    "밴드 로우": "근력",

    "플랭크": "코어",
    "사이드 플랭크": "코어",
    "크런치": "코어",

    "전신 스트레칭": "스트레칭",
    "고관절 열기 스트레칭": "스트레칭",

    "요가 루틴": "요가",
    "비둘기 자세": "요가",
    "전굴": "요가",
    "고양이-소": "요가",

    "빠르게 걷기": "유산소",
    "인터벌 러닝": "유산소",
    "조깅": "유산소",
    "버피": "유산소",
    "점핑잭": "유산소"
}

ROUTINE_EXERCISES = list(CATEGORY_MAP.keys())

video_list = []

for ex in ROUTINE_EXERCISES:
    category = CATEGORY_MAP[ex]

    # CSV의 소분류에서 category 포함되는 영상 찾기
    matched = df[
        df["소분류"].str.contains(category, na=False) |
        df["제목"].str.contains(category, na=False)
        ]

    if len(matched) > 0:
        url = matched.iloc[0]["동영상주소"]
    else:
        url = None

    video_list.append({
        "운동명": ex,
        "카테고리": category,
        "영상URL": url
    })

video_df = pd.DataFrame(video_list)
video_df.to_csv("exercise_video_mapping.csv", index=False, encoding="utf-8-sig")

print("[완료] exercise_video_mapping.csv 생성됨")
print(video_df)
