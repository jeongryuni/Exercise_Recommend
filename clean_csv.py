import pandas as pd
import os

def clean_csv(input_path, output_path):
    """
    원본 CSV에서 MVM_PRSCRPTN_CN 컬럼만 추출하여 새로운 CSV로 저장하는 함수
    """
    if not os.path.exists(input_path):
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return

    try:
        # CSV 로드
        df = pd.read_csv(input_path, encoding="utf-8")

        # 필요한 컬럼만 추출
        col = "MVM_PRSCRPTN_CN"
        if col not in df.columns:
            print(f"CSV에 '{col}' 컬럼이 없습니다.")
            return

        df_new = df[[col]].dropna()

        # 저장
        df_new.to_csv(output_path, index=False, encoding="utf-8")
        print(f"새 CSV 생성 완료: {output_path}")

    except Exception as e:
        print("CSV 처리 중 오류 발생:", e)


if __name__ == "__main__":
    # 원본 CSV 경로
    input_csv = "data/KS_NFA_FTNESS_MESURE_MVN_PRSCRPTN_GNRLZ_INFO_202503.csv"

    # 새 CSV 저장 경로
    output_csv = "data/exercise_routine_clean.csv"

    clean_csv(input_csv, output_csv)
