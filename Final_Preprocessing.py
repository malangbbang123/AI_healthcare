import pandas as pd
import re

df = pd.read_csv("/workspace/source/code_je/251104/clean_data/Merged.csv")
drop_col = ["가족력_3.고혈압_진단", "가족력_4.당뇨병_진단", "가족력_6.기타암_본인", "가족력_6.기타암_부모", "가족력_6.기타암_자매", "가족력_6.기타암_질환명", "가족력_6.기타암_형제",
            "여성력_2.복용_피임약", "흡연_2.궐련형전자담배_금연기간", "흡연_2.궐련형전자담배_흡연기간", "흡연_2.궐련형전자담배_흡연량", "흡연_2.궐련형전자담배_흡연력_평생", "흡연_3.액상형전자담배_흡연력_최근한달",
            "흡연_3.액상형전자담배_흡연력_평생", "생활습관-흡연" ,"일주일음주량", "하루폭음량(음주)"]

d_c = [i for i in drop_col if i in df.columns]
drop_df = df.drop(d_c, axis=1)

# 1 : 잔, 2: 병, 3: 캔, 4: cc

def soju_gram(x): return 7 if x == 1 else 50
def beer_gram(x): return 14 if x == 1 else 20
def liquer_gram(x): return 14 if x == 1 else 160
def makgeolli_gram(x): return 10 if x == 1 else 30
def wine_gram(x): return 12 if x == 1 else 60


disease_family = {}
for i in drop_df.columns:
    if "가족력" in i:
        disease_name = i.split("_")[-2]
        if disease_name in disease_family:
            disease_family[disease_name].append(i)
        else:
            disease_family[disease_name] = []
            disease_family[disease_name].append(i)

all_family_col = []
for i in disease_family:
    family_col = []
    if len(disease_family[i]) == 1:
        continue
    else: 
        for j in disease_family[i]:
            if "본인" in j:
                continue
            else:
                family_col.append(j)
                all_family_col.append(j)
        drop_df[f"가족력_{i}_통합"] = drop_df[family_col].sum(axis=1)
        # print(drop_df[f"가족력_{i}_통합"].value_counts())
        drop_df[f"가족력_{i}_통합"] = drop_df[f"가족력_{i}_통합"].apply(lambda x:1 if x != 0 else 0)
        drop_df.drop(family_col, axis=1, inplace=True)

workout_cols = [i for i in drop_df.columns if "강도" in i]

## 한적 없음: 0시간, 3일+1시간: 3*1시간=3시간, 5일+30분:5*0.5=2.5시간 

workout_cols = [i for i in drop_df.columns if "강도" in i]

drop_df['신체활동_고강도_통합'] = drop_df.apply(lambda row: row['신체활동_1.고강도_일주일'] * ((row['신체활동_1.고강도_하루_시간'] + (row['신체활동_1.고강도_하루_분']/60))), axis=1)
drop_df['신체활동_중강도_통합'] = drop_df.apply(lambda row: row['신체활동_2.중강도_일주일'] * ((row['신체활동_2.중강도_하루_시간'] + (row['신체활동_2.중강도_하루_분']/60))), axis=1)

drop_df.loc[drop_df['생활습관-신체활동'] == 0, "신체활동_고강도_통합"] = 0
drop_df.loc[drop_df['생활습관-신체활동'] == 0, "신체활동_중강도_통합"] = 0

drop_df.loc[drop_df["신체활동_고강도_통합"].isna(), "신체활동_고강도_통합"] = int(drop_df["신체활동_고강도_통합"].mean())
drop_df.loc[drop_df["신체활동_중강도_통합"].isna(), "신체활동_고강도_통합"] = int(drop_df["신체활동_중강도_통합"].mean())

drop_df.drop(workout_cols, axis=1, inplace=True)


drop_df.loc[(drop_df['흡연_1.일반담배_흡연력_평생'] == 1) & (drop_df['흡연_1.일반담배_흡연기간'] != 0), '흡연_1.일반담배_흡연기간'] = 0
drop_df.loc[(drop_df['흡연_1.일반담배_흡연력_평생'] == 1) & (drop_df['흡연_1.일반담배_흡연량'] != 0), '흡연_1.일반담배_흡연량'] = 0
drop_df.loc[(drop_df['흡연_1.일반담배_흡연력_평생'] == 1) & (drop_df['흡연_1.일반담배_금연기간'] != 0), '흡연_1.일반담배_금연기간'] = 0

drop_df.loc[drop_df['흡연_1.일반담배_흡연기간'].isna(), '흡연_1.일반담배_흡연기간'] = int(drop_df['흡연_1.일반담배_흡연기간'].mean())
drop_df.loc[drop_df['흡연_1.일반담배_흡연량'].isna(), '흡연_1.일반담배_흡연량'] = int(drop_df['흡연_1.일반담배_흡연량'].mean())
drop_df.loc[drop_df['흡연_1.일반담배_금연기간'].isna(), '흡연_1.일반담배_금연기간'] = int(drop_df['흡연_1.일반담배_금연기간'].mean())

average_col = []
maximum_col = []
for i in drop_df.columns:
    if "평균음주량" in i:
        average_col.append(i)
    elif "최대음주량" in i:
        maximum_col.append(i)


drop_df['음주_2.평균음주량_소주(단위)'] = drop_df['음주_2.평균음주량_소주(단위)'].apply(soju_gram)
drop_df['음주_2.평균음주량_맥주(단위)'] = drop_df['음주_2.평균음주량_맥주(단위)'].apply(beer_gram)
drop_df['음주_2.평균음주량_양주(단위)'] = drop_df['음주_2.평균음주량_양주(단위)'].apply(liquer_gram)
drop_df['음주_2.평균음주량_막걸리(단위)'] = drop_df['음주_2.평균음주량_막걸리(단위)'].apply(makgeolli_gram)
drop_df['음주_2.평균음주량_와인(단위)'] = drop_df['음주_2.평균음주량_와인(단위)'].apply(wine_gram)

drop_df['음주_평균음주량_통합'] = drop_df.apply(lambda row:(row['음주_2.평균음주량_소주(단위)'] * row['음주_2.평균음주량_소주(양)']) + 
            (row['음주_2.평균음주량_맥주(단위)'] * row['음주_2.평균음주량_맥주(양)']) + 
            (row['음주_2.평균음주량_양주(단위)'] * row['음주_2.평균음주량_양주(양)']) + 
            (row['음주_2.평균음주량_막걸리(단위)'] * row['음주_2.평균음주량_막걸리(양)']) + 
            (row['음주_2.평균음주량_와인(단위)'] * row['음주_2.평균음주량_와인(양)'])
            , axis=1)

drop_df['음주_3.최대음주량_소주(단위)'] = drop_df['음주_3.최대음주량_소주(단위)'].apply(soju_gram)
drop_df['음주_3.최대음주량_맥주(단위)'] = drop_df['음주_3.최대음주량_맥주(단위)'].apply(beer_gram)
drop_df['음주_3.최대음주량_양주(단위)'] = drop_df['음주_3.최대음주량_양주(단위)'].apply(liquer_gram)
drop_df['음주_3.최대음주량_막걸리(단위)'] = drop_df['음주_3.최대음주량_막걸리(단위)'].apply(makgeolli_gram)
drop_df['음주_3.최대음주량_와인(단위)'] = drop_df['음주_3.최대음주량_와인(단위)'].apply(wine_gram)

drop_df['음주_최대음주량_통합'] = drop_df.apply(lambda row:(row['음주_3.최대음주량_소주(단위)'] * row['음주_3.최대음주량_소주(양)']) + 
            (row['음주_3.최대음주량_맥주(단위)'] * row['음주_3.최대음주량_맥주(양)']) + 
            (row['음주_3.최대음주량_양주(단위)'] * row['음주_3.최대음주량_양주(양)']) + 
            (row['음주_3.최대음주량_막걸리(단위)'] * row['음주_3.최대음주량_막걸리(양)']) + 
            (row['음주_3.최대음주량_와인(단위)'] * row['음주_3.최대음주량_와인(양)'])
            , axis=1)

drop_df['음주_평균음주량_통합'] = drop_df['음주_평균음주량_통합'].fillna(int(drop_df['음주_평균음주량_통합'].mean()))
drop_df['음주_최대음주량_통합'] = drop_df['음주_최대음주량_통합'].fillna(int(drop_df['음주_최대음주량_통합'].mean()))

drop_df.drop(average_col+maximum_col, axis=1, inplace=True)

alcohol_cols = ["음주_1.음주상태_음주_신체부상",
"음주_1.음주상태_음주_빈도_일수",
"음주_1.음주상태_음주_빈도_일수_6잔(맥주2,000cc이상)",
"음주_1.음주횟수_일주일",
"음주_평균음주량_통합",
"음주_최대음주량_통합",
"음주_1.음주상태_금주권고",
"음주_2.음주처방_금주_절주_처방여부",
"음주_2.음주처방_현재음주습관유지",
"음주_2.음주처방_음주습관변경",
"음주_2.음주처방_회복까지금주",
"음주_2.음주처방_완전히금주",
"음주_2.음주처방_병원진료_금주보조제_처방",
"음주_2.음주처방_병원진료필요",
"음주_2.음주처방_평가점수"]

for col in alcohol_cols:
    drop_df.loc[drop_df['생활습관-음주'] == 0, col] = 0
    drop_df.loc[drop_df[col].isna(), col] = drop_df[col].mean()

drop_df.to_csv("/workspace/source/code_je/251104/clean_data/Prep_251104.csv", index=False)