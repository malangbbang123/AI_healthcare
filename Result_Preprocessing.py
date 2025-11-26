import pandas as pd 
import re
from sklearn.preprocessing import LabelEncoder
import os 

data_path = "/workspace/source/Neurodigm/Data/"
print(f"*************************Loaded result data*************************")

Result = pd.read_parquet(os.path.join(data_path,"Data_Result_Neurodigm_tmp_Train.parquet"))
result_codebook = pd.read_excel(os.path.join(data_path, "Data_Catalog.xlsx"), engine="openpyxl")
## 코드명 - 검사코드 짝맞추기 
codes = result_codebook['검사코드'].tolist()
drop_list = [i for i in codes if result_codebook[result_codebook["검사코드"] == i]["분류"].tolist()[0] == "기본2"]
remove_list = ["HL288", "HL361", "HL290", "HL111", "HL426", "MG003", "HE104", "NN101", "HL065", "XC011", "XC012", "KK056", "RR004"]
Result = Result.drop(remove_list+drop_list, axis=1)
rename_result = {i:result_codebook[result_codebook['검사코드'] == i]["검사명"].tolist()[0] for i in codes}
rename_Result = Result.rename(columns=rename_result)
print(f"******************************Replaced column name******************************")

rename_Result['YEAR'] = rename_Result['ORDDD(검진일자)'].apply(lambda x: str(x)[:4])
rename_Result = rename_Result.sort_values(by=["S_PID", "ORDDD(검진일자)"], ascending=[True, True]).reset_index(drop=True)
year_list = ['2021', '2022', '2023']
rename_Result = rename_Result[rename_Result['YEAR'].isin(year_list)].reset_index(drop=True)
print(f"******************************Filtered 2021~2023******************************")

check_col = ['HBs Ag(정량)', 'HBs Ab(정량)', 'HBe Ag', 'anti-HBc', 'anti-HBe','anti-HCV(수치)', 'anti-HIV (AIDS)', 'Rubella Ab IgM', 'Rubella Ab IgG']
for i in check_col:
    print(rename_Result[i].value_counts())
## 제거 변수 
remove_col = ["시력", "청력", "이경", "색각", "본인기재", "보청기", "알레르기", "초음파", "CT", "MRI", "MRA", "baPWV(좌)", "baPWV(우)", "cardiac risk index", "CAVI(좌)", "CAVI(우)"]
drop_col = []
for col in rename_Result.columns:
    for rem_col in remove_col:
        if rem_col in col:
            drop_col.append(col)

rename_Result = rename_Result.drop(drop_col, axis=1)
print(f"******************************Removed column {drop_col}******************************")

print(f"******************************Starting Fillna & Remove Outlier******************************")

pos_neg_col = []
for col in rename_Result.columns:
    if rename_Result[col].dtype == "object":
        if "음성" in rename_Result[col].unique():
            pos_neg_col.append(col)



## 음/양성 변수 
def pos_neg_to_num(x):
    mapping = {"음성": 0, "경계": 1, "양성": 2, "양성1": 3, "양성2": 4}
    return mapping.get(x, np.nan)

    
drop_col = []
for col in pos_neg_col:

    if len(rename_Result[col].unique()) == 1:
        drop_col.append(col)
        continue
    if "음성" in rename_Result[col].unique():
        rename_Result[col] = rename_Result[col].fillna("음성")
        rename_Result[col] = rename_Result[col].map(pos_neg_to_num)
    else:
        rename_Result[col] = rename_Result[col].astype(float)
        mean_val = rename_Result[col].mean()
        rename_Result[col] = rename_Result[col].fillna(mean_val)

rename_Result = rename_Result.drop(drop_col, axis=1)


stay_dict = {'AA001': '신장',
 'AA002': '체중',
 'AA003': '비만도',
 'AA004': '체질량지수(BMI)',
 'AA005': '수축기혈압',
 'AA006': '이완기혈압',
 'AA012': '허리둘레',
 'AA143': '※혈압처방',
 'AA050': '맥박수',
 'KK056': '※팀파노메트리',
 'AA082': '과거병력 진단여부',
 'AA083': '과거병력 약물치료여부',
 'AA085': '생활습관-흡연',
 'AA086': '생활습관-음주',
 'AA140': '일주일음주량',
 'AA139': '하루폭음량(음주)',
 'AA087': '생활습관-신체활동',
 'AA088': '생활습관-근력운동',
 'RR004': 'KDSQ-C(인지기능장애평가도구)',
 'HH001': '혈색소량(Hb)',
 'HH002': '혈구용적치(HCT)',
 'HH003': '백혈구수',
 'HH004': '적혈구수',
 'HH005': '혈소판수',
 'HH006': '적혈구용적(MCV)',
 'HH007': '적혈구혈색소량(MCH)',
 'HH008': '적혈구혈혈색소농도(MCHC)',
 'HH009': 'RDW',
 'HH010': 'PDW',
 'HH011': 'MPV',
 'HH018': '분획호중구',
 'HH020': '림프구',
 'HH021': '단핵구',
 'HH022': '호산구',
 'HH023': '호염기구',
 'HH097': '혈색소',
 'HH013': 'ESR(적혈구침강속도)',
 'HE208': 'PL(인지질)',
 'TS159': '혈중 연(납)(소방,경찰)',
 'TS161': '혈중 카드뮴(소방,경찰)',
 'TS168': '혈중 카복시헤모글로빈(소방,경찰)',
 'HH016': '망상적혈구수(Reti. Count)',
 'NR008': 'Nicotine',
 'HH049': '니코틴 정량(혈청)',
 'HE210': '혈중알콜',
 'HE219': 'LAP',
 'HE062': 'Vitamin B12',
 'HH033': 'ABO형',
 'HH034': 'Rh형',
 'HE027': 'Ca(칼슘)',
 'HE038': 'P(인)',
 'HE034': 'K+(칼륨)',
 'HE041': 'Na+(나트륨)',
 'HE035': 'Cl (염소)',
 'HE207': 'Mg(마그네슘)',
 'NN002': '당(요검사)',
 'NN003': '요단백',
 'NN004': '요잠혈',
 'NN005': '요중백혈구(WBC)',
 'NN007': '유로빌리노겐',
 'NN008': '요빌리루빈',
 'NN009': '요중Nitrite',
 'NN010': 'pH (요검사)',
 'NN011': '요중케톤체량',
 'NN012': '요비중(SG)',
 'NN018': '상피세포',
 'NN072': '요침사 WBC',
 'NN073': '요침사 RBC',
 'HE004': '공복혈당',
 'HL036': 'HbA1C',
 'HL037': '인슐린',
 'HE203': 'Insulin Ab',
 'WE002': '즉시혈당',
 'HE006': '총콜레스테롤(T-Cholesterol)',
 'HE007': '중성지방(Triglyceride)',
 'HE008': 'HDL-Cholesterol',
 'HE010': 'LDL-Cholesterol',
 'HE009': 'LDL-효소',
 'HE029': 'B-Lipoprotein',
 'HE024': 'CPK',
 'HE212': 'Homocysteine',
 'XC024': 'baPWV(좌)',
 'XC025': 'baPWV(우)',
 'XC281': 'CAVI(좌)',
 'XC282': 'CAVI(우)',
 'HE202': 'cardiac risk index',
 'HE030': '아포지단백 A (Apolipoprotein A)',
 'HE031': '아포지단백 B (Apolipoprotein B)',
 'HE116': 'LP(a)',
 'HE201': 'B/A-I 비율',
 'HL032': 'CK-MB',
 'HE213': 'NT-PRO BNP(혈청)(심장기능)',
 'HE211': 'Troponin-I',
 'HE079': '마이오글로빈(Myoglobin)',
 'HE022': 'TIBC',
 'HE023': 'Fe',
 'HE026': 'UIBC',
 'HE205': '철포화율',
 'HE110': 'Ferritin',
 'BB024': '글로불린(Globulin)',
 'HE001': 'AST(SGOT)',
 'HE002': 'ALT(SGPT)',
 'HE003': 'γ-GTP',
 'HE011': '알부민',
 'HE012': '총단백(T.Protein)',
 'HE013': '총빌리루빈',
 'HE014': '직접빌리루빈',
 'HE015': 'Alk.Phosphatase(ALP)',
 'HE016': 'LDH',
 'HE020': 'A/G 비율',
 'HE064': '간접빌리루빈',
 'HE018': 'BUN/Cr 비율',
 'HE019': '*사구체여과율(GFR)',
 'HE032': '요소질소',
 'HE033': 'Creatinine',
 'HL038': 'T3',
 'HL041': 'Free T4',
 'HL042': 'TSH',
 'HL040': 'Free T3',
 'HL039': 'T4',
 'HE037': '25-OH Vitamin D',
 'HE025': 'Amylase',
 'HE214': 'Lipase',
 'TP004': '폐활량(FVC)',
 'TP005': '폐활량(FVC)% ',
 'TP006': '1초량(FEV1)',
 'TP007': '1초량(%)',
 'TP008': '1초율(FEV1/FVC%)',
 'TP012': '최대호기유수(PEF)',
 'TP013': '최대호기유속예측치%',
 'TP064': 'FVC<FIVC',
 'HL421': 'I/II ratio',
 'HL418': '펩시노겐I',
 'HL419': '펩시노겐II',
 'HE217': 'H.pylori(생화학)',
 'M_S021': 'H.pylori',
 'MG001': '헬리코박터(CLO Test)',
 'HL022': 'PSA ( 남자 )',
 'HL023': 'CEA',
 'HL024': 'CA 19-9',
 'HL025': 'CA 125II ( 여자 )',
 'HL027': 'CA 15-3 ( 여자 )',
 'HL030': 'AFP(E)-수치',
 'HL316': 'CYFRA',
 'HL387': 'AFP(R)-수치',
 'HH051': 'SCC',
 'HL410': 'ROMA(폐경전)',
 'HL411': 'ROMA(폐경후)',
 'HL422': 'NSE',
 'AA030': '근육량(Soft Lean Mass)',
 'AA031': '체지방량',
 'AA158': '골격근량(kg)',
 'AA159': 'S-체지방률(%)',
 'AA166': '골격근률(%)',
 'AA167': '체지방분석 적정체중',
 'AA168': '체지방분석 체중조절',
 'AA169': '체지방분석 지방조절',
 'AA170': '체지방분석 근육조절',
 'AA171': '체지방분석 내장지방레벨',
 'AA172': '체지방분석 복부비만율',
 'AA178': 'I-체지방률(%)',
 'AA181': 'S-골격근상경계',
 'AA182': 'S-골격근하경계',
 'AA179': '표준 체중',
 'HL058': '테스토스테론 (testosterone)',
 'HL062': '에스트라디올(estradiol) E2',
 'HL315': 'Estrogen(Total)',
 'HL057': '난포자극호르몬 (FSH)',
 'HL059': '황체형성호르몬(LH)',
 'HL046': '혈청 칼시토닌',
 'HE044': '오스테오칼신 (Osteocalcin)',
 'HL330': '알레르기-흰얼굴 호박벌독',
 'HL417': '마스토체크(유방암조기진단)',
 'HL415': '알츠온(알츠하이머위험도검사)',
 'HL427': 'KL-6',
 'NN102': 'M2-PK(대장암조기진단)',
 'CG153': '분변잠혈검사결과(음성/양성)',
 'HL412': 'M2BPGi 간섬유화 조기진단',
 'HE215': '활성산소량(d-ROMs)',
 'HE216': '총항산화력검사(BAP)',
 'HL324': 'NK 세포 활성',
 'HL373': '항-Ro(SSA)항체',
 'HL374': '항-La(SSB) 항체',
 'HT001': 'β2 -MG',
 'HE017': '요산(Uric Acid)',
 'HL047': 'hs-CRP',
 'HL048': 'RF(정량)',
 'HL007': 'B형간염결과',
 'CG590': 'HBs Ab(정성)-(채용전용)',
 'CG592': 'HBs Ag(정성)-(채용전용)',
 'HL005': 'HBs Ag(정량)',
 'HL006': 'HBs Ab(정량)',
 'HL150': 'HBe Ag',
 'HL011': 'anti-HBc',
 'HL009': 'anti-HBe',
 'HL014': 'anti-HCV(수치)',
 'HL015': 'HAV Ab IgM',
 'TS182': 'HAV IgM',
 'HL016': 'anti-HAV, Total',
 'HL018': 'RPR 정밀(매독)',
 'HL162': 'anti-HIV (AIDS)',
 'HL068': 'Rubella Ab IgM',
 'HL069': 'Rubella Ab IgG',
 'HL020': 'TLPA(매독)',
 'HH094': 'Procalcitonin(PCT) (패혈증)',
 'HL416': '코로나항체검사',
 'MG008': '분변원충검사',
 'CY101': '자궁검체상태',
 'CY102': '자궁경부선상피세포',
 'CY103': '자궁유형별진단',
 'CY104': '상피세포이상(편평상피세포이상)',
 'CY106': '상피세포이상(편평상피세포이상_위험구분)',
 'CY119': '중복자궁(유/무)',
 'HL175': '(HPV Genotyping)total',
 'HL177': '(HPV Genotyping)음성 타입개수',
 'HL178': '(HPV Genotyping)고위험군 타입',
 'HL179': '(HPV Genotyping)고위험군 결과값',
 'HL180': '(HPV Genotyping)고위험군 타입개수 ',
 'HL181': '(HPV Genotyping)잠재적위험군 타입',
 'HL182': '(HPV Genotyping)잠재적위험군 결과값',
 'HL183': '(HPV Genotyping)잠재적위험군 타입개수 ',
 'HL184': '(HPV Genotyping)저위험군 타입',
 'HL185': '(HPV Genotyping)저위험군 결과값',
 'HL186': '(HPV Genotyping)저위험군 타입개수',
 'HL187': '(HPV Genotyping)소견',
 'HL171': 'HPV 감염검사(검사실)',
 'HL304': 'HPV DNA Screening',
 'XC104': 'Calcium score(심장석회화CT)',
 'XC100': '전체지방(복부비만CT)',
 'XC101': '내장지방(복부비만CT)',
 'XC102': '피하지방(복부비만CT)',
 'XC103': '복부비만율(복부비만CT)'}

drop_col = []
for i in rename_Result.columns:
    if rename_Result[i].dtype == "object":
        if i not in stay_dict.values():
            if i not in ["S_PID", "RGSTNO(접수번호)", "SEX(성별)", "YEAR"]:
                drop_col.append(i)

rename_Result = rename_Result.drop(drop_col, axis=1)

rename_Result["※혈압처방"] = rename_Result["※혈압처방"].fillna("정상")
rename_Result["※혈압처방"] = rename_Result["※혈압처방"].apply(lambda x: "정상" if re.search("정상", x) else "비정상")

replace_col = {"SEX(성별)":{"F":1, "M":0}, 
              "과거병력":{"무":0, "유":1},
               "※혈압처방":{"정상":0, "비정상":1},
               "생활습관":{"양호":0, "개선필요":1},
               "ABO형":{"A":0, "B":1, "O":2, "AB":3},
               "Rh형":{"RH+":0, "RH-":1, "Weak D":2},
               "split_dash":["상피세포", "요침사 WBC", "요침사 RBC"],
               "FVC<FIVC":{"미해당":0, "해당":1},
               "B형간염결과":{"항체 없음":0, "항체 있음": 1, "B형간염보유자의심":2, "판정보류":3},
               "자궁경부선상피세포":{"무":0, "유":1},
               "상피세포이상(편평상피세포이상)":{"비정형 편평상피세포(ASC)":1, "저등급 편평 상피내 병변(L-SIL)":2, "고등급 편평 상피내 병변(H-SIL)":3,"침윤성 편평 세포암종(SCC)":4},
               "상피세포이상(편평상피세포이상_위험구분)":{"일반(ASC-US)":1, "고위험(ASC-H)":2},
               "자궁유형별진단":{'1':"양성", "상피세포 이상":"양성"},
               "중복자궁(유/무)":{"미해당":0, "해당":1, "0":0},
                "many_cat_col":["(HPV Genotyping)고위험군 타입", "(HPV Genotyping)잠재적위험군 타입","(HPV Genotyping)저위험군 타입"]
              }
               
le = LabelEncoder()

for col in rename_Result.columns:
    for rep_col in replace_col:
        if re.search(rep_col, col):
            if rep_col in ["자궁유형별진단", "상피세포이상(편평상피세포이상_위험구분)", "상피세포이상(편평상피세포이상)"]:
                fill_val = 0
            elif rep_col == "중복자궁(유/무)":
                fill_val = "미해당"
            fill_val = rename_Result[col].mode().values[0]
            rename_Result[col] = rename_Result[col].fillna(fill_val)
            rename_Result[col] = rename_Result[col].replace(replace_col[rep_col])
            print("Done !! ")
        elif col in replace_col[rep_col]:
            fill_val = rename_Result[col].mode().values[0]
            if rep_col == "split_dash":
                 rename_Result[col] = rename_Result[col].fillna(fill_val)
                 rename_Result[col] = rename_Result[col].apply(lambda x:int(x.split("-")[1]) if x != "many" else 40)
                 print("Done!! split_col")
            if rep_col == "many_cat_col":
                rename_Result[col] = rename_Result[col].fillna(fill_val)
                rename_Result[col] = le.fit_transform(rename_Result[col])
                print("Done!! many_cat_col")
        else:
            continue


for col in rename_Result.columns:
    if i in ["S_PID", "RGSTNO(접수번호)", "YEAR", "BRTHDD(생년월일)", "ORDDD(검진일자)"]:
        continue
    if rename_Result[col].dtype == "object":
        rename_Result[col] = pd.to_numeric(rename_Result[col], errors="coerce")
    
        
gender_diff = ["신장", "체중", "비만도", "체질량지수(BMI)", "허리둘레", "수축기혈압", "이완기혈압", "맥박수", "PSA ( 남자 )", "CA 125II ( 여자 )", "CA 15-3 ( 여자 )", 
 "ROMA(폐경전)", "ROMA(폐경후)", "S-체지방률(%)", "골격근량(kg)", "골격근률(%)", "I-체지방률(%)", "S-골격근상경계", "S-골격근하경계", "표준 체중", 
"테스토스테론 (testosterone)", "에스트라디올(estradiol) E2", "Estrogen(Total)"]

# drop_col = []
for col in rename_Result.columns:
    if col in ["S_PID", "RGSTNO(접수번호)", "YEAR", "BRTHDD(생년월일)", "ORDDD(검진일자)"]:
        continue
    else:
        if len(rename_Result[col].unique()) > 10:
            if i in gender_diff:
                man_mean = rename_Result[rename_Result["SEX(성별)"] == 0][col].mean()
                woman_mean = rename_Result[rename_Result["SEX(성별)"] == 1][col].mean()
                if pd.isna(man_mean):
                    man_mean = 0
                if pd.isna(woman_mean):
                    woman_mean = 0
                    
                rename_Result.loc[(rename_Result["SEX(성별)"] == 0) & (rename_Result[col].isna()), col] = man_mean
                rename_Result.loc[(rename_Result["SEX(성별)"] == 1) & (rename_Result[col].isna()), col] = woman_mean
            else:
                try:
                    mean_val = rename_Result[col].mean()
                    rename_Result[col] = rename_Result[col].fillna(mean_val)
                except Exception as e:
                    rename_Result[col] = pd.to_numeric(rename_Result[col], errors="coerce")
                    mean_val = rename_Result[col].mean()
                    rename_Result[col] = rename_Result[col].fillna(mean_val)

rename_Result = rename_Result.drop(drop_col, axis=1)
print(f"******************************Removed column {drop_col}******************************")

output_path = "/workspace/source/code_je/251104/clean_data"
rename_Result.to_csv(os.path.join(output_path, "Result.csv"), index=False)
print(f"*************************Finished Fillna & Remove Outlier*************************")
print(f"*************************Saved result data to the {output_path}*************************")
