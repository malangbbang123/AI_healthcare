import re
import pandas as pd 
import os 

root_dir = "/workspace/source/code_je/251104"
Result = pd.read_csv(os.path.join(root_dir, "clean_data", "Result.csv"))
Survey = pd.read_csv(os.path.join(root_dir, "clean_data", "Survey.csv"))
print("**************************Loaded Result, Survey Data**************************")
# 겹치는 변수 제거 
Survey.drop(['SEX(성별)', 'AGE(나이)', 'BRTHDD(생년월일)', 'RGSTNO(접수번호)'], axis=1, inplace=True)

for i in Result.columns:
    if Result[i].isnull().sum() > 0:
        Result[i] = pd.to_numeric(i, errors='coerce')
        Result[i].fillna(Result[i].mean(), inplace=True)

print(Result[['HBs Ag(정량)', 'HBs Ab(정량)', 'HBe Ag', 'anti-HBc', 'anti-HBe', 'anti-HCV(수치)', 'anti-HIV (AIDS)', 'Rubella Ab IgM', 'Rubella Ab IgG']])
Result['RGSTNO(접수번호)'] = Result['RGSTNO(접수번호)'].astype(object)
drop_col = []
for i in Result.columns:
    for j in ['시력', '청력', 'CT', "MRA", "MRI", '진단']:
        if re.search(j, i):
            drop_col.append(i)

Result.drop(drop_col, axis=1, inplace=True)
print("*****************************Merged Survey & Result *****************************")

merged = pd.merge(Survey, Result, on=['S_PID', 'ORDDD(검진일자)'], how='inner')

print(f"**************************Drop col is {drop_col}**************************")
output_path = "/workspace/source/code_je/251104/clean_data/Merged.csv"
merged.to_csv(output_path, index=False)
print(f"**************************Saved to the {output_path}**************************")