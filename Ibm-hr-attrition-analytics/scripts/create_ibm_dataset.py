"""
IBM HR Analytics Dataset — High-Fidelity Recreation
Exactly 1470 rows, 237 attrition (16.12%), strong feature correlations.
"""
import pandas as pd, numpy as np, os

np.random.seed(0)
N = 1470
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

depts = (['Research & Development']*961 + ['Sales']*446 + ['Human Resources']*63)
np.random.shuffle(depts); depts = np.array(depts)

role_map = {
    'Research & Development': ['Research Scientist','Laboratory Technician','Healthcare Representative','Manufacturing Director','Research Director','Manager'],
    'Sales': ['Sales Executive','Sales Representative','Manager'],
    'Human Resources': ['Human Resources','Manager'],
}
role_weights = {'Research Scientist':0.20,'Laboratory Technician':0.18,'Healthcare Representative':0.10,'Manufacturing Director':0.10,'Research Director':0.06,'Manager':0.06,'Sales Executive':0.22,'Sales Representative':0.07,'Human Resources':0.05}
roles = []
for d in depts:
    rs = role_map[d]; ww = np.array([role_weights.get(r,0.05) for r in rs]); ww/=ww.sum()
    roles.append(np.random.choice(rs, p=ww))
roles = np.array(roles)

age        = np.clip(np.round(np.random.normal(36.9,9.1,N)).astype(int),18,60)
gender     = np.random.choice(['Male','Female'],N,p=[0.60,0.40])
edu_w      = np.array([0.0517,0.1884,0.4816,0.2313,0.0469]); edu_w/=edu_w.sum()
education  = np.random.choice([1,2,3,4,5],N,p=edu_w)
ef_w       = np.array([0.4136,0.3156,0.0952,0.0898,0.0272,0.0586]); ef_w/=ef_w.sum()
edu_field  = np.random.choice(['Life Sciences','Medical','Marketing','Technical Degree','Human Resources','Other'],N,p=ef_w)
marital_w  = np.array([0.3197,0.4572,0.2231]); marital_w/=marital_w.sum()
marital    = np.random.choice(['Single','Married','Divorced'],N,p=marital_w)
travel_w   = np.array([0.7089,0.1891,0.1020]); travel_w/=travel_w.sum()
travel     = np.random.choice(['Travel_Rarely','Travel_Frequently','Non-Travel'],N,p=travel_w)

income_mu  = {'Laboratory Technician':3237,'Sales Representative':3427,'Human Resources':3800,'Research Scientist':4500,'Healthcare Representative':5000,'Sales Executive':6900,'Manufacturing Director':7000,'Manager':17000,'Research Director':15947}
monthly_income = np.array([int(np.clip(np.random.normal(income_mu.get(r,6000),income_mu.get(r,6000)*0.25),1009,19999)) for r in roles])
job_level  = np.where(monthly_income<3500,1,np.where(monthly_income<6000,2,np.where(monthly_income<10000,3,np.where(monthly_income<15000,4,5))))

daily_rate   = np.random.randint(102,1500,N)
hourly_rate  = np.random.randint(30,100,N)
monthly_rate = np.random.randint(2094,26999,N)
pct_hike     = np.random.randint(11,26,N)
performance  = np.random.choice([3,4],N,p=[0.8435,0.1565])
stock_opt    = np.random.choice([0,1,2,3],N,p=[0.4388,0.3592,0.1306,0.0714])
env_sat      = np.random.choice([1,2,3,4],N,p=[0.1014,0.1986,0.3517,0.3483])
job_inv      = np.random.choice([1,2,3,4],N,p=[0.0433,0.1651,0.5442,0.2474])
job_sat      = np.random.choice([1,2,3,4],N,p=[0.1122,0.1986,0.3333,0.3559])
rel_sat      = np.random.choice([1,2,3,4],N,p=[0.0952,0.1918,0.3741,0.3389])
wlb          = np.random.choice([1,2,3,4],N,p=[0.0544,0.1762,0.4469,0.3225])
overtime     = np.random.choice(['Yes','No'],N,p=[0.2864,0.7136])
years_co     = np.clip(np.round(np.random.exponential(6.5,N)).astype(int),0,40)
yrs_role     = np.minimum(np.random.randint(0,19,N),years_co)
yrs_promo    = np.minimum(np.random.randint(0,16,N),years_co)
yrs_mgr      = np.minimum(np.random.randint(0,18,N),years_co)
total_yrs    = np.clip(years_co+np.random.randint(0,20,N),0,40)
num_cos_w    = np.array([0.2517,0.1986,0.1769,0.1510,0.0789,0.0585,0.0367,0.0218,0.0190,0.0069]); num_cos_w/=num_cos_w.sum()
num_cos      = np.random.choice(range(0,10),N,p=num_cos_w)
train_w      = np.array([0.0571,0.1061,0.2884,0.2789,0.1714,0.0748,0.0233]); train_w/=train_w.sum()
training     = np.random.choice([0,1,2,3,4,5,6],N,p=train_w)
dist_home    = np.random.randint(1,30,N)

p = np.full(N,0.05)
p[overtime=='Yes']               += 0.22
p[job_sat<=2]                    += 0.12
p[monthly_income<3500]           += 0.11
p[wlb==1]                        += 0.09
p[travel=='Travel_Frequently']   += 0.08
p[marital=='Single']             += 0.07
p[years_co<=2]                   += 0.10
p[env_sat==1]                    += 0.07
p[job_inv==1]                    += 0.07
p[dist_home>20]                  += 0.05
p[num_cos>=5]                    += 0.05
p[stock_opt==0]                  += 0.04
p[job_level==1]                  += 0.05
p[age<30]                        += 0.04
p[pct_hike<=12]                  += 0.03
p = np.clip(p,0,0.92)
attrition = np.where(np.random.random(N)<p,'Yes','No')

yes_idx = np.where(attrition=='Yes')[0]; no_idx = np.where(attrition=='No')[0]; target=237
if len(yes_idx)>target:
    flip=yes_idx[np.argsort(p[yes_idx])[:len(yes_idx)-target]]; attrition[flip]='No'
elif len(yes_idx)<target:
    flip=no_idx[np.argsort(p[no_idx])[::-1][:target-len(yes_idx)]]; attrition[flip]='Yes'

df = pd.DataFrame({'Age':age,'Attrition':attrition,'BusinessTravel':travel,'DailyRate':daily_rate,'Department':depts,'DistanceFromHome':dist_home,'Education':education,'EducationField':edu_field,'EmployeeCount':1,'EmployeeNumber':np.arange(1,N+1),'EnvironmentSatisfaction':env_sat,'Gender':gender,'HourlyRate':hourly_rate,'JobInvolvement':job_inv,'JobLevel':job_level,'JobRole':roles,'JobSatisfaction':job_sat,'MaritalStatus':marital,'MonthlyIncome':monthly_income,'MonthlyRate':monthly_rate,'NumCompaniesWorked':num_cos,'Over18':'Y','OverTime':overtime,'PercentSalaryHike':pct_hike,'PerformanceRating':performance,'RelationshipSatisfaction':rel_sat,'StandardHours':80,'StockOptionLevel':stock_opt,'TotalWorkingYears':total_yrs,'TrainingTimesLastYear':training,'WorkLifeBalance':wlb,'YearsAtCompany':years_co,'YearsInCurrentRole':yrs_role,'YearsSinceLastPromotion':yrs_promo,'YearsWithCurrManager':yrs_mgr})

df.to_csv(os.path.join(OUT_DIR,'WA_Fn-UseC_-HR-Employee-Attrition.csv'),index=False)
with pd.ExcelWriter(os.path.join(OUT_DIR,'WA_Fn-UseC_-HR-Employee-Attrition.xlsx'),engine='openpyxl') as w:
    df.to_excel(w,sheet_name='HR-Employee-Attrition',index=False)

print(f"Rows:{len(df)} | Attrition Yes:{(df.Attrition=='Yes').sum()} ({(df.Attrition=='Yes').mean()*100:.2f}%)")
print(f"OT-Yes attr: {(df[df.OverTime=='Yes'].Attrition=='Yes').mean()*100:.1f}% | OT-No attr: {(df[df.OverTime=='No'].Attrition=='Yes').mean()*100:.1f}%")
print("Saved.")
