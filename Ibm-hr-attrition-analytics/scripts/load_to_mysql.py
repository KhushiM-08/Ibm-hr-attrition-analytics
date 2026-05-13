"""
Load IBM HR dataset + ML risk scores into MySQL
Usage: python scripts/load_to_mysql.py --password yourpassword
"""
import pandas as pd, argparse, sys, os

def main(host, user, password, database):
    try:
        import mysql.connector
    except ImportError:
        print("Run: pip install mysql-connector-python --break-system-packages")
        sys.exit(1)

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    risk_csv = os.path.join(DATA_DIR, 'hr_attrition_with_risk.csv')
    if not os.path.exists(risk_csv):
        print("Run scripts/train_model.py first to generate risk scores.")
        sys.exit(1)

    df = pd.read_csv(risk_csv)
    print(f"Loaded {len(df)} records")

    conn   = mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM employee_risk_scores")
    cursor.execute("DELETE FROM employees")
    conn.commit()

    col_map = {
        'EmployeeNumber':'employee_number','Age':'age','Attrition':'attrition',
        'BusinessTravel':'business_travel','DailyRate':'daily_rate',
        'Department':'department','DistanceFromHome':'distance_from_home',
        'Education':'education','EducationField':'education_field',
        'EnvironmentSatisfaction':'environment_satisfaction','Gender':'gender',
        'HourlyRate':'hourly_rate','JobInvolvement':'job_involvement',
        'JobLevel':'job_level','JobRole':'job_role','JobSatisfaction':'job_satisfaction',
        'MaritalStatus':'marital_status','MonthlyIncome':'monthly_income',
        'MonthlyRate':'monthly_rate','NumCompaniesWorked':'num_companies_worked',
        'OverTime':'overtime','PercentSalaryHike':'percent_salary_hike',
        'PerformanceRating':'performance_rating',
        'RelationshipSatisfaction':'relationship_satisfaction',
        'StockOptionLevel':'stock_option_level','TotalWorkingYears':'total_working_years',
        'TrainingTimesLastYear':'training_times_last_year',
        'WorkLifeBalance':'work_life_balance','YearsAtCompany':'years_at_company',
        'YearsInCurrentRole':'years_in_current_role',
        'YearsSinceLastPromotion':'years_since_promotion',
        'YearsWithCurrManager':'years_with_curr_manager',
    }

    db_cols = list(col_map.values())
    sql_emp = f"INSERT INTO employees ({','.join(db_cols)}) VALUES ({','.join(['%s']*len(db_cols))})"

    n = 0
    for _, row in df.iterrows():
        vals = tuple(None if pd.isna(row.get(k)) else row.get(k) for k in col_map.keys())
        cursor.execute(sql_emp, vals)
        n += 1

    conn.commit()
    print(f"Inserted {n} employees")

    if 'AttritionRisk_Score' in df.columns:
        sql_risk = "INSERT INTO employee_risk_scores (employee_number,risk_score,risk_tier) VALUES (%s,%s,%s)"
        for _, row in df.iterrows():
            if not pd.isna(row.get('AttritionRisk_Score')):
                cursor.execute(sql_risk,(
                    int(row['EmployeeNumber']),
                    round(float(row['AttritionRisk_Score']),4),
                    str(row.get('AttritionRisk_Tier','Low'))
                ))
        conn.commit()
        print("Risk scores inserted")

    cursor.close(); conn.close()
    print("\nDone! Connect Power BI to: vw_attrition_dashboard")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host',     default='localhost')
    p.add_argument('--user',     default='root')
    p.add_argument('--password', required=True)
    p.add_argument('--database', default='ibm_hr_attrition')
    args = p.parse_args()
    main(args.host, args.user, args.password, args.database)
