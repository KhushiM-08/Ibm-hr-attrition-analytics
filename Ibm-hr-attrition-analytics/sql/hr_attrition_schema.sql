-- ================================================================
--  IBM HR Analytics — SQL Schema & Analytical Queries
--  Database  : MySQL 8.0+
--  Dataset   : WA_Fn-UseC_-HR-Employee-Attrition (IBM / Kaggle)
--  Run       : mysql -u root -p < sql/hr_attrition_schema.sql
-- ================================================================

CREATE DATABASE IF NOT EXISTS ibm_hr_attrition;
USE ibm_hr_attrition;

-- ── Drop existing ──────────────────────────────────────────
DROP TABLE IF EXISTS employee_risk_scores;
DROP TABLE IF EXISTS employees;

-- ── TABLE: employees ──────────────────────────────────────
CREATE TABLE employees (
    employee_number         INT             PRIMARY KEY,
    age                     TINYINT         NOT NULL CHECK (age BETWEEN 18 AND 65),
    attrition               ENUM('Yes','No') NOT NULL,
    business_travel         ENUM('Non-Travel','Travel_Rarely','Travel_Frequently'),
    daily_rate              INT,
    department              VARCHAR(60)     NOT NULL,
    distance_from_home      TINYINT,
    education               TINYINT         CHECK (education BETWEEN 1 AND 5),
    education_field         VARCHAR(50),
    environment_satisfaction TINYINT        CHECK (environment_satisfaction BETWEEN 1 AND 4),
    gender                  ENUM('Male','Female'),
    hourly_rate             SMALLINT,
    job_involvement         TINYINT         CHECK (job_involvement BETWEEN 1 AND 4),
    job_level               TINYINT         CHECK (job_level BETWEEN 1 AND 5),
    job_role                VARCHAR(60),
    job_satisfaction        TINYINT         CHECK (job_satisfaction BETWEEN 1 AND 4),
    marital_status          ENUM('Single','Married','Divorced'),
    monthly_income          INT             NOT NULL,
    monthly_rate            INT,
    num_companies_worked    TINYINT,
    overtime                ENUM('Yes','No') NOT NULL DEFAULT 'No',
    percent_salary_hike     TINYINT,
    performance_rating      TINYINT         CHECK (performance_rating IN (3,4)),
    relationship_satisfaction TINYINT       CHECK (relationship_satisfaction BETWEEN 1 AND 4),
    stock_option_level      TINYINT         CHECK (stock_option_level BETWEEN 0 AND 3),
    total_working_years     TINYINT,
    training_times_last_year TINYINT,
    work_life_balance       TINYINT         CHECK (work_life_balance BETWEEN 1 AND 4),
    years_at_company        TINYINT,
    years_in_current_role   TINYINT,
    years_since_promotion   TINYINT,
    years_with_curr_manager TINYINT,
    created_at              TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_dept       (department),
    INDEX idx_attrition  (attrition),
    INDEX idx_jobrole    (job_role),
    INDEX idx_income     (monthly_income),
    INDEX idx_overtime   (overtime)
);

-- ── TABLE: employee_risk_scores (ML output) ────────────────
CREATE TABLE employee_risk_scores (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    employee_number INT             NOT NULL UNIQUE,
    risk_score      DECIMAL(6,4)    NOT NULL,
    risk_tier       ENUM('Low','Medium','High') NOT NULL,
    model_name      VARCHAR(40)     DEFAULT 'RandomForest_v1',
    scored_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_number) REFERENCES employees(employee_number) ON DELETE CASCADE
);

-- ================================================================
--  STORED PROCEDURES
-- ================================================================

-- Proc 1: Department attrition summary
DELIMITER $$
DROP PROCEDURE IF EXISTS sp_dept_summary$$
CREATE PROCEDURE sp_dept_summary()
BEGIN
    SELECT
        department,
        COUNT(*)                                              AS total_employees,
        SUM(attrition = 'Yes')                               AS attrited,
        SUM(attrition = 'No')                                AS retained,
        ROUND(AVG(attrition = 'Yes') * 100, 2)              AS attrition_rate_pct,
        ROUND(AVG(monthly_income), 0)                        AS avg_salary,
        ROUND(AVG(job_satisfaction), 2)                      AS avg_job_satisfaction,
        ROUND(AVG(work_life_balance), 2)                     AS avg_work_life_balance,
        ROUND(AVG(years_at_company), 2)                      AS avg_tenure_years
    FROM employees
    GROUP BY department
    ORDER BY attrition_rate_pct DESC;
END$$
DELIMITER ;

-- Proc 2: High risk employees (with ML score)
DELIMITER $$
DROP PROCEDURE IF EXISTS sp_high_risk_employees$$
CREATE PROCEDURE sp_high_risk_employees(IN dept_filter VARCHAR(60))
BEGIN
    SELECT
        e.employee_number,
        e.age,
        e.department,
        e.job_role,
        e.gender,
        e.marital_status,
        e.monthly_income,
        e.overtime,
        e.job_satisfaction,
        e.work_life_balance,
        e.years_at_company,
        e.business_travel,
        r.risk_score,
        r.risk_tier
    FROM employees e
    INNER JOIN employee_risk_scores r
        ON e.employee_number = r.employee_number
    WHERE r.risk_tier = 'High'
      AND (dept_filter = '' OR dept_filter IS NULL OR e.department = dept_filter)
    ORDER BY r.risk_score DESC;
END$$
DELIMITER ;

-- Proc 3: Salary benchmark by role and attrition
DELIMITER $$
DROP PROCEDURE IF EXISTS sp_salary_benchmark$$
CREATE PROCEDURE sp_salary_benchmark()
BEGIN
    SELECT
        job_role,
        department,
        ROUND(AVG(CASE WHEN attrition='Yes' THEN monthly_income END), 0) AS avg_salary_left,
        ROUND(AVG(CASE WHEN attrition='No'  THEN monthly_income END), 0) AS avg_salary_stayed,
        ROUND(AVG(CASE WHEN attrition='No'  THEN monthly_income END) -
              AVG(CASE WHEN attrition='Yes' THEN monthly_income END), 0) AS salary_gap,
        COUNT(*) AS total
    FROM employees
    GROUP BY job_role, department
    HAVING total >= 10
    ORDER BY salary_gap DESC;
END$$
DELIMITER ;

-- ================================================================
--  ANALYTICAL QUERIES (IBM HR Dataset)
-- ================================================================

-- Q1: Overall attrition summary
SELECT
    COUNT(*)                                        AS total_employees,
    SUM(attrition = 'Yes')                         AS employees_left,
    SUM(attrition = 'No')                          AS employees_stayed,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_rate_pct,
    ROUND(AVG(age), 1)                             AS avg_age,
    ROUND(AVG(monthly_income), 0)                  AS avg_monthly_income,
    ROUND(AVG(years_at_company), 2)                AS avg_tenure
FROM employees;

-- Q2: Attrition by department — full breakdown
SELECT
    department,
    COUNT(*)                                        AS total,
    SUM(attrition = 'Yes')                         AS left_count,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct,
    ROUND(AVG(monthly_income), 0)                  AS avg_income,
    ROUND(AVG(job_satisfaction), 2)                AS avg_job_sat,
    ROUND(AVG(work_life_balance), 2)               AS avg_wlb,
    SUM(overtime = 'Yes')                          AS overtime_workers
FROM employees
GROUP BY department
ORDER BY attrition_pct DESC;

-- Q3: Overtime impact — the #1 driver
SELECT
    overtime,
    COUNT(*)                                        AS total,
    SUM(attrition = 'Yes')                         AS left_count,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct,
    ROUND(AVG(monthly_income), 0)                  AS avg_income,
    ROUND(AVG(job_satisfaction), 2)                AS avg_job_sat
FROM employees
GROUP BY overtime;

-- Q4: Attrition by Job Role (sorted by risk)
SELECT
    job_role,
    department,
    COUNT(*)                                        AS total,
    SUM(attrition = 'Yes')                         AS left_count,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct,
    ROUND(AVG(monthly_income), 0)                  AS avg_income,
    ROUND(AVG(years_at_company), 1)                AS avg_tenure
FROM employees
GROUP BY job_role, department
HAVING total >= 10
ORDER BY attrition_pct DESC;

-- Q5: Tenure cohort analysis
SELECT
    CASE
        WHEN years_at_company BETWEEN 0 AND 2   THEN '0-2 years  (Critical)'
        WHEN years_at_company BETWEEN 3 AND 5   THEN '3-5 years'
        WHEN years_at_company BETWEEN 6 AND 10  THEN '6-10 years'
        WHEN years_at_company BETWEEN 11 AND 20 THEN '11-20 years'
        ELSE '20+ years   (Veterans)'
    END                                             AS tenure_band,
    COUNT(*)                                        AS employees,
    SUM(attrition = 'Yes')                         AS attrited,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct,
    ROUND(AVG(monthly_income), 0)                  AS avg_income
FROM employees
GROUP BY tenure_band
ORDER BY MIN(years_at_company);

-- Q6: Income quartile analysis using NTILE window function
SELECT
    income_quartile,
    CONCAT('Q', income_quartile)                   AS quartile_label,
    MIN(monthly_income)                            AS min_income,
    MAX(monthly_income)                            AS max_income,
    COUNT(*)                                        AS employees,
    SUM(attrition = 'Yes')                         AS attrited,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct
FROM (
    SELECT *, NTILE(4) OVER (ORDER BY monthly_income) AS income_quartile
    FROM employees
) t
GROUP BY income_quartile
ORDER BY income_quartile;

-- Q7: Satisfaction composite score vs attrition
SELECT
    employee_number,
    department,
    job_role,
    monthly_income,
    overtime,
    attrition,
    job_satisfaction,
    environment_satisfaction,
    work_life_balance,
    relationship_satisfaction,
    job_involvement,
    ROUND(
        (job_satisfaction + environment_satisfaction +
         work_life_balance + relationship_satisfaction + job_involvement) / 5.0
    , 2)                                           AS composite_satisfaction_score
FROM employees
ORDER BY composite_satisfaction_score ASC
LIMIT 50;

-- Q8: Running attrition total by age (window function)
SELECT
    age,
    COUNT(*)                                                    AS employees,
    SUM(attrition = 'Yes')                                     AS left_this_age,
    SUM(SUM(attrition = 'Yes')) OVER (ORDER BY age)           AS running_attrition_total,
    ROUND(AVG(attrition = 'Yes') * 100, 2)                    AS attrition_pct_this_age,
    ROUND(AVG(monthly_income), 0)                              AS avg_income
FROM employees
GROUP BY age
ORDER BY age;

-- Q9: Marital status breakdown
SELECT
    marital_status,
    COUNT(*)                                        AS total,
    SUM(attrition = 'Yes')                         AS attrited,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct,
    ROUND(AVG(monthly_income), 0)                  AS avg_income,
    SUM(overtime = 'Yes')                          AS overtime_workers,
    ROUND(AVG(job_satisfaction), 2)                AS avg_job_sat
FROM employees
GROUP BY marital_status
ORDER BY attrition_pct DESC;

-- Q10: Business travel vs attrition
SELECT
    business_travel,
    COUNT(*)                                        AS total,
    SUM(attrition = 'Yes')                         AS attrited,
    ROUND(AVG(attrition = 'Yes') * 100, 2)        AS attrition_pct,
    ROUND(AVG(monthly_income), 0)                  AS avg_income
FROM employees
GROUP BY business_travel
ORDER BY attrition_pct DESC;

-- Q11: Risk score summary (after ML pipeline)
SELECT
    risk_tier,
    COUNT(*)                                                   AS employee_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2)        AS pct_of_total,
    ROUND(AVG(risk_score) * 100, 2)                           AS avg_risk_pct,
    ROUND(MIN(risk_score) * 100, 2)                           AS min_risk_pct,
    ROUND(MAX(risk_score) * 100, 2)                           AS max_risk_pct
FROM employee_risk_scores
GROUP BY risk_tier
ORDER BY FIELD(risk_tier, 'High', 'Medium', 'Low');

-- Q12: Top 15 highest-risk CURRENT employees
SELECT
    e.employee_number,
    e.age,
    e.department,
    e.job_role,
    e.gender,
    e.monthly_income,
    e.overtime,
    e.job_satisfaction,
    e.years_at_company,
    e.marital_status,
    r.risk_score,
    r.risk_tier
FROM employees e
JOIN employee_risk_scores r ON e.employee_number = r.employee_number
WHERE e.attrition = 'No'
ORDER BY r.risk_score DESC
LIMIT 15;

-- ================================================================
--  VIEW: vw_attrition_dashboard (connect directly to Power BI)
-- ================================================================
CREATE OR REPLACE VIEW vw_attrition_dashboard AS
SELECT
    e.*,
    COALESCE(r.risk_score, 0)                                  AS ml_risk_score,
    COALESCE(r.risk_tier, 'Unscored')                          AS ml_risk_tier,
    ROUND(
        (e.job_satisfaction + e.environment_satisfaction +
         e.work_life_balance + e.relationship_satisfaction + e.job_involvement) / 5.0
    , 2)                                                       AS composite_satisfaction,
    CASE
        WHEN e.years_at_company <= 2   THEN '0-2 yrs'
        WHEN e.years_at_company <= 5   THEN '3-5 yrs'
        WHEN e.years_at_company <= 10  THEN '6-10 yrs'
        ELSE '10+ yrs'
    END                                                        AS tenure_band,
    CASE
        WHEN e.monthly_income < 3000   THEN 'Q1 (<$3k)'
        WHEN e.monthly_income < 6000   THEN 'Q2 ($3-6k)'
        WHEN e.monthly_income < 10000  THEN 'Q3 ($6-10k)'
        ELSE 'Q4 ($10k+)'
    END                                                        AS income_band,
    CASE
        WHEN e.job_satisfaction = 1 AND e.overtime = 'Yes'    THEN 'Critical'
        WHEN e.job_satisfaction <= 2 OR e.overtime = 'Yes'    THEN 'At Risk'
        WHEN e.work_life_balance <= 2                          THEN 'Monitor'
        ELSE 'Stable'
    END                                                        AS hr_flag
FROM employees e
LEFT JOIN employee_risk_scores r ON e.employee_number = r.employee_number;
