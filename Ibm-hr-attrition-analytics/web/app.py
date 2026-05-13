"""
HR Attrition Analysis — Flask Web Application
IBM HR Analytics Dataset (1470 employees, 35 features)
Run: python web/app.py  →  http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd, numpy as np, json, os, pickle
from sklearn.preprocessing import LabelEncoder

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, '..', 'data')
REPORTS_DIR = os.path.join(BASE_DIR, '..', 'reports')

app = Flask(__name__)

# ── helpers ─────────────────────────────────────────────────
def load_df():
    p = os.path.join(DATA_DIR, 'hr_attrition_with_risk.csv')
    return pd.read_csv(p)

def load_metrics():
    with open(os.path.join(DATA_DIR, 'metrics.json')) as f:
        return json.load(f)

# ── pages ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', metrics=load_metrics())

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', metrics=load_metrics())

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/explorer')
def explorer():
    return render_template('explorer.html')

@app.route('/insights')
def insights():
    return render_template('insights.html', metrics=load_metrics())

# ── API ──────────────────────────────────────────────────────
@app.route('/api/metrics')
def api_metrics():
    return jsonify(load_metrics())

@app.route('/api/employees')
def api_employees():
    df = load_df()
    cols = ['Age','Department','JobRole','Gender','MaritalStatus','MonthlyIncome',
            'YearsAtCompany','Attrition','OverTime','JobSatisfaction',
            'WorkLifeBalance','EnvironmentSatisfaction','BusinessTravel',
            'DistanceFromHome','Education','JobLevel','StockOptionLevel',
            'AttritionRisk_Score','AttritionRisk_Tier']
    cols = [c for c in cols if c in df.columns]
    return jsonify(df[cols].to_dict(orient='records'))

@app.route('/api/dept-breakdown')
def api_dept():
    df = load_df()
    out = df.groupby('Department').agg(
        total      =('Attrition','count'),
        attrition  =('Attrition', lambda x:(x=='Yes').sum()),
        rate       =('Attrition', lambda x:round((x=='Yes').mean()*100,2)),
        avg_income =('MonthlyIncome', lambda x:round(x.mean(),0)),
        avg_jobsat =('JobSatisfaction', lambda x:round(x.mean(),2)),
        high_risk  =('AttritionRisk_Tier', lambda x:(x=='High').sum()),
    ).reset_index()
    return jsonify(out.to_dict(orient='records'))

@app.route('/api/role-breakdown')
def api_role():
    df = load_df()
    out = df.groupby('JobRole').agg(
        total    =('Attrition','count'),
        attrition=('Attrition', lambda x:(x=='Yes').sum()),
        rate     =('Attrition', lambda x:round((x=='Yes').mean()*100,2)),
        avg_income=('MonthlyIncome', lambda x:round(x.mean(),0)),
    ).reset_index().sort_values('rate',ascending=False)
    return jsonify(out.to_dict(orient='records'))

@app.route('/api/age-dist')
def api_age():
    df = load_df()
    bins   = list(range(18, 65, 5))
    labels = [f'{b}-{b+4}' for b in bins[:-1]]
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    out = df.groupby(['AgeGroup','Attrition']).size().unstack(fill_value=0).reset_index()
    out.columns = [str(c) for c in out.columns]
    return jsonify(out.to_dict(orient='records'))

@app.route('/api/income-dist')
def api_income():
    df = load_df()
    bins   = [0,2000,4000,6000,8000,10000,15000,20000]
    labels = ['<2k','2-4k','4-6k','6-8k','8-10k','10-15k','15k+']
    df['IncomeBand'] = pd.cut(df['MonthlyIncome'], bins=bins, labels=labels)
    out = df.groupby(['IncomeBand','Attrition']).size().unstack(fill_value=0).reset_index()
    out.columns = [str(c) for c in out.columns]
    return jsonify(out.to_dict(orient='records'))

@app.route('/api/satisfaction')
def api_satisfaction():
    df  = load_df()
    sat = ['JobSatisfaction','EnvironmentSatisfaction',
           'WorkLifeBalance','RelationshipSatisfaction','JobInvolvement']
    out = []
    for col in sat:
        for attr in ['Yes','No']:
            m = df[df['Attrition']==attr][col].mean()
            out.append({'metric':col,'attrition':attr,'mean':round(m,3)})
    return jsonify(out)

@app.route('/api/overtime-attrition')
def api_overtime():
    df  = load_df()
    out = df.groupby(['OverTime','Attrition']).size().unstack(fill_value=0)
    out = (out.div(out.sum(axis=1),axis=0)*100).round(2).reset_index()
    out.columns = [str(c) for c in out.columns]
    return jsonify(out.to_dict(orient='records'))

@app.route('/api/marital-attrition')
def api_marital():
    df  = load_df()
    out = df.groupby(['MaritalStatus','Attrition']).size().unstack(fill_value=0)
    out = (out.div(out.sum(axis=1),axis=0)*100).round(2).reset_index()
    out.columns = [str(c) for c in out.columns]
    return jsonify(out.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        model_path = os.path.join(DATA_DIR, 'rf_model.pkl')
        fc_path    = os.path.join(DATA_DIR, 'feature_cols.json')
        le_path    = os.path.join(DATA_DIR, 'label_encoders.pkl')

        with open(model_path,'rb') as f: model = pickle.load(f)
        with open(fc_path)         as f: feature_cols = json.load(f)
        with open(le_path,'rb')    as f: le_map = pickle.load(f)

        df_ref = pd.read_csv(os.path.join(DATA_DIR,'hr_attrition_clean.csv'))
        df_ref_m = df_ref.copy()
        for col, le in le_map.items():
            if col in df_ref_m.columns:
                df_ref_m[col] = le.transform(df_ref_m[col].astype(str))

        inp = pd.DataFrame([data])
        for col, le in le_map.items():
            if col in inp.columns:
                try:    inp[col] = le.transform(inp[col].astype(str))
                except: inp[col] = 0

        for col in feature_cols:
            if col not in inp.columns:
                inp[col] = df_ref_m[col].median()

        X_in = inp[feature_cols]
        prob = model.predict_proba(X_in)[0][1]
        tier = 'High' if prob>=0.60 else ('Medium' if prob>=0.30 else 'Low')

        fi = dict(zip(feature_cols, model.feature_importances_))
        top5 = sorted(fi.items(), key=lambda x:x[1], reverse=True)[:5]

        return jsonify({
            'probability': round(float(prob)*100,1),
            'tier': tier,
            'top_factors': [{'feature':k,'importance':round(v,4)} for k,v in top5]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)

if __name__ == '__main__':
    print("\n" + "="*52)
    print("  IBM HR Attrition Analytics — Running")
    print("  http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=True, port=5000)
