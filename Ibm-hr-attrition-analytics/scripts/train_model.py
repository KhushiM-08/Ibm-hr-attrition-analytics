"""
IBM HR Analytics — Full Analysis Pipeline
==========================================
Dataset : WA_Fn-UseC_-HR-Employee-Attrition.csv  (IBM, 1470 rows, 35 cols)
Steps   : Load → Clean → EDA → Feature Eng → 3 Models → SHAP → Export
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, json, pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve, accuracy_score,
                                       f1_score, precision_score, recall_score)
import shap

# ── Paths ────────────────────────────────────────────────────
BASE        = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR    = os.path.join(BASE, 'data')
REPORTS_DIR = os.path.join(BASE, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────
COLORS = {
    'red'   : '#E24B4A',
    'green' : '#1D9E75',
    'blue'  : '#378ADD',
    'purple': '#7F77DD',
    'orange': '#EF9F27',
    'teal'  : '#17BECF',
}
ATTRITION_PAL = {'Yes': COLORS['red'], 'No': COLORS['green']}

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.facecolor'    : '#FAFAFA',
    'figure.facecolor'  : 'white',
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
})

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  IBM HR ANALYTICS — ATTRITION ANALYSIS PIPELINE")
print("=" * 60)

# ── 1. LOAD ──────────────────────────────────────────────────
print("\n[1/7] Loading IBM HR dataset...")
csv = os.path.join(DATA_DIR, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
df  = pd.read_csv(csv)
print(f"      Shape  : {df.shape}")
print(f"      Missing: {df.isnull().sum().sum()}")
print(f"      Attrition Yes: {(df['Attrition']=='Yes').sum()} "
      f"({(df['Attrition']=='Yes').mean()*100:.1f}%)")

# ── 2. CLEAN ─────────────────────────────────────────────────
print("\n[2/7] Cleaning data...")
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Attrition']
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols = [c for c in num_cols if c != 'Attrition_Binary']

print(f"      Categorical : {cat_cols}")
print(f"      Numerical   : {len(num_cols)} columns")

df_clean = df.copy()
df_clean.to_csv(os.path.join(DATA_DIR, 'hr_attrition_clean.csv'), index=False)

# ── 3. EDA CHARTS ────────────────────────────────────────────
print("\n[3/7] Generating EDA charts...")

# Fig 01 — Attrition distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['Attrition'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[ATTRITION_PAL[k] for k in counts.index],
              width=0.4, edgecolor='white', linewidth=2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
            f'{val}\n({val/len(df)*100:.1f}%)',
            ha='center', fontsize=11, fontweight='600')
ax.set_title('Employee Attrition Distribution', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('Attrition'); ax.set_ylabel('Number of Employees')
ax.set_ylim(0, counts.max()*1.25)
save('fig_01_attrition_distribution.png')

# Fig 02 — Attrition by Department
fig, ax = plt.subplots(figsize=(9, 4))
dept = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()*100
dept.plot(kind='bar', ax=ax,
          color=[ATTRITION_PAL['No'], ATTRITION_PAL['Yes']],
          edgecolor='white', linewidth=1.2, width=0.6)
ax.set_title('Attrition Rate by Department (%)', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel(''); ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.legend(['Stayed', 'Left'], loc='upper right')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=9, padding=3)
save('fig_02_attrition_by_department.png')

# Fig 03 — Age vs Attrition
fig, ax = plt.subplots(figsize=(10, 4))
for lbl, grp in df.groupby('Attrition'):
    ax.hist(grp['Age'], bins=20, alpha=0.70, label=lbl,
            color=ATTRITION_PAL[lbl], edgecolor='white')
ax.axvline(df[df['Attrition']=='Yes']['Age'].mean(), color=COLORS['red'],
           linestyle='--', linewidth=2, alpha=0.8,
           label=f"Left mean={df[df['Attrition']=='Yes']['Age'].mean():.1f}")
ax.axvline(df[df['Attrition']=='No']['Age'].mean(), color=COLORS['green'],
           linestyle='--', linewidth=2, alpha=0.8,
           label=f"Stayed mean={df[df['Attrition']=='No']['Age'].mean():.1f}")
ax.set_title('Age Distribution by Attrition', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(fontsize=9)
save('fig_03_age_distribution.png')

# Fig 04 — Monthly Income vs Attrition (box plot)
fig, ax = plt.subplots(figsize=(8, 4))
data_box = [df[df['Attrition']=='No']['MonthlyIncome'].values,
            df[df['Attrition']=='Yes']['MonthlyIncome'].values]
bp = ax.boxplot(data_box, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2.5))
for patch, color in zip(bp['boxes'], [COLORS['green'], COLORS['red']]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(['Stayed', 'Left'])
ax.set_title('Monthly Income Distribution by Attrition', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('Attrition Status'); ax.set_ylabel('Monthly Income ($)')
med_stayed = np.median(df[df['Attrition']=='No']['MonthlyIncome'])
med_left   = np.median(df[df['Attrition']=='Yes']['MonthlyIncome'])
ax.text(1, med_stayed+200, f'Median: ${med_stayed:,.0f}', ha='center', fontsize=10, color=COLORS['green'])
ax.text(2, med_left+200,   f'Median: ${med_left:,.0f}',   ha='center', fontsize=10, color=COLORS['red'])
save('fig_04_income_boxplot.png')

# Fig 05 — Overtime vs Attrition
fig, ax = plt.subplots(figsize=(7, 4))
ot = df.groupby(['OverTime','Attrition']).size().unstack(fill_value=0)
ot_pct = ot.div(ot.sum(axis=1), axis=0)*100
ot_pct.plot(kind='bar', ax=ax,
            color=[ATTRITION_PAL['No'], ATTRITION_PAL['Yes']],
            edgecolor='white', width=0.5)
ax.set_title('Overtime vs Attrition Rate', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('OverTime'); ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Stayed', 'Left'])
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=9, padding=2)
save('fig_05_overtime_vs_attrition.png')

# Fig 06 — Satisfaction heatmap
fig, ax = plt.subplots(figsize=(10, 4))
sat_cols = ['JobSatisfaction','EnvironmentSatisfaction',
            'WorkLifeBalance','RelationshipSatisfaction','JobInvolvement']
sat_means = df.groupby('Attrition')[sat_cols].mean()
sns.heatmap(sat_means, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, vmin=1, vmax=4,
            cbar_kws={'shrink':0.8,'label':'Score (1=Low, 4=High)'})
ax.set_title('Mean Satisfaction Scores — Stayed vs Left', fontsize=14, fontweight='600', pad=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(['Job Sat','Env Sat','Work-Life','Relationship','Job Inv'], rotation=20, ha='right')
save('fig_06_satisfaction_heatmap.png')

# Fig 07 — Job Role attrition rates
fig, ax = plt.subplots(figsize=(10, 5))
role_attr = df.groupby('JobRole').agg(
    Total=('Attrition','count'),
    Left=('Attrition', lambda x: (x=='Yes').sum())
).reset_index()
role_attr['Rate'] = role_attr['Left'] / role_attr['Total'] * 100
role_attr = role_attr.sort_values('Rate', ascending=True)
colors_bar = [COLORS['red'] if r > 20 else COLORS['blue'] for r in role_attr['Rate']]
bars = ax.barh(role_attr['JobRole'], role_attr['Rate'],
               color=colors_bar, edgecolor='white')
ax.axvline(role_attr['Rate'].mean(), color='gray', linestyle='--',
           linewidth=1.5, label=f"Average {role_attr['Rate'].mean():.1f}%")
ax.set_title('Attrition Rate by Job Role', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('Attrition Rate (%)'); ax.legend()
for bar, val in zip(bars, role_attr['Rate']):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)
save('fig_07_jobrole_attrition.png')

# Fig 08 — Marital Status vs Attrition
fig, ax = plt.subplots(figsize=(7, 4))
ms = df.groupby(['MaritalStatus','Attrition']).size().unstack(fill_value=0)
ms_pct = ms.div(ms.sum(axis=1),axis=0)*100
ms_pct.plot(kind='bar', ax=ax,
            color=[ATTRITION_PAL['No'], ATTRITION_PAL['Yes']],
            edgecolor='white', width=0.55)
ax.set_title('Marital Status vs Attrition Rate', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel(''); ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Stayed','Left'])
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=9, padding=2)
save('fig_08_marital_attrition.png')

# Fig 09 — Income by Department (violin)
fig, ax = plt.subplots(figsize=(10, 5))
depts = df['Department'].unique()
for i, dept in enumerate(depts):
    sub = df[df['Department']==dept]
    parts = ax.violinplot([sub[sub['Attrition']=='No']['MonthlyIncome'].values,
                           sub[sub['Attrition']=='Yes']['MonthlyIncome'].values],
                          positions=[i*3, i*3+1], widths=0.8)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS['green'] if j==0 else COLORS['red'])
        pc.set_alpha(0.7)
ax.set_xticks([0.5, 3.5, 6.5])
ax.set_xticklabels(depts)
ax.set_title('Income Distribution by Department & Attrition', fontsize=14, fontweight='600', pad=12)
ax.set_ylabel('Monthly Income ($)')
green_p = mpatches.Patch(color=COLORS['green'], alpha=0.7, label='Stayed')
red_p   = mpatches.Patch(color=COLORS['red'],   alpha=0.7, label='Left')
ax.legend(handles=[green_p, red_p])
save('fig_09_income_violin.png')

# Fig 10 — Distance from Home vs Attrition
fig, ax = plt.subplots(figsize=(9, 4))
bins = [0,5,10,15,20,30]
labels_b = ['0-5','6-10','11-15','16-20','21-30']
df['DistBand'] = pd.cut(df['DistanceFromHome'], bins=bins, labels=labels_b)
dist = df.groupby(['DistBand','Attrition']).size().unstack(fill_value=0)
dist_pct = dist.div(dist.sum(axis=1),axis=0)*100
dist_pct.plot(kind='bar', ax=ax,
              color=[ATTRITION_PAL['No'], ATTRITION_PAL['Yes']],
              edgecolor='white', width=0.6)
ax.set_title('Distance from Home vs Attrition', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('Distance Band (km)'); ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Stayed','Left'])
save('fig_10_distance_attrition.png')

df.drop(columns=['DistBand'], inplace=True, errors='ignore')
print(f"      10 EDA charts saved → reports/")

# ── 4. FEATURE ENGINEERING ───────────────────────────────────
print("\n[4/7] Feature engineering...")

df_model = df_clean.copy()
le_map   = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    le_map[col] = le

feature_cols = [c for c in df_model.columns
                if c not in ['Attrition','Attrition_Binary']]
X = df_model[feature_cols]
y = df_model['Attrition_Binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s   = scaler.transform(X_test)

print(f"      Features : {len(feature_cols)}")
print(f"      Train    : {len(X_train)} | Test: {len(X_test)}")
print(f"      Class ratio (train) Yes={y_train.sum()} No={len(y_train)-y_train.sum()}")

# ── 5. MODEL TRAINING ────────────────────────────────────────
print("\n[5/7] Training 3 models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42),
    'Logistic Regression': LogisticRegression(
        max_iter=2000, class_weight='balanced',
        solver='lbfgs', random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_s if name == 'Logistic Regression' else X_train
    X_te = X_test_s  if name == 'Logistic Regression' else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='roc_auc')
    results[name] = {
        'model'    : model,
        'accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'roc_auc'  : round(roc_auc_score(y_test, y_prob), 4),
        'f1'       : round(f1_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall'   : round(recall_score(y_test, y_pred), 4),
        'cv_mean'  : round(cv.mean(), 4),
        'cv_std'   : round(cv.std(), 4),
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
    }
    print(f"      {name:<25} "
          f"Acc:{results[name]['accuracy']:.3f}  "
          f"AUC:{results[name]['roc_auc']:.3f}  "
          f"F1:{results[name]['f1']:.3f}  "
          f"CV:{results[name]['cv_mean']:.3f}±{results[name]['cv_std']:.3f}")

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best = results[best_name]
print(f"\n      ✔ Best model: {best_name} (AUC={best['roc_auc']})")

# ── 6. EVALUATION CHARTS ─────────────────────────────────────
print("\n[6/7] Generating model evaluation charts...")

# Fig 11 — ROC curves all 3 models
fig, ax = plt.subplots(figsize=(7, 5))
roc_colors = [COLORS['purple'], COLORS['red'], COLORS['green']]
for (name, res), col in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
            color=col, linewidth=2.2)
ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.4,label='Random (0.500)')
ax.fill_between(fpr, tpr, alpha=0.03, color=COLORS['purple'])
ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right', fontsize=10)
save('fig_11_roc_curves.png')

# Fig 12 — Confusion matrix (best model)
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted Stay','Predicted Leave'],
            yticklabels=['Actual Stay','Actual Leave'],
            linewidths=0.5, cbar=False, annot_kws={'size':13,'weight':'bold'})
ax.set_title(f'Confusion Matrix\n{best_name}', fontsize=13, fontweight='600', pad=12)
save('fig_12_confusion_matrix.png')

# Fig 13 — Feature importance (RF)
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
q3 = fi.quantile(0.75)
colors_fi = [COLORS['red'] if v >= q3 else COLORS['blue'] for v in fi.values]
bars = ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='white')
ax.set_title('Top 15 Feature Importances — Random Forest', fontsize=14, fontweight='600', pad=12)
ax.set_xlabel('Importance Score')
high_p  = mpatches.Patch(color=COLORS['red'],  label='High importance (top 25%)')
norm_p  = mpatches.Patch(color=COLORS['blue'], label='Normal importance')
ax.legend(handles=[high_p, norm_p], loc='lower right')
for bar, val in zip(bars, fi.values):
    ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
save('fig_13_feature_importance.png')

# Fig 14 — SHAP importance
print("      Computing SHAP values...")
try:
    explainer = shap.TreeExplainer(rf)
    X_shap    = X_test.sample(min(300, len(X_test)), random_state=42)
    shap_vals = explainer.shap_values(X_shap)
    sv = shap_vals[:,:,1] if shap_vals.ndim == 3 else shap_vals
    shap_mean = np.abs(sv).mean(axis=0)
    shap_ser  = pd.Series(shap_mean, index=feature_cols).sort_values(ascending=True).tail(15)
    fig, ax   = plt.subplots(figsize=(9, 6))
    ax.barh(shap_ser.index, shap_ser.values, color=COLORS['purple'], edgecolor='white')
    ax.set_title('SHAP — Mean |SHAP Value| per Feature\n(Impact on Attrition Prediction)',
                 fontsize=13, fontweight='600', pad=12)
    ax.set_xlabel('Mean |SHAP value|')
    save('fig_14_shap_importance.png')
    print("      SHAP chart saved.")
except Exception as e:
    print(f"      SHAP skipped: {e}")

# Fig 15 — Model metrics comparison
fig, ax = plt.subplots(figsize=(9, 4))
metrics_compare = ['accuracy','roc_auc','f1','precision','recall']
labels_m = ['Accuracy','ROC-AUC','F1','Precision','Recall']
model_names_list = list(results.keys())
x = np.arange(len(metrics_compare))
width = 0.25
bar_colors = [COLORS['purple'], COLORS['red'], COLORS['green']]
for i, (mname, col) in enumerate(zip(model_names_list, bar_colors)):
    vals = [results[mname][m] for m in metrics_compare]
    bars_m = ax.bar(x + i*width, vals, width, label=mname, color=col,
                    edgecolor='white')
    for bar, v in zip(bars_m, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.2f}', ha='center', fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels_m)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison — All Metrics', fontsize=14, fontweight='600', pad=12)
ax.legend(loc='upper right', fontsize=9)
save('fig_15_model_comparison.png')

print(f"      15 charts saved → reports/")

# ── 7. EXPORT ────────────────────────────────────────────────
print("\n[7/7] Exporting model artifacts and reports...")

# Save model artifacts
with open(os.path.join(DATA_DIR, 'rf_model.pkl'),       'wb') as f: pickle.dump(rf, f)
with open(os.path.join(DATA_DIR, 'scaler.pkl'),         'wb') as f: pickle.dump(scaler, f)
with open(os.path.join(DATA_DIR, 'feature_cols.json'),  'w')  as f: json.dump(feature_cols, f)
with open(os.path.join(DATA_DIR, 'label_encoders.pkl'), 'wb') as f: pickle.dump(le_map, f)

# Risk scores on entire dataset
X_all           = df_model[feature_cols]
risk_scores     = rf.predict_proba(X_all)[:, 1]
df_out          = df_clean.copy()
df_out['AttritionRisk_Score'] = risk_scores.round(4)
df_out['AttritionRisk_Tier']  = pd.cut(risk_scores,
                                        bins=[0,.30,.60,1.0],
                                        labels=['Low','Medium','High'])
df_out.to_csv(os.path.join(DATA_DIR, 'hr_attrition_with_risk.csv'), index=False)

# ── Excel report (5 sheets) ──────────────────────────────────
excel_path = os.path.join(DATA_DIR, 'hr_attrition_report.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1: Full dataset with risk
    df_out.to_excel(writer, sheet_name='Employee Risk Data', index=False)

    # Sheet 2: Model summary
    summary = pd.DataFrame({
        'Metric': ['Dataset','Total Employees','Attrition Count','Attrition Rate (%)',
                   'High Risk','Medium Risk','Low Risk',
                   'Best Model','Best ROC-AUC','Best Accuracy','Best F1',
                   'RF AUC','GB AUC','LR AUC'],
        'Value': [
            'IBM HR Analytics (WA_Fn-UseC_-HR-Employee-Attrition)',
            len(df_out),
            int((df_out['Attrition']=='Yes').sum()),
            round((df_out['Attrition']=='Yes').mean()*100,2),
            int((df_out['AttritionRisk_Tier']=='High').sum()),
            int((df_out['AttritionRisk_Tier']=='Medium').sum()),
            int((df_out['AttritionRisk_Tier']=='Low').sum()),
            best_name,
            best['roc_auc'],
            best['accuracy'],
            best['f1'],
            results['Random Forest']['roc_auc'],
            results['Gradient Boosting']['roc_auc'],
            results['Logistic Regression']['roc_auc'],
        ]
    })
    summary.to_excel(writer, sheet_name='Model Summary', index=False)

    # Sheet 3: Department breakdown
    dept_risk = df_out.groupby('Department').agg(
        Total        =('Attrition','count'),
        Attrition_Yes=('Attrition', lambda x:(x=='Yes').sum()),
        Attrition_Rate=('Attrition',lambda x:round((x=='Yes').mean()*100,2)),
        Avg_Risk_Score=('AttritionRisk_Score',lambda x:round(x.mean(),4)),
        High_Risk    =('AttritionRisk_Tier', lambda x:(x=='High').sum()),
        Avg_Income   =('MonthlyIncome',       lambda x:round(x.mean(),0)),
        Avg_JobSat   =('JobSatisfaction',     lambda x:round(x.mean(),2)),
    ).reset_index()
    dept_risk.to_excel(writer, sheet_name='Dept Breakdown', index=False)

    # Sheet 4: Job Role breakdown
    role_risk = df_out.groupby('JobRole').agg(
        Total        =('Attrition','count'),
        Attrition_Yes=('Attrition', lambda x:(x=='Yes').sum()),
        Attrition_Rate=('Attrition',lambda x:round((x=='Yes').mean()*100,2)),
        Avg_Income   =('MonthlyIncome',lambda x:round(x.mean(),0)),
        High_Risk    =('AttritionRisk_Tier',lambda x:(x=='High').sum()),
    ).reset_index().sort_values('Attrition_Rate',ascending=False)
    role_risk.to_excel(writer, sheet_name='Role Breakdown', index=False)

    # Sheet 5: Feature importance
    fi_df = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fi_df.to_excel(writer, sheet_name='Feature Importance', index=False)

print(f"      Excel report: {excel_path}")

# ── Metrics JSON (for web app) ───────────────────────────────
role_risk_list = role_risk.to_dict(orient='records')
for item in role_risk_list:
    for k,v in item.items():
        if hasattr(v,'item'): item[k]=v.item()

metrics_out = {
    'dataset'         : 'IBM HR Analytics Employee Attrition',
    'total_employees' : int(len(df_out)),
    'attrition_count' : int((df_out['Attrition']=='Yes').sum()),
    'attrition_rate'  : round((df_out['Attrition']=='Yes').mean()*100,2),
    'high_risk'       : int((df_out['AttritionRisk_Tier']=='High').sum()),
    'medium_risk'     : int((df_out['AttritionRisk_Tier']=='Medium').sum()),
    'low_risk'        : int((df_out['AttritionRisk_Tier']=='Low').sum()),
    'best_model'      : best_name,
    'best_auc'        : best['roc_auc'],
    'best_accuracy'   : best['accuracy'],
    'best_f1'         : best['f1'],
    'model_results'   : {k:{m:v for m,v in r.items()
                             if m not in ['model','y_pred','y_prob']}
                         for k,r in results.items()},
    'dept_breakdown'  : dept_risk.to_dict(orient='records'),
    'role_breakdown'  : role_risk_list,
    'top_features'    : fi_df.head(10).to_dict(orient='records'),
}
for k,v in metrics_out.items():
    if isinstance(v,list):
        for item in v:
            if isinstance(item,dict):
                for ik,iv in item.items():
                    if hasattr(iv,'item'): item[ik]=iv.item()

with open(os.path.join(DATA_DIR,'metrics.json'),'w') as f:
    json.dump(metrics_out, f, indent=2, default=str)

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Charts   : reports/ ({len(os.listdir(REPORTS_DIR))} files)")
print(f"  Data     : data/hr_attrition_with_risk.csv")
print(f"  Excel    : data/hr_attrition_report.xlsx (5 sheets)")
print(f"  Model    : data/rf_model.pkl")
print(f"  Metrics  : data/metrics.json")
print("=" * 60)
