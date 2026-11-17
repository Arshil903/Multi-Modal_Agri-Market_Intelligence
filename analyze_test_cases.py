"""Analyze test cases used for metric calculation - FOR VIVA PREPARATION."""

import pandas as pd
import numpy as np

print("="*80)
print("TEST CASE ANALYSIS - HOW YOUR METRICS ARE CALCULATED")
print("="*80)

# Load data
df = pd.read_csv('data/processed/gujarat_top_multimodal_master.csv', parse_dates=['Date'])
bench_df = pd.read_csv('data/processed/model_benchmark_results.csv')

# Filter Onion (as in your report)
onion_df = df[df['Commodity'] == 'Onion'].copy()
onion_df = onion_df.sort_values('Date')

print(f"\n📁 DATASET OVERVIEW:")
print(f"   Total Onion records: {len(onion_df)}")
print(f"   Date range: {onion_df['Date'].min().strftime('%Y-%m-%d')} to {onion_df['Date'].max().strftime('%Y-%m-%d')}")

# After feature engineering, some rows have NaN (first 14 days for lag features)
usable_df = onion_df.dropna(subset=['Price_Lag_14', 'Price_Roll_Mean_30', 'Target_Price'])
print(f"   Usable rows (after dropping NaN): {len(usable_df)}")

# 80/20 chronological split
split_idx = int(len(usable_df) * 0.80)
train_df = usable_df.iloc[:split_idx]
test_df = usable_df.iloc[split_idx:]

print(f"\n🔀 TRAIN/TEST SPLIT (Chronological, No Shuffle):")
print(f"   Train samples: {len(train_df)} (80%)")
print(f"   Test samples: {len(test_df)} (20%)")
print(f"   Split date: {test_df.iloc[0]['Date'].strftime('%Y-%m-%d')}")

print(f"\n🧪 TEST SET (What Your Metrics Are Calculated From):")
print(f"   Start: {test_df['Date'].min().strftime('%Y-%m-%d')}")
print(f"   End: {test_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"   Duration: {(test_df['Date'].max() - test_df['Date'].min()).days} days")
print(f"   Total test samples: {len(test_df)}")

# Regression target analysis
print(f"\n💰 REGRESSION TARGET (Next-Day Price):")
print(f"   Mean price: ₹{test_df['Target_Price'].mean():.2f}/quintal")
print(f"   Std dev: ₹{test_df['Target_Price'].std():.2f}")
print(f"   Min: ₹{test_df['Target_Price'].min():.2f}")
print(f"   Max: ₹{test_df['Target_Price'].max():.2f}")
print(f"   Range: ₹{test_df['Target_Price'].max() - test_df['Target_Price'].min():.2f}")

# Classification target analysis
shock_counts = test_df['Target_Shock'].value_counts()
n_normal = shock_counts.get(0, 0)
n_shock = shock_counts.get(1, 0)

print(f"\n⚠️  CLASSIFICATION TARGET (Shock Detection):")
print(f"   Normal days (class 0): {n_normal} ({n_normal/len(test_df)*100:.1f}%)")
print(f"   Shock days (class 1): {n_shock} ({n_shock/len(test_df)*100:.1f}%)")
print(f"   Class imbalance: {n_normal:.0f}:{n_shock:.0f} ratio")

if n_shock > 0:
    shock_dates = test_df[test_df['Target_Shock'] == 1][['Date', 'Modal_Price', 'Target_Price', 'Price_Roll_Mean_30', 'Price_Roll_Std_30']]
    print(f"\n   🔴 ACTUAL SHOCK DAYS IN TEST SET ({n_shock} total):")
    for idx, row in shock_dates.iterrows():
        z_score = (row['Target_Price'] - row['Price_Roll_Mean_30']) / row['Price_Roll_Std_30']
        print(f"      • {row['Date'].strftime('%Y-%m-%d')}: ₹{row['Modal_Price']:.0f} → ₹{row['Target_Price']:.0f} (z={z_score:.2f})")

# Get actual benchmark metrics
print(f"\n📊 YOUR REPORTED METRICS (from model_benchmark_results.csv):")
print(f"\n   REGRESSION (Price Forecasting):")
ridge_reg = bench_df[(bench_df['Commodity']=='Onion') & (bench_df['Task']=='A_Price_Forecast') & (bench_df['Model']=='Ridge') & (bench_df['Regime']=='market_only')].iloc[0]
print(f"      Ridge (Market-Only):")
print(f"         R² = {ridge_reg['R2']:.4f} (explains {ridge_reg['R2']*100:.2f}% of variance)")
print(f"         RMSE = ₹{ridge_reg['RMSE']:.2f}/quintal")
print(f"         MAE = ₹{ridge_reg['MAE']:.2f}/quintal")

xgb_reg = bench_df[(bench_df['Commodity']=='Onion') & (bench_df['Task']=='A_Price_Forecast') & (bench_df['Model']=='XGBoost') & (bench_df['Regime']=='market_only')].iloc[0]
print(f"      XGBoost (Market-Only):")
print(f"         R² = {xgb_reg['R2']:.4f}")
print(f"         RMSE = ₹{xgb_reg['RMSE']:.2f}/quintal")

bilstm_reg = bench_df[(bench_df['Commodity']=='Onion') & (bench_df['Task']=='A_Price_Forecast') & (bench_df['Model']=='BiLSTM') & (bench_df['Regime']=='market_only')].iloc[0]
print(f"      BiLSTM (Market-Only):")
print(f"         R² = {bilstm_reg['R2']:.4f}")
print(f"         RMSE = ₹{bilstm_reg['RMSE']:.2f}/quintal")

print(f"\n   CLASSIFICATION (Shock Detection):")
logreg_clf = bench_df[(bench_df['Commodity']=='Onion') & (bench_df['Task']=='B_Shock_Detection') & (bench_df['Model']=='LogisticRegression') & (bench_df['Regime']=='market_only')].iloc[0]
print(f"      Logistic Regression (Market-Only) - ONLY MODEL THAT DETECTS:")
print(f"         Accuracy = {logreg_clf['Accuracy']:.4f} ({logreg_clf['Accuracy']*100:.2f}%)")
print(f"         Precision = {logreg_clf['Precision']:.4f}")
print(f"         Recall = {logreg_clf['Recall']:.4f}")
print(f"         ROC-AUC = {logreg_clf['ROC_AUC']:.4f}")
print(f"         Macro F1 = {logreg_clf['Macro_F1']:.4f}")

rf_clf = bench_df[(bench_df['Commodity']=='Onion') & (bench_df['Task']=='B_Shock_Detection') & (bench_df['Model']=='RandomForest_SMOTE') & (bench_df['Regime']=='market_only')].iloc[0]
print(f"      Random Forest (Market-Only):")
print(f"         Accuracy = {rf_clf['Accuracy']:.4f} ({rf_clf['Accuracy']*100:.2f}%)")
print(f"         Precision = {rf_clf['Precision']:.4f}")
print(f"         Recall = {rf_clf['Recall']:.4f} (predicts ALL as normal!)")

# Back-calculate confusion matrix from metrics
print(f"\n🔍 CONFUSION MATRIX RECONSTRUCTION:")
print(f"   (How {n_shock} shock days and {n_normal} normal days are classified)")

n_test = len(test_df)
acc_logreg = logreg_clf['Accuracy']
prec_logreg = logreg_clf['Precision']
rec_logreg = logreg_clf['Recall']

# From recall: TP = recall × actual_positives
tp_logreg = int(rec_logreg * n_shock)
fn_logreg = n_shock - tp_logreg

# From precision: TP / (TP + FP) = precision → FP = TP/precision - TP
if prec_logreg > 0:
    fp_logreg = int(tp_logreg / prec_logreg - tp_logreg)
else:
    fp_logreg = 0

tn_logreg = n_normal - fp_logreg

print(f"\n   Logistic Regression (Market-Only):")
print(f"   ┌─────────────┬──────────────┬─────────────┐")
print(f"   │             │ Pred Normal  │ Pred Shock  │")
print(f"   ├─────────────┼──────────────┼─────────────┤")
print(f"   │ True Normal │ TN = {tn_logreg:<3}      │ FP = {fp_logreg:<3}     │ → {n_normal} normal days")
print(f"   │ True Shock  │ FN = {fn_logreg:<3}      │ TP = {tp_logreg:<3}     │ → {n_shock} shock days")
print(f"   └─────────────┴──────────────┴─────────────┘")
print(f"   Accuracy = (TP + TN) / Total = ({tp_logreg} + {tn_logreg}) / {n_test} = {acc_logreg:.4f}")
print(f"   Precision = TP / (TP + FP) = {tp_logreg} / ({tp_logreg} + {fp_logreg}) = {prec_logreg:.4f}")
print(f"   Recall = TP / (TP + FN) = {tp_logreg} / ({tp_logreg} + {fn_logreg}) = {rec_logreg:.4f}")

# Random Forest
acc_rf = rf_clf['Accuracy']
prec_rf = rf_clf['Precision']
rec_rf = rf_clf['Recall']

print(f"\n   Random Forest (Market-Only) - Predicts ALL as Normal:")
print(f"   ┌─────────────┬──────────────┬─────────────┐")
print(f"   │             │ Pred Normal  │ Pred Shock  │")
print(f"   ├─────────────┼──────────────┼─────────────┤")
print(f"   │ True Normal │ TN = {n_normal:<3}      │ FP = 0       │ → {n_normal} normal days")
print(f"   │ True Shock  │ FN = {n_shock:<3}      │ TP = 0       │ → {n_shock} shock days")
print(f"   └─────────────┴──────────────┴─────────────┘")
print(f"   Accuracy = TN / Total = {n_normal} / {n_test} = {acc_rf:.4f} (HIGH but USELESS!)")
print(f"   Precision = 0 / 0 = {prec_rf:.4f} (never predicts positive)")
print(f"   Recall = 0 / {n_shock} = {rec_rf:.4f} (misses all shocks)")

# Sample test cases
print(f"\n📋 SAMPLE TEST CASES (First 10 of {len(test_df)}):")
print(test_df[['Date', 'Modal_Price', 'Target_Price', 'Target_Shock', 'Price_Roll_Mean_30']].head(10).to_string(index=False))

print(f"\n📋 SAMPLE TEST CASES (Last 10):")
print(test_df[['Date', 'Modal_Price', 'Target_Price', 'Target_Shock']].tail(10).to_string(index=False))

# Key takeaway
print(f"\n" + "="*80)
print("🎯 KEY TAKEAWAYS FOR VIVA:")
print("="*80)
print(f"1. Test set: {len(test_df)} onion price observations from {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"2. Extreme class imbalance: {n_shock} shocks vs {n_normal} normal days (~{n_shock/len(test_df)*100:.1f}% positive class)")
print(f"3. Ridge achieves 91.76% R² because prices are highly autocorrelated (Price_Lag_1 is strongest predictor)")
print(f"4. Only Logistic Regression detects shocks (recall=50% means it found {tp_logreg} out of {n_shock} shock days)")
print(f"5. Random Forest has 97.24% accuracy but 0% recall—it NEVER predicts shocks!")
print(f"6. Accuracy is misleading for imbalanced data. Always use ROC-AUC and recall.")
print(f"7. Back-calculation suggests ~{tp_logreg} true positive, ~{fp_logreg} false positives in test set")
print("="*80)
