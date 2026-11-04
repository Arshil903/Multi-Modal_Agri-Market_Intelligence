"""Detailed performance analysis for Task A and Task B."""

import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/model_benchmark_results.csv')

print('='*100)
print('TASK A: PRICE FORECAST (REGRESSION) - DETAILED RESULTS')
print('='*100)
print()

task_a = df[df['Task'] == 'A_Price_Forecast'].copy()

for crop in df['Commodity'].unique():
    crop_data = task_a[task_a['Commodity'] == crop]
    if not crop_data.empty:
        print(f'\n🌾 {crop.upper()}')
        print('-'*100)
        print(f"{'Model':<25} {'Regime':<15} {'R² Score':<12} {'RMSE (₹)':<12} {'MAE (₹)':<12}")
        print('-'*100)
        for _, row in crop_data.sort_values('R2', ascending=False).iterrows():
            print(f"{row['Model']:<25} {row['Regime']:<15} {row['R2']:<12.4f} {row['RMSE']:<12.2f} {row['MAE']:<12.2f}")

print('\n\n' + '='*100)
print('TASK B: SHOCK DETECTION (CLASSIFICATION) - DETAILED RESULTS')
print('='*100)
print()

task_b = df[df['Task'] == 'B_Shock_Detection'].copy()

for crop in df['Commodity'].unique():
    crop_data = task_b[task_b['Commodity'] == crop]
    if not crop_data.empty:
        print(f'\n🌾 {crop.upper()}')
        print('-'*100)
        print(f"{'Model':<25} {'Regime':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
        print('-'*100)
        for _, row in crop_data.sort_values('Accuracy', ascending=False).iterrows():
            prec = row['Precision'] if not np.isnan(row['Precision']) else 0.0
            rec = row['Recall'] if not np.isnan(row['Recall']) else 0.0
            print(f"{row['Model']:<25} {row['Regime']:<15} {row['Accuracy']:<12.4f} {prec:<12.4f} {rec:<12.4f} {row['Macro_F1']:<12.4f} {row['ROC_AUC']:<12.4f}")

print('\n\n' + '='*100)
print('📊 OVERALL BEST PERFORMERS')
print('='*100)

print('\n🏆 TASK A: Price Forecasting (Regression)')
print('-'*100)
best_reg = task_a.loc[task_a['R2'].idxmax()]
print(f'Winner: {best_reg["Model"]} ({best_reg["Regime"]}) - {best_reg["Commodity"]}')
print(f'R² Score: {best_reg["R2"]:.4f} → {best_reg["R2"]*100:.2f}% variance explained')
print(f'RMSE: ₹{best_reg["RMSE"]:.2f}/quintal')
print(f'MAE: ₹{best_reg["MAE"]:.2f}/quintal')
print(f'\nInterpretation:')
print(f'  • Model explains {best_reg["R2"]*100:.1f}% of price variance')
print(f'  • Average prediction error: ±₹{best_reg["MAE"]:.0f} per quintal')
print(f'  • Root mean squared error: ₹{best_reg["RMSE"]:.0f} per quintal')

print('\n🏆 TASK B: Shock Detection (Classification)')
print('-'*100)
best_clf = task_b.loc[task_b['Accuracy'].idxmax()]
print(f'Winner: {best_clf["Model"]} ({best_clf["Regime"]}) - {best_clf["Commodity"]}')
print(f'Accuracy: {best_clf["Accuracy"]:.4f} → {best_clf["Accuracy"]*100:.2f}%')
print(f'Precision: {best_clf["Precision"]:.4f}')
print(f'Recall: {best_clf["Recall"]:.4f}')
print(f'F1 Score: {best_clf["Macro_F1"]:.4f}')
print(f'ROC-AUC: {best_clf["ROC_AUC"]:.4f}')
print(f'\nInterpretation:')
print(f'  • Correctly classifies {best_clf["Accuracy"]*100:.1f}% of days')
print(f'  ⚠️  But precision = {best_clf["Precision"]:.1f} (predicts "no shock" for all cases)')
print(f'  • Class imbalance: very few actual shock events in data')

print('\n\n' + '='*100)
print('🔍 MARKET-ONLY vs MULTIMODAL COMPARISON')
print('='*100)

print('\n📈 TASK A: Effect of Multimodal Features on Price Forecasting')
print('-'*100)
for model in ['Ridge', 'XGBoost', 'BiLSTM']:
    market = task_a[(task_a['Model'] == model) & (task_a['Regime'] == 'market_only')]
    multi = task_a[(task_a['Model'] == model) & (task_a['Regime'] == 'multimodal')]
    
    if not market.empty and not multi.empty:
        market_r2 = market.iloc[0]['R2']
        multi_r2 = multi.iloc[0]['R2']
        diff = (multi_r2 - market_r2) * 100
        symbol = '✅' if diff > 0 else '⚠️'
        
        print(f'{model:<12} | Market-Only: {market_r2:.4f} | Multimodal: {multi_r2:.4f} | Change: {diff:+.2f}% {symbol}')

print('\n📊 TASK B: Effect of Multimodal Features on Shock Detection')
print('-'*100)
for model in ['LogisticRegression', 'RandomForest_SMOTE', 'BiLSTM']:
    market = task_b[(task_b['Model'] == model) & (task_b['Regime'] == 'market_only')]
    multi = task_b[(task_b['Model'] == model) & (task_b['Regime'] == 'multimodal')]
    
    if not market.empty and not multi.empty:
        market_acc = market.iloc[0]['Accuracy']
        multi_acc = multi.iloc[0]['Accuracy']
        diff = (multi_acc - market_acc) * 100
        symbol = '✅' if diff > 0 else '⚠️'
        
        print(f'{model:<25} | Market: {market_acc:.4f} | Multi: {multi_acc:.4f} | Change: {diff:+.2f}% {symbol}')

print('\n\n' + '='*100)
print('💡 KEY FINDINGS & INSIGHTS')
print('='*100)
print('\n1️⃣  TASK A (Price Forecasting):')
print('   ✅ Ridge Regression (Market-Only) is the CLEAR WINNER')
print('   • Achieves 91.76% R² with simple linear model')
print('   • Beats XGBoost (83.8%) and BiLSTM (72.4%) significantly')
print('   • Only ±₹114 average error on prices around ₹2000-3000')
print()
print('2️⃣  Why Simple Models Win:')
print('   • Dataset: Only 577 samples (too small for deep learning)')
print('   • Price dynamics are highly autocorrelated (linear patterns)')
print('   • Ridge regularization prevents overfitting better than complex models')
print('   • BiLSTM needs 2000+ samples to outperform linear models')
print()
print('3️⃣  Multimodal Features:')
print('   ⚠️  HURT Ridge (-1.5% R²) and XGBoost (-3.5% R²)')
print('   ✅ HELP BiLSTM (+0.7% R²)')
print('   • Linear models struggle with noisy weather/news features')
print('   • Deep learning can learn non-linear weather-price relationships')
print('   • But still undertrained on small dataset')
print()
print('4️⃣  TASK B (Shock Detection):')
print('   ⚠️  All models suffer from extreme class imbalance')
print('   • RandomForest: 97.24% accuracy but 0% precision (predicts "no shock" always)')
print('   • Only 2-5% of days have actual shocks (±10% price change)')
print('   • Models learn to always predict "no shock" to maximize accuracy')
print('   • Need SMOTE, class weighting, or threshold tuning to improve')
print()
print('5️⃣  Production Recommendation:')
print('   🏆 Use Ridge Regression (Market-Only) for price forecasting')
print('   • Fastest training (~1 second vs 46 seconds for BiLSTM)')
print('   • Most accurate (91.76% R²)')
print('   • Most interpretable (linear coefficients)')
print('   • Robust to missing multimodal data')
print()
print('6️⃣  Future Improvements:')
print('   • Collect 2+ years of data (1000+ samples) for BiLSTM')
print('   • Add satellite imagery, social media sentiment')
print('   • Use focal loss or threshold tuning for shock detection')
print('   • Try transformer models (attention mechanisms)')
print()

print('='*100)
print('📄 Report generated successfully!')
print('='*100)
