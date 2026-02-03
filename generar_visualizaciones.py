# -*- coding: utf-8 -*-
"""
GENERADOR DE VISUALIZACIONES PARA PORTFOLIO
Genera 4 imágenes separadas ideales para mostrar en portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print('='*80)
print('📊 GENERADOR DE VISUALIZACIONES PARA PORTFOLIO')
print('='*80)

# ============================================================================
# CARGAR Y PREPARAR DATOS
# ============================================================================

print('\n1️⃣ Cargando datos...')
df = pd.read_csv('data/superstore_master_clean.csv')
df_enhanced = df.copy()
df_enhanced['Order Date'] = pd.to_datetime(df_enhanced['Order Date'])

# Feature Engineering (mismo que el modelo mejorado)
print('2️⃣ Aplicando feature engineering...')

# Agregaciones por cliente
customer_stats = df_enhanced.groupby('Customer ID').agg({
    'Sales': ['mean', 'std', 'sum', 'count'],
    'Profit': ['mean', 'sum'],
    'Discount': ['mean', 'max'],
    'Shipping_Days': 'mean'
}).round(2)
customer_stats.columns = [
    'Customer_Avg_Sale', 'Customer_Sale_Std', 'Customer_Total_Sales',
    'Customer_Order_Count', 'Customer_Avg_Profit', 'Customer_Total_Profit',
    'Customer_Avg_Discount', 'Customer_Max_Discount', 'Customer_Avg_Shipping'
]
df_enhanced = df_enhanced.merge(customer_stats, on='Customer ID', how='left')

# Agregaciones por producto
product_stats = df_enhanced.groupby('Product ID').agg({
    'Sales': ['mean', 'std', 'sum', 'count'],
    'Profit': ['mean', 'sum'],
    'Discount': 'mean',
    'Shipping_Days': 'mean'
}).round(2)
product_stats.columns = [
    'Product_Avg_Sale', 'Product_Sale_Std', 'Product_Total_Sales',
    'Product_Sale_Count', 'Product_Avg_Profit', 'Product_Total_Profit',
    'Product_Avg_Discount', 'Product_Avg_Shipping'
]
df_enhanced = df_enhanced.merge(product_stats, on='Product ID', how='left')

# Agregaciones por categoría y región
category_stats = df_enhanced.groupby('Category').agg({'Sales': 'mean', 'Profit': 'mean'}).round(2)
category_stats.columns = ['Category_Avg_Sale', 'Category_Avg_Profit']
df_enhanced = df_enhanced.merge(category_stats, on='Category', how='left')

region_stats = df_enhanced.groupby('Region').agg({'Sales': 'mean', 'Profit': 'mean'}).round(2)
region_stats.columns = ['Region_Avg_Sale', 'Region_Avg_Profit']
df_enhanced = df_enhanced.merge(region_stats, on='Region', how='left')

# Ratios
df_enhanced['Sale_vs_CustomerAvg'] = df_enhanced['Sales'] / (df_enhanced['Customer_Avg_Sale'] + 1)
df_enhanced['Sale_vs_ProductAvg'] = df_enhanced['Sales'] / (df_enhanced['Product_Avg_Sale'] + 1)
df_enhanced['Sale_vs_CategoryAvg'] = df_enhanced['Sales'] / (df_enhanced['Category_Avg_Sale'] + 1)
df_enhanced['Discount_vs_CustomerAvg'] = df_enhanced['Discount'] / (df_enhanced['Customer_Avg_Discount'] + 0.01)
df_enhanced['Discount_vs_ProductAvg'] = df_enhanced['Discount'] / (df_enhanced['Product_Avg_Discount'] + 0.01)
df_enhanced['Profit_vs_CustomerAvg'] = df_enhanced['Profit'] / (df_enhanced['Customer_Avg_Profit'] + 1)

# Interacciones
df_enhanced['Discount_x_CustomerAvgSale'] = df_enhanced['Discount'] * df_enhanced['Customer_Avg_Sale']
df_enhanced['Discount_x_ProductAvgSale'] = df_enhanced['Discount'] * df_enhanced['Product_Avg_Sale']

# Temporales
df_enhanced['Is_Weekend'] = df_enhanced['Order Date'].dt.dayofweek.isin([5, 6]).astype(int)
df_enhanced['Is_MonthEnd'] = (df_enhanced['Order Date'].dt.day > 25).astype(int)
df_enhanced['Days_Since_Year_Start'] = df_enhanced['Order Date'].dt.dayofyear

# ============================================================================
# PREPARAR MODELO
# ============================================================================

print('3️⃣ Entrenando modelo...')

# Codificar categóricas
categorical = ['Segment', 'Ship Mode', 'Category', 'Sub-Category', 'Region']
for col in categorical:
    le = LabelEncoder()
    df_enhanced[f'{col}_enc'] = le.fit_transform(df_enhanced[col])

# Features
features_v2 = [
    'Segment_enc', 'Ship Mode_enc', 'Category_enc', 'Sub-Category_enc', 'Region_enc',
    'Discount', 'Shipping_Days', 'Month', 'Quarter', 'Year',
    'Is_Weekend', 'Is_MonthEnd', 'Days_Since_Year_Start',
    'Customer_Avg_Sale', 'Customer_Sale_Std', 'Customer_Total_Sales',
    'Customer_Order_Count', 'Customer_Avg_Profit', 'Customer_Total_Profit',
    'Customer_Avg_Discount', 'Customer_Max_Discount', 'Customer_Avg_Shipping',
    'Product_Avg_Sale', 'Product_Sale_Std', 'Product_Total_Sales',
    'Product_Sale_Count', 'Product_Avg_Profit', 'Product_Total_Profit',
    'Product_Avg_Discount', 'Product_Avg_Shipping',
    'Category_Avg_Sale', 'Category_Avg_Profit',
    'Region_Avg_Sale', 'Region_Avg_Profit',
    'Sale_vs_CustomerAvg', 'Sale_vs_ProductAvg', 'Sale_vs_CategoryAvg',
    'Discount_vs_CustomerAvg', 'Discount_vs_ProductAvg',
    'Profit_vs_CustomerAvg',
    'Discount_x_CustomerAvgSale', 'Discount_x_ProductAvgSale'
]

X = df_enhanced[features_v2].fillna(0)
y = df_enhanced['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar los 3 modelos
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)

rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10,
                            min_samples_leaf=4, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

gb = GradientBoostingRegressor(n_estimators=200, max_depth=10, learning_rate=0.05,
                                subsample=0.8, min_samples_split=10, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mae = mean_absolute_error(y_test, gb_pred)

# Seleccionar mejor modelo
models = [
    ('Linear Regression', lr_r2, lr_rmse, lr_mae, lr_pred, lr),
    ('Random Forest', rf_r2, rf_rmse, rf_mae, rf_pred, rf),
    ('Gradient Boosting', gb_r2, gb_rmse, gb_mae, gb_pred, gb)
]
models.sort(key=lambda x: x[1], reverse=True)

best_model_name, best_r2, best_rmse, best_mae, best_predictions, best_model = models[0]

print(f'   Mejor modelo: {best_model_name}')
print(f'   R² = {best_r2:.4f}')
print(f'   RMSE = ${best_rmse:.2f}')

errors = best_predictions - y_test

# ============================================================================
# GENERAR LAS 4 VISUALIZACIONES
# ============================================================================

print('\n' + '='*80)
print('🎨 GENERANDO VISUALIZACIONES')
print('='*80)

# ---------------------------------------------------------------------------
# IMAGEN 1: PREDICCIÓN VS REAL
# ---------------------------------------------------------------------------
print('\n📈 Imagen 1: Predicción vs Real...')

plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_predictions, alpha=0.6, s=50, color='#2E86AB', 
            edgecolor='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=3, label='Predicción Perfecta', alpha=0.8)

plt.xlabel('Ventas Reales ($)', fontweight='bold', fontsize=14)
plt.ylabel('Ventas Predichas ($)', fontweight='bold', fontsize=14)
plt.title(f'Predicción vs Realidad\n{best_model_name} - R² = {best_r2:.4f} ({best_r2*100:.1f}%)',
         fontweight='bold', fontsize=16, pad=20)

# Métricas en el gráfico
textstr = f'R² = {best_r2:.4f}\nRMSE = ${best_rmse:,.2f}\nMAE = ${best_mae:,.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/1_prediccion_vs_real.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('   ✓ Guardada: img/1_prediccion_vs_real.png')

# ---------------------------------------------------------------------------
# IMAGEN 2: FEATURE IMPORTANCE (TOP 10)
# ---------------------------------------------------------------------------
print('\n🔍 Imagen 2: Feature Importance (Top 10)...')

# Usar RF para feature importance (siempre disponible)
feat_imp = pd.DataFrame({
    'Feature': features_v2,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
top_10_feat = feat_imp.head(10)
feat_names = [f.replace('_enc', '').replace('_', ' ') for f in top_10_feat['Feature']]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
y_pos = np.arange(len(feat_names))
bars = plt.barh(y_pos, top_10_feat['Importance'].values, color=colors,
                edgecolor='black', linewidth=1.5)

plt.yticks(y_pos, feat_names, fontsize=12)
plt.gca().invert_yaxis()
plt.xlabel('Importancia', fontweight='bold', fontsize=14)
plt.title('Top 10 Features Más Importantes\nRandom Forest',
         fontweight='bold', fontsize=16, pad=20)

for i, v in enumerate(top_10_feat['Importance'].values):
    plt.text(v, i, f'  {v:.4f} ({v*100:.1f}%)', va='center', fontsize=11, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('img/2_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('   ✓ Guardada: img/2_feature_importance.png')

# ---------------------------------------------------------------------------
# IMAGEN 3: DISTRIBUCIÓN DE ERRORES
# ---------------------------------------------------------------------------
print('\n📊 Imagen 3: Distribución de Errores...')

plt.figure(figsize=(12, 8))
n, bins, patches = plt.hist(errors, bins=60, color='#6A4C93', alpha=0.7,
                             edgecolor='black', linewidth=1.2)

# Colorear barras según error
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if abs(bin_center) < 50:
        patch.set_facecolor('#51CF66')  # Verde
    elif abs(bin_center) < 150:
        patch.set_facecolor('#FFD93D')  # Amarillo
    else:
        patch.set_facecolor('#FF6B6B')  # Rojo

plt.axvline(0, color='red', linestyle='--', linewidth=3, label='Error = 0 (Perfecto)', alpha=0.8)
plt.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2.5,
           label=f'Media = ${errors.mean():.2f}', alpha=0.8)
plt.axvline(np.median(errors), color='green', linestyle='--', linewidth=2.5,
           label=f'Mediana = ${np.median(errors):.2f}', alpha=0.8)

plt.xlabel('Error de Predicción ($)', fontweight='bold', fontsize=14)
plt.ylabel('Frecuencia', fontweight='bold', fontsize=14)
plt.title(f'Distribución de Errores de Predicción\nRMSE = ${best_rmse:,.2f} | MAE = ${best_mae:,.2f}',
         fontweight='bold', fontsize=16, pad=20)

textstr = f'Estadísticas:\nMedia: ${errors.mean():.2f}\nMediana: ${np.median(errors):.2f}\nStd: ${errors.std():.2f}\nRMSE: ${best_rmse:.2f}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('img/3_distribucion_errores.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('   ✓ Guardada: img/3_distribucion_errores.png')

# ---------------------------------------------------------------------------
# IMAGEN 4: COMPARACIÓN V1 VS V2
# ---------------------------------------------------------------------------
print('\n📉 Imagen 4: Comparación V1 vs V2...')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Comparación: Modelo Original vs Mejorado', fontweight='bold', fontsize=18, y=0.98)

# Subgráfico 1: R² Score
versions = ['V1\nOriginal', 'V2\nMejorado']
r2_values = [0.1059, best_r2]
colors_v = ['#FF6B6B', '#51CF66']

bars1 = ax1.bar(versions, r2_values, color=colors_v, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('R² Score', fontweight='bold', fontsize=14)
ax1.set_title('R² Score: Capacidad Predictiva', fontweight='bold', fontsize=14)
ax1.set_ylim([0, 1])
ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excelente (>0.8)')
ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Aceptable (>0.5)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars1, r2_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}\n({score*100:.1f}%)', ha='center', va='bottom',
            fontweight='bold', fontsize=13)

improvement_pct = ((best_r2 - 0.1059) / 0.1059) * 100
ax1.text(0.5, 0.5, f'↑ +{improvement_pct:.0f}%\nde mejora',
        ha='center', va='center', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        transform=ax1.transAxes)

# Subgráfico 2: RMSE
rmse_values = [773.07, best_rmse]
colors_rmse = ['#FF6B6B', '#51CF66']

bars2 = ax2.bar(versions, rmse_values, color=colors_rmse, edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('RMSE ($)', fontweight='bold', fontsize=14)
ax2.set_title('RMSE: Error de Predicción', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

for bar, rmse in zip(bars2, rmse_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${rmse:.2f}', ha='center', va='bottom',
            fontweight='bold', fontsize=13)

reduction_pct = ((773.07 - best_rmse) / 773.07) * 100
ax2.text(0.5, 0.5, f'↓ -{reduction_pct:.0f}%\nde error',
        ha='center', va='center', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
        transform=ax2.transAxes)

plt.tight_layout()
plt.savefig('img/4_comparacion_v1_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('   ✓ Guardada: img/4_comparacion_v1_v2.png')

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print('\n' + '='*80)
print('✅ VISUALIZACIONES COMPLETADAS')
print('='*80)
print('\n📁 Imágenes generadas para portfolio:')
print('   1. 1_prediccion_vs_real.png       - Scatter: Predicciones vs Realidad')
print('   2. 2_feature_importance.png       - Barras: Top 10 features')
print('   3. 3_distribucion_errores.png     - Histograma: Distribución de errores')
print('   4. 4_comparacion_v1_v2.png        - Comparativa: Antes vs Después')
print('\n💡 Estas 4 imágenes son ideales para:')
print('   • Portfolio web')
print('   • Presentaciones')
print('   • README de GitHub')
print('   • LinkedIn')
print('\n🎯 Modelo usado: {}'.format(best_model_name))
print(f'   R² = {best_r2:.4f} ({best_r2*100:.1f}%)')
print(f'   RMSE = ${best_rmse:,.2f}')
print(f'   MAE = ${best_mae:,.2f}')
print('\n' + '='*80)