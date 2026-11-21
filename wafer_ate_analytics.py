import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

start_time = time.time()
np.random.seed(42)

# ==========================================
# 1. MULTI-WAFER LOT SIMULATION (3 FAB CHAMBERS)
# ==========================================
# Generate circular 300mm wafer die grid (-15 to +15)
base_coords = [(x, y) for x in range(-15, 16) for y in range(-15, 16) if x**2 + y**2 <= 15**2]
n_dies_per_wafer = len(base_coords)

chambers = ['Chamber_A (Baseline)', 'Chamber_B (Edge Drift)', 'Chamber_C (Scratch Fault)']
lot_dfs = []

for wafer_id, chamber in enumerate(chambers, start=1):
    w_df = pd.DataFrame(base_coords, columns=['Die_X', 'Die_Y'])
    w_df['Wafer_ID'] = f"WAFER_0{wafer_id}"
    w_df['Chamber'] = chamber
    w_df['Radius'] = np.sqrt(w_df['Die_X']**2 + w_df['Die_Y']**2)

    # Simulate 6 ATE electrical parameters
    w_df['Supply_Voltage_V']   = np.random.normal(1.20, 0.015, n_dies_per_wafer)
    w_df['Clock_Freq_MHz']     = np.random.normal(2400, 40, n_dies_per_wafer)
    w_df['Leakage_Current_mA'] = np.random.normal(14.5, 2.0, n_dies_per_wafer)
    w_df['Access_Time_ns']     = np.random.normal(8.20, 0.35, n_dies_per_wafer)

    # Inject chamber-specific hardware faults
    if 'Chamber_B' in chamber:
        # Edge-ring plasma etch drift (radius > 13.0)
        edge_mask = w_df['Radius'] > 13.0
        w_df.loc[edge_mask, 'Leakage_Current_mA'] += np.random.normal(11.5, 1.8, edge_mask.sum())
    elif 'Chamber_C' in chamber:
        # Linear robotic handling scratch along diagonal (|X - Y| <= 1 and X > -5)
        scratch_mask = (np.abs(w_df['Die_X'] - w_df['Die_Y']) <= 1) & (w_df['Die_X'] > -6)
        w_df.loc[scratch_mask, 'Leakage_Current_mA'] += np.random.normal(14.0, 2.2, scratch_mask.sum())

    # Add 2 correlated redundant tests (candidates for ATE test-time reduction)
    w_df['Standby_Power_mW'] = (w_df['Leakage_Current_mA'] * 1.25) + np.random.normal(0, 0.18, n_dies_per_wafer)
    w_df['Thermal_Drift_C']  = (w_df['Clock_Freq_MHz'] * 0.03) + np.random.normal(0, 0.35, n_dies_per_wafer)

    # Hard Binning: Pass (1) if Leakage <= 22.0 mA and Voltage within [1.14V, 1.26V]
    w_df['Pass_Fail'] = np.where(
        (w_df['Leakage_Current_mA'] <= 22.0) &
        (w_df['Supply_Voltage_V'].between(1.14, 1.26)), 1, 0
    )
    lot_dfs.append(w_df)

df_lot = pd.concat(lot_dfs, ignore_index=True)

print("=== MULTI-WAFER LOT YIELD SUMMARY ===")
for chamber, grp in df_lot.groupby('Chamber'):
    print(f"{chamber:<28} | Yield: {grp['Pass_Fail'].mean()*100:.2f}% | Mean Leakage: {grp['Leakage_Current_mA'].mean():.2f} mA")

# ==========================================
# 2. ONE-WAY ANOVA ACROSS FABRICATION CHAMBERS
# ==========================================
g_a = df_lot[df_lot['Chamber'] == chambers[0]]['Leakage_Current_mA']
g_b = df_lot[df_lot['Chamber'] == chambers[1]]['Leakage_Current_mA']
g_c = df_lot[df_lot['Chamber'] == chambers[2]]['Leakage_Current_mA']

f_stat, p_val = stats.f_oneway(g_a, g_b, g_c)
print(f"\n=== ONE-WAY ANOVA (CHAMBER LEAKAGE VARIANCE) ===")
print(f"F-Statistic: {f_stat:.2f} | P-Value: {p_val:.4e}")
if p_val < 0.05:
    print("Conclusion: Statistically significant chamber drift detected (p < 0.05). Maintenance required.\n")

# ==========================================
# 3. PROCESS CAPABILITY INDEX (Cpk) AUDIT
# ==========================================
def calculate_cpk(series, lsl, usl):
    mean = series.mean()
    std = series.std()
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    return min(cpu, cpl)

print("=== PROCESS CAPABILITY INDEX (Cpk) AUDIT (TARGET >= 1.33) ===")
specs = {
    'Supply_Voltage_V':   (1.14, 1.26),
    'Clock_Freq_MHz':     (2250, 2550),
    'Leakage_Current_mA': (5.00, 22.0),
    'Access_Time_ns':     (6.80, 9.60)
}
for param, (lsl, usl) in specs.items():
    cpk = calculate_cpk(df_lot[param], lsl, usl)
    status = "CAPABLE (Six Sigma)" if cpk >= 1.33 else "AT RISK (Action Needed)"
    print(f"{param:<20} | LSL: {lsl:<5} | USL: {usl:<5} | Cpk: {cpk:.2f} -> {status}")

# ==========================================
# 4. ATE TEST-TIME & COST REDUCTION ENGINE
# ==========================================
test_cols = ['Supply_Voltage_V', 'Clock_Freq_MHz', 'Leakage_Current_mA', 'Access_Time_ns', 'Standby_Power_mW', 'Thermal_Drift_C']
test_times_ms = {col: 120 for col in test_cols}  # Assume 120 ms ATE probe time per test per die

corr_matrix = df_lot[test_cols].corr(method='pearson')
redundant_tests = set()

for i in range(len(test_cols)):
    for j in range(i + 1, len(test_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            redundant_tests.add(test_cols[j])

total_time_before = sum(test_times_ms.values())
time_saved_per_die = sum(test_times_ms[t] for t in redundant_tests)
pct_reduction = (time_saved_per_die / total_time_before) * 100

# Financial savings across 10,000 wafers/year at $0.08/sec ATE machine rate
annual_dies = 10000 * n_dies_per_wafer
annual_hours_saved = (annual_dies * (time_saved_per_die / 1000.0)) / 3600.0
annual_savings_usd = annual_hours_saved * 3600 * 0.08

print(f"\n=== ATE TEST-TIME & COST REDUCTION SUMMARY ===")
print(f"Redundant Tests Eliminated (|r| > 0.95): {list(redundant_tests)}")
print(f"Test Time Reduction: {pct_reduction:.1f}% ({time_saved_per_die} ms saved per microchip)")
print(f"Projected Annual Fab Savings (10k wafers): {annual_hours_saved:,.1f} ATE hours (~${annual_savings_usd:,.0f}/year)\n")

# ==========================================
# 5. SPATIAL DEFECT SIGNATURE CLASSIFIER (ON FAILING DIES)
# ==========================================
# Filter to defective dies only (Pass_Fail == 0) to diagnose spatial failure root causes
defective_dies = df_lot[df_lot['Pass_Fail'] == 0].copy()
defective_dies['Diag_Offset'] = np.abs(defective_dies['Die_X'] - defective_dies['Die_Y'])

features = ['Die_X', 'Die_Y', 'Radius', 'Diag_Offset', 'Leakage_Current_mA']
X_ml = defective_dies[features]
y_ml = defective_dies['Chamber']

X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.25, random_state=42, stratify=y_ml
)

rf_spatial = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_spatial.fit(X_train, y_train)

print("=== SPATIAL DEFECT SIGNATURE CLASSIFIER (DEFECTIVE DIES) ===")
print(f"Root-Cause Spatial Pattern Classification Accuracy: {rf_spatial.score(X_test, y_test)*100:.2f}%")
print(f"Total Pipeline Execution Time: {time.time() - start_time:.2f}s\n")

# ==========================================
# 6. 3-PANEL EXECUTIVE TEST DASHBOARD
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Wafer Map showing Chamber C (Diagonal Scratch Defect)
w3 = df_lot[df_lot['Chamber'] == chambers[2]]
p3, f3 = w3[w3['Pass_Fail'] == 1], w3[w3['Pass_Fail'] == 0]
axes[0].scatter(p3['Die_X'], p3['Die_Y'], color='#2ecc71', marker='s', s=38, label='Pass')
axes[0].scatter(f3['Die_X'], f3['Die_Y'], color='#e74c3c', marker='s', s=38, label='Fail (Scratch)')
axes[0].set_title(f"Wafer 03 Spatial Map (Yield: {w3['Pass_Fail'].mean()*100:.1f}%)")
axes[0].set_xlabel("Die X Coordinate")
axes[0].set_ylabel("Die Y Coordinate")
axes[0].set_aspect('equal')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.2)

# Panel 2: Pearson Correlation Heatmap for Test Reduction
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar=False)
axes[1].set_title("ATE Parametric Correlation Matrix")
axes[1].tick_params(axis='x', rotation=35)

# Panel 3: Chamber Leakage Comparison (One-Way ANOVA - Warning-Free)
sns.boxplot(
    data=df_lot, x='Chamber', y='Leakage_Current_mA',
    hue='Chamber', palette=['#2ecc71', '#f39c12', '#e74c3c'],
    legend=False, ax=axes[2]
)
axes[2].axhline(22.0, color='black', linestyle='--', label='Upper Spec Limit (22 mA)')
axes[2].set_title(f"Chamber Drift ANOVA (F={f_stat:.1f}, p={p_val:.1e})")
axes[2].set_xlabel("")
axes[2].set_ylabel("Leakage Current (mA)")
axes[2].tick_params(axis='x', rotation=15)
axes[2].legend(loc='upper left')

plt.tight_layout()
output_img = "ate_yield_dashboard.png"
plt.savefig(output_img, dpi=300)
print(f"Saved 3-panel ATE dashboard to: {output_img}")
plt.show()
