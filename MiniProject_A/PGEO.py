import arch
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import xgboost as xgb

from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATA ACQUISITION & PREPROCESSING
# ==========================================
ticker = 'PGEO.JK'
print(f"Narik data historis buat {ticker}...")
data = yf.download(ticker, start='2020-01-01', end='2023-12-31')

# Safety check struktur kolom yfinance (MultiIndex vs SingleIndex)
if isinstance(data.columns, pd.MultiIndex):
    prices = data[("Close", ticker)]
else:
    prices = data["Close"] if "Close" in data.columns else data.iloc[:, 0]

prices = pd.Series(prices).dropna()
returns = np.log(prices / prices.shift(1)).dropna()
returns_percentage = returns * 100

# ==========================================
# 2. GARCH(1,1) FITTING & VOLATILITY CLUSTERING
# ==========================================
print("Fitting model GARCH(1,1) dengan Student-t distribution...")
model = arch.arch_model(returns_percentage, vol="GARCH", p=1, q=1, dist="t")
results = model.fit(disp="off")

print(results.summary())

# Ambil Standardized Residuals (wajib i.i.d. untuk EVT)
std_residuals = results.std_resid
conditional_volatility = results.conditional_volatility

# Visualisasi 1: Volatilitas Bersyarat GARCH
plt.figure(figsize=(10, 4))
plt.plot(conditional_volatility, color="orange", lw=1.2)
plt.title(f"Conditional Volatility GARCH(1,1) - {ticker}")
plt.xlabel("Waktu")
plt.ylabel("Volatilitas Harian (%)")
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 3. EXTREME VALUE THEORY (EVT - GPD) ON RESIDUALS
# ==========================================
losses_resid = -std_residuals
threshold = np.percentile(losses_resid, 90) # Threshold persentil 90
excesses = losses_resid[losses_resid > threshold] - threshold 

c, loc, scale = stats.genpareto.fit(excesses)
print(f"\n--- Hasil EVT-GPD (Peaks-Over-Threshold) ---")
print(f"Parameter Shape (xi/c) GPD: {c:.4f} (Menunjukkan tingkat fat-tail)")
print(f"Parameter Scale (sigma) GPD: {scale:.4f}")

# ==========================================
# 4. MACHINE LEARNING (XGBOOST) — CLEANED
# ==========================================
print("\nMenyiapkan fitur ML tanpa data leakage...")
df_ml = pd.DataFrame({"Returns": returns_percentage})
df_ml["lag_1"] = df_ml["Returns"].shift(1)
df_ml["Rolling_Vol"] = df_ml["Returns"].rolling(window=5).std()

# Perbaikan: Rolling median digeser .shift(1) agar data hari t tidak bocor ke target
rolling_median_abs = df_ml["Returns"].abs().rolling(window=5).median().shift(1)
df_ml["Target"] = (df_ml["Returns"].abs() > rolling_median_abs).astype(int)

df_ml = df_ml.dropna()

X = df_ml[["lag_1", "Rolling_Vol"]]
y = df_ml["Target"]

# Split data secara time-series (tanpa shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

model_xgb = xgb.XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42
)
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi Model XGBoost: {acc:.4f}")

# ==========================================
# 5. MONTE CARLO SIMULATION (VaR & ES 95%)
# ==========================================
print("\nMenjalankan simulasi Monte Carlo...")
last_volt = conditional_volatility.iloc[-1]
last_mean = results.params['mu']
nu_fitted = results.params['nu']

num_simulations = 10000
simulated_returns = stats.t.rvs(
    df=nu_fitted, loc=last_mean, scale=last_volt, size=num_simulations
)

confidence_level = 0.95
var_95 = np.percentile(simulated_returns, (1 - confidence_level) * 100)
es_95 = simulated_returns[simulated_returns <= var_95].mean()

print(f"--- Hasil Risiko Pasar 1-Hari ke Depan ({ticker}) ---")
print(f"VaR 95%      : {-var_95:.4f}%")
print(f"Expected Shortfall (ES) 95% : {-es_95:.4f}%")

# Visualisasi 2: Distribusi Simulasi Monte Carlo, VaR, & Tail Risk (ES)
plt.figure(figsize=(9, 4))
plt.hist(
    simulated_returns,
    bins=50,
    density=True,
    alpha=0.6,
    color="skyblue",
    label="Monte Carlo Scenarios",
)
plt.axvline(
    var_95, color="red", linestyle="--", lw=2, label=f"VaR 95% ({abs(var_95):.2f}%)"
)
plt.axvspan(
    simulated_returns.min(),
    var_95,
    color="red",
    alpha=0.2,
    label="Tail Risk Region (ES)",
)
plt.title(f"Monte Carlo VaR & Expected Shortfall Simulation - {ticker}")
plt.xlabel("Simulated Return (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()