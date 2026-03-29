"""Startup wrapper — diagnose import failures before launching main engine."""
import sys
print(f"Python {sys.version}", flush=True)

# Test critical imports
for mod in ["requests", "numpy", "websocket", "sklearn", "pandas"]:
    try:
        __import__(mod)
        print(f"  {mod} OK", flush=True)
    except Exception as e:
        print(f"  {mod} FAILED: {e}", flush=True)

# Test lightgbm specifically
try:
    import lightgbm
    print(f"  lightgbm OK: {lightgbm.__version__}", flush=True)
except Exception as e:
    print(f"  lightgbm FAILED: {e}", flush=True)

# Test pickle load
try:
    import pickle
    with open("model_v4_profitable.pkl", "rb") as f:
        m = pickle.load(f)
    print(f"  model_v4_profitable.pkl OK: {len(m.get('features', []))} features", flush=True)
except Exception as e:
    print(f"  model load FAILED: {e}", flush=True)

print("Launching quant_engine.py...", flush=True)

# Import and run
import quant_engine
