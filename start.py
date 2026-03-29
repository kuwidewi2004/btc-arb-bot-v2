import sys, os
print(f"START Python {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"FILES: {os.listdir('.')[:10]}", flush=True)

# Skip lightgbm entirely — just launch the engine
os.environ["SKIP_LIGHTGBM"] = "1"
import quant_engine
