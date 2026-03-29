import sys, os
print(f"START Python {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"FILES: {os.listdir('.')[:10]}", flush=True)

# Import and run the engine
import quant_engine
print("QUANT ENGINE v5.0 STARTING", flush=True)
quant_engine.run()
