# -*- coding: utf-8 -*-
"""
Execute crowd_counting_density_maps.ipynb and save outputs inplace.
"""
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = "crowd_counting_density_maps.ipynb"
print(f"[INFO] Reading {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=1200, kernel_name="python3")

print("[INFO] Starting execution of all notebook cells...")
t0 = time.time()
try:
    ep.preprocess(nb, {"metadata": {"path": "."}})
    elapsed = time.time() - t0
    print(f"[SUCCESS] Notebook executed successfully in {elapsed:.1f} seconds!")
except Exception as e:
    print(f"[ERROR] Execution failed: {e}")
    raise

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"[INFO] Successfully saved executed notebook with all outputs to {notebook_path}!")
