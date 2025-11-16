# =========================================================
# GraphEvac Project Makefile
# Author: 时子延
# Description: One-command automation for simulation, logging, and visualization
# =========================================================

SRC_DIR := src
LOG_DIR := log
OUT_DIR := output

# Virtual environment (PEP 668 safe install)
VENV_DIR := .venv
PYTHON   := $(VENV_DIR)/bin/python
PIP      := $(VENV_DIR)/bin/pip
PYTHON_SYS := python3
PYTHONPATH_SRC := $(shell pwd)/src

BATCH_FLOORS ?= $(shell PYTHONPATH=$(PYTHONPATH_SRC) $(PYTHON_SYS) -c "from configs import BATCH_CONFIG; print(BATCH_CONFIG.get('floors', '1-18'))")
BATCH_LAYOUTS ?= $(shell PYTHONPATH=$(PYTHONPATH_SRC) $(PYTHON_SYS) -c "from configs import BATCH_CONFIG; print(BATCH_CONFIG.get('layouts', 'BASELINE,T,L'))")
BATCH_OCC ?= $(shell PYTHONPATH=$(PYTHONPATH_SRC) $(PYTHON_SYS) -c "from configs import BATCH_CONFIG; print(BATCH_CONFIG.get('occ', '5-10'))")
BATCH_RESP ?= $(shell PYTHONPATH=$(PYTHONPATH_SRC) $(PYTHON_SYS) -c "from configs import BATCH_CONFIG; print(BATCH_CONFIG.get('resp', '1-10'))")

MAIN := $(SRC_DIR)/main.py
VIZ  := $(SRC_DIR)/viz.py
REQ  := requirements.txt

# =========================================================
# Default target
# =========================================================
.PHONY: all
all: run visualize

# =========================================================
# 1. Environment Setup
# =========================================================
.PHONY: venv
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	@echo "🐍 Creating virtual environment at $(VENV_DIR) ..."
	@python3 -m venv $(VENV_DIR)
	@$(PYTHON) -m pip install --upgrade pip

.PHONY: install
install: venv
	@echo "📦 Installing dependencies into $(VENV_DIR) ..."
	@$(PIP) install -r $(REQ)

# =========================================================
# 2. Run Simulation
# =========================================================
.PHONY: run
run: 
	@echo "🚀 Running evacuation sweep simulation..."
	@mkdir -p $(LOG_DIR) $(OUT_DIR)
	@OUTPUT_ROOT=$(OUT_DIR) $(PYTHON) $(MAIN)
	@echo "✅ Simulation complete. Check output/*/run.log for detailed logs"

# =========================================================
# 3. Visualization
# =========================================================
.PHONY: visualize
visualize: install
	@echo "🎨 Rebuilding visualization assets from the latest run..."
	@OUTPUT_ROOT=$(OUT_DIR) $(PYTHON) $(SRC_DIR)/render_viz.py --output-root $(OUT_DIR)

# =========================================================
# 4. Clean-up
# =========================================================
.PHONY: clean
clean:
	@echo "🧹 Cleaning logs and outputs..."
	rm -rf $(LOG_DIR)/* $(OUT_DIR)/*
	@echo "✅ Clean complete."

# =========================================================
# 5. Batch sweeps
# =========================================================
.PHONY: batch
batch: 
	@echo "📊 Running batch sweeps and exporting CSV..."
	@mkdir -p $(OUT_DIR)
	@$(PYTHON) $(SRC_DIR)/batch.py --floors "$(BATCH_FLOORS)" --layouts "$(BATCH_LAYOUTS)" --occ "$(BATCH_OCC)" --resp "$(BATCH_RESP)" --max-exit-combos "$${MAX_EXIT_COMBOS:-}" --out "$(OUT_DIR)/batch_results.csv"
	@echo "✅ CSV saved to $(OUT_DIR)/batch_results.csv"

# =========================================================
# 6. Quick debug
# =========================================================
.PHONY: debug
debug: install
	@echo "🔍 Debug mode: print SIM_CONFIG"
	@$(PYTHON) - <<-'PY'
	from pprint import pprint
	from configs import SIM_CONFIG
	pprint(SIM_CONFIG)
	PY

# =========================================================
# 7. Help message
# =========================================================
.PHONY: help
help:
	@echo ""
	@echo "Usage:"
	@echo "  make install     # Create venv and install deps"
	@echo "  make venv        # Create virtual env (.venv)"
	@echo "  make run         # Run the main simulation"
	@echo "  make visualize   # Run + generate plots & GIF"
	@echo "  make batch       # Sweep params and export CSV"
	@echo "  make clean       # Remove output & log files"
	@echo "  make debug       # Print current configuration"
	@echo ""
