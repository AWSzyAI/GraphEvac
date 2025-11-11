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
run: install
	@echo "🚀 Running evacuation sweep simulation..."
	@mkdir -p $(LOG_DIR) $(OUT_DIR)
	@$(PYTHON) $(MAIN) | tee $(LOG_DIR)/run.log
	@echo "✅ Simulation complete. Logs saved to $(LOG_DIR)/run.log"

# =========================================================
# 3. Visualization
# =========================================================
.PHONY: visualize
visualize: run
	@echo "🎨 Visualization artifacts are under $(OUT_DIR)/"

# =========================================================
# 4. Clean-up
# =========================================================
.PHONY: clean
clean:
	@echo "🧹 Cleaning logs and outputs..."
	rm -rf $(LOG_DIR)/* $(OUT_DIR)/*
	@echo "✅ Clean complete."

# =========================================================
# 5. Quick debug
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
# 6. Help message
# =========================================================
.PHONY: help
help:
	@echo ""
	@echo "Usage:"
	@echo "  make install     # Create venv and install deps"
	@echo "  make venv        # Create virtual env (.venv)"
	@echo "  make run         # Run the main simulation"
	@echo "  make visualize   # Run + generate plots & GIF"
	@echo "  make clean       # Remove output & log files"
	@echo "  make debug       # Print current configuration"
	@echo ""
