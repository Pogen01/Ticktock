PY ?= python3

.PHONY: install data train test serve all

install:
	$(PY) -m pip install -r requirements.txt

train:            ## build dataset, fit + score all models, save to models/ and reports/
	$(PY) scripts/run_pipeline.py

data: train

test:
	$(PY) -m pytest -q

serve:            ## demo at http://127.0.0.1:8000
	$(PY) -m uvicorn app.server:app --port 8000

all: install train test
