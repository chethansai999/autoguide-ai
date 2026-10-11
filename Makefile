# AutoGuide AI - common commands. Windows without make: run the commands after the colon lines.
PY ?= python

.PHONY: setup test offline-test data chunk index status api ui synth train-nlu eval-nlu eval-dev docker

setup:            ## install runtime deps and create .env
	$(PY) -m pip install -r requirements.txt
	@test -f .env || cp .env.example .env

test:             ## full offline test-suite (no keys, no downloads)
	$(PY) -m pytest -q

status:           ## what exists at each pipeline stage
	$(PY) ingest.py status

data:             ## load the NHTSA ratings CSV into the database
	$(PY) ingest.py load-safercar

chunk:            ## STRATEGY=fixed|structure|both
	$(PY) ingest.py chunk --strategy $(or $(STRATEGY),fixed)

index:            ## STRATEGY=fixed|structure|both
	$(PY) ingest.py index --strategy $(or $(STRATEGY),fixed)

api:              ## FastAPI on :8000 (docs at /docs)
	uvicorn api.main:app --reload --port 8000

ui:               ## Streamlit UI on :8501
	streamlit run app.py

synth:            ## synthetic NLU training data from the vehicles table
	$(PY) -m finetune.synth_data --n 8000

train-nlu:        ## fine-tune DistilBERT intent + slot models (needs requirements-finetune.txt)
	$(PY) -m finetune.train_nlu

eval-nlu:         ## BACKEND=rules|hf|llm  SET=synthetic|realistic
	$(PY) -m finetune.eval_nlu --backend $(or $(BACKEND),rules) --set $(or $(SET),synthetic)

eval-dev:         ## baselines B0-B3 on the golden DEV split
	$(PY) -m evaluation.run_evaluation --baseline all --split dev

docker:           ## Postgres + API + UI
	docker compose up --build
