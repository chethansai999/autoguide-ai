# Deploy AutoGuide AI on Streamlit Cloud (free, permanent link)

This puts the app on a stable `*.streamlit.app` address that anyone can open
from a phone or laptop. It runs entirely on **your** GitHub + Streamlit
accounts. Three steps, about 10 minutes (most of it waiting for the first
build).

Already done for you in the code:

- The app configures itself from Streamlit **Secrets** (no `.env` needed).
- The vehicle database is **not** in git (it is gitignored), so the app
  automatically loads all 17,313 NHTSA vehicle records from the committed
  `data/structured/Safercar_data.csv` the first time it starts.

---

## Step 1 — Commit and push the code (your computer, terminal)

From the folder where you cloned this repo:

```bash
cd autoguide-ai
git add app.py core/llm.py core/config.py api/readiness.py .env.example \
        .streamlit/config.toml DEPLOY_STREAMLIT.md start-all.sh start-public.sh
git status
git commit -m "Complete UI, add Ollama provider, prep Streamlit Cloud deploy"
git push origin main
```

At the `git status` step, check the list of files about to be committed:
it must **not** contain `.env` or anything in `logs/`. Your `.env` file holds
API keys — it is gitignored on purpose and must never be committed or pushed.

What those files are:

- `app.py` — the finished 5-view UI, plus the cloud startup bootstrap
- `core/llm.py`, `core/config.py` — the local Llama (Ollama) provider + settings
- `api/readiness.py` — backend status reporting used by the sidebar
- `.env.example` — documents every setting (no real keys in it)
- `.streamlit/config.toml` — the app's color theme
- `start-all.sh`, `start-public.sh` — scripts to run the app locally

## Step 2 — Create the app on Streamlit Cloud

1. Go to **https://share.streamlit.io** and click **Continue with GitHub**
   (sign in with the GitHub account that owns this repo).
2. Click **New app**.
3. Fill in:
   - **Repository:** `chethansai999/autoguide-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings**:
   - **Python version:** pick **3.11** or **3.12** if asked.
   - **Secrets:** paste the block from Step 3 below.
5. Click **Deploy**. The first build takes several minutes (it installs the
   Python packages). When it finishes you get your permanent link, like
   `https://autoguide-ai-xxxx.streamlit.app`.

## Step 3 — Secrets to paste (Advanced settings → Secrets)

```toml
LLM_PROVIDER = "mock"
EMBEDDING_BACKEND = "hashing"
RERANKER_BACKEND = "none"
ENABLE_LIVE_RECALLS_API = "true"
```

Why these values:

- `LLM_PROVIDER = "mock"` — Streamlit Cloud cannot run the local Llama
  model (Ollama runs on your own machine only). Mock mode still returns
  real NHTSA data with real citations; the written answers are simple
  extractive placeholders. See the upgrade below for real written answers.
- `EMBEDDING_BACKEND = "hashing"` and `RERANKER_BACKEND = "none"` — the
  free tier has limited memory; the full AI search models are heavy, and
  they are not needed for vehicle lookups. (They only matter for the
  article library, which is empty — see "What doesn't work" below.)
- `ENABLE_LIVE_RECALLS_API = "true"` — lets the Recalls view fetch live
  recall data from NHTSA.

You can also change secrets later: open the app → **Manage app** (bottom
right) → **⋮ menu → Settings → Secrets**, edit, save, then reboot the app.

### Optional upgrade: real written answers (still free)

The mock answers show the data but read like placeholders. For real
AI-written answers, get a free API key and change two lines in Secrets:

- **Google Gemini (recommended, free tier):** get a key at
  https://aistudio.google.com (sign in → *Get API key*), then set:

  ```toml
  LLM_PROVIDER = "gemini"
  GEMINI_API_KEY = "paste-your-key-here"
  ```

- **OpenAI (paid):** get a key at https://platform.openai.com, then set:

  ```toml
  LLM_PROVIDER = "openai"
  OPENAI_API_KEY = "paste-your-key-here"
  ```

Save the secrets and reboot the app (Manage app → ⋮ → Reboot app).
Keys belong **only** in the Secrets box — never in code or in git.

---

## What works on Streamlit Cloud

- All 5 views: Safety Assistant (chat), Vehicle Lookup, Vehicle Recalls,
  Safety Features, NHTSA Resources
- Vehicle safety-rating lookups from the full 17,313-record NHTSA data,
  with citations
- Clarifying questions with clickable options (e.g. picking a model year)
- Live NHTSA recalls, feedback buttons, Export Chat

## What doesn't work there (and why)

- **"How does X work?" concept questions** (automatic emergency braking,
  airbags, …): the app answers these from an article library that was
  never built — the team's source documents (IIHS PDFs) are not in this
  repo. The app says so honestly instead of making something up. To fix
  this later, the team adds the documents and builds the index
  (`python ingest.py add-docs`, `chunk`, `index`).
- **Local Llama answers:** Ollama runs on a local machine only. On cloud,
  use mock mode or the Gemini/OpenAI upgrade above.
- **Chat memory:** the cloud database is temporary storage. If the app is
  redeployed or rebooted, past conversations are lost — but the vehicle
  data is **not**: it reloads automatically from the CSV in the repo at
  every startup. Free-tier apps also sleep after inactivity; the first
  visit afterwards takes a few seconds to wake them.

## Troubleshooting

- **App shows an error on first boot:** open **Manage app** → ⋮ →
  **Reboot app**. First boots can race the one-time vehicle-data load.
- **Still broken:** in **Manage app**, read the log output at the bottom —
  the last error line says what failed (usually a missing package or a
  typo in Secrets). Secrets must be valid TOML: quotes around every value,
  one setting per line.
- **Changed the code?** Push to `main` on GitHub and Streamlit Cloud
  redeploys automatically (or use Reboot app to force it).
