import html
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from packaging.version import Version

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

# --- Streamlit Cloud bootstrap (1/2): secrets -> environment ------------------
# On Streamlit Community Cloud there is no .env file: the app is configured in
# the dashboard's Secrets store instead. Bridge those secrets into real
# environment variables BEFORE anything reads settings (in core/config.py,
# real env vars win over the .env file). A value already set in the
# environment is never overridden. Locally and in tests there is no secrets
# store, so this block silently no-ops.
_SECRET_ENV_KEYS = (
    'LLM_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_MODEL', 'ANTHROPIC_API_KEY',
    'ANTHROPIC_MODEL', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'OLLAMA_API_BASE',
    'OLLAMA_MODEL', 'EMBEDDING_BACKEND', 'EMBEDDING_MODEL', 'RERANKER_BACKEND',
    'RERANKER_MODEL', 'NLU_BACKEND', 'NLU_LLM_MODEL', 'DATABASE_URL',
    'DATA_DIR', 'SAFERCAR_CSV', 'CHROMA_DB_PATH', 'CHUNK_STRATEGY',
    'AUTOGUIDE_API_URL', 'ENABLE_LIVE_RECALLS_API',
)
try:
    for _key in _SECRET_ENV_KEYS:
        if _key not in os.environ:
            _value = st.secrets.get(_key)
            if isinstance(_value, (str, int, float, bool)):
                os.environ[_key] = str(_value)
except Exception:
    pass

# Backend integration (see api/client.py). UI design: Sai Soujanya Yendrathi.
from api.client import answer_question, readiness, submit_feedback  # noqa: E402
from core.config import get_settings  # noqa: E402

# Streamlit >= 1.50 replaced use_container_width with width='stretch' (the old
# flag floods the log with deprecation warnings and will be removed).
FULL_WIDTH = {'width': 'stretch'} if Version(st.__version__) >= Version('1.50') else {'use_container_width': True}
st.set_page_config(page_title='AutoGuide AI | Safety Intelligence', page_icon='🚙', layout='wide', initial_sidebar_state='expanded')
_SETTINGS = get_settings()


# --- Streamlit Cloud bootstrap (2/2): vehicle data ----------------------------
def _ensure_vehicles_loaded(settings) -> None:
    """Load the NHTSA vehicle CSV when the vehicles table is empty.

    Hosted deployments (e.g. Streamlit Community Cloud) build the app from
    the git repo only, and the SQLite database is gitignored, so the app
    would boot with no vehicle data at all. The committed Safercar CSV is
    the source of truth: load it here, before the readiness check below, so
    even the first page view already shows the real vehicle count. This is
    a no-op whenever vehicles already exist (local runs, tests), and the
    only cost per rerun is one COUNT query. Any failure is logged and left
    for the readiness check to report, exactly like a local empty DB.
    """
    log = logging.getLogger(__name__)
    try:
        from sqlalchemy import func, select

        from db.models import Vehicle
        from db.session import get_sessionmaker, init_db
        from structured.safercar import load_safercar

        init_db()
        session_factory = get_sessionmaker()
        with session_factory() as db:
            count = db.scalar(select(func.count()).select_from(Vehicle)) or 0
        if count:
            return
        with session_factory() as db:
            loaded = load_safercar(db, settings.safercar_csv)
        log.info('Auto-loaded %d vehicles from %s (database was empty)',
                 loaded, settings.safercar_csv)
    except Exception as error:
        log.warning('Vehicle auto-load skipped (%s)', type(error).__name__)


_ensure_vehicles_loaded(_SETTINGS)
_PROVIDER = _SETTINGS.resolved_llm_provider()
DEMO = _PROVIDER == 'mock'
MODEL = {'openai': _SETTINGS.openai_model, 'anthropic': _SETTINGS.anthropic_model, 'gemini': _SETTINGS.gemini_model, 'ollama': _SETTINGS.ollama_model}.get(_PROVIDER, 'mock (no API key)')
try:
    _STATUS = readiness()
    _PROVIDER = _STATUS['llm_provider']
    MODEL = _STATUS['llm_model']
    DEMO = _PROVIDER == 'mock'
except Exception as error:
    logging.getLogger(__name__).warning('Backend readiness check failed (%s)', type(error).__name__)
    _STATUS = None

INDEX_READY = bool(_STATUS and _STATUS.get('checks', {}).get('vector_index'))

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp {background:#f3f6fb;color:#172a46;font-family:Inter,Arial,sans-serif}
[data-testid="stMainBlockContainer"] {max-width:1540px;padding:1.4rem 2rem 4rem}
[data-testid="stSidebar"] {background:#0b1c38;border-right:1px solid #29415f}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 {color:#e8f0ff!important}
[data-testid="stSidebar"] button {border:1px solid #3f5577!important;background:#192f52!important;color:#fff!important}
.hero-copy {background:linear-gradient(115deg,#fff,#eaf2ff);border:1px solid #e2eaf4;border-radius:17px;padding:30px;min-height:260px}
.hero-copy .eyebrow{color:#4269a6;font-size:12px;letter-spacing:1.3px;font-weight:800}
.hero-copy h1 {color:#142d51;font-size:46px;line-height:1.07;margin:16px 0 12px;font-weight:800}
.hero-copy h1 span {color:#1769e4}
.hero-copy p {font-size:15px;color:#526883;line-height:1.7;max-width:560px}
.surface {background:#fff;border:1px solid #e0e7f1;border-radius:14px;padding:18px;margin:8px 0 18px;box-shadow:0 3px 12px rgba(18,45,85,.035)}
.topic {min-height:92px;border-radius:0 0 12px 12px;padding:13px 12px;background:white;border:1px solid #e3ebf7;border-top:0;margin-top:-5px}
.topic strong{font-size:13px;color:#153254}
.topic small{display:block;font-size:11px;color:#667992;line-height:1.6;padding-top:6px}
.sec-title{font-size:21px;color:#172d4b;font-weight:800;margin:12px 0 8px}
.subtle {font-size:12px;color:#72829a}
[data-testid="stChatMessage"]{border:1px solid #e1e8f1;border-radius:13px;background:white;padding:8px}
[data-testid="stMetric"]{background:#f8fbff;border:1px solid #e8eef6;border-radius:11px;padding:11px}
.stButton>button{border-radius:9px;border:1px solid #d5e1f1;font-weight:650}
[data-testid="stImage"] img{border-radius:12px 12px 0 0} .topic small{min-height:40px} .stApp{background:#f4f7fb} .sec-title{font-size:22px} [data-testid="stSidebar"]{background:#071c37} .hero-copy{background:linear-gradient(120deg,#fdfdff,#ecf4ff)}
@media(max-width:900px){.hero-copy h1{font-size:32px}.hero-copy{min-height:0;padding:20px}}
</style>
''', unsafe_allow_html=True)

if 'messages' not in st.session_state: st.session_state.messages=[]
if 'pending' not in st.session_state: st.session_state.pending=None
if 'session_id' not in st.session_state: st.session_state.session_id=None
if 'view' not in st.session_state: st.session_state.view='chat'

ALLOWED_SOURCE_HOSTS=('nhtsa.gov','autoguide.com','iihs.org')
EXACT_SOURCE_HOSTS=('en.wikipedia.org','www.fueleconomy.gov')

def nhtsa_link(url):
    """Allow links only to the approved project sources."""
    try:
        u=urlparse(str(url)); h=u.hostname or ''
        return u.scheme=='https' and not u.username and not u.password and u.port in (None,443) and (h in EXACT_SOURCE_HOSTS or any(h==d or h.endswith('.'+d) for d in ALLOWED_SOURCE_HOSTS))
    except (TypeError, ValueError): return False

def show_sources(sources):
    for n,source in enumerate(sources,1):
        if not isinstance(source,dict): continue
        url=source.get('url','')
        if nhtsa_link(url):
            st.link_button(f'{source.get("ref",f"[{n}]")} {source.get("title","Vehicle reference")}',url,**FULL_WIDTH)
            if urlparse(url).hostname=='en.wikipedia.org':
                st.caption('Based on Wikipedia contributors. Extracted and paraphrased text; '
                           '[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). '
                           f'[Contributors and history]({url.split("?")[0]}?action=history).')

def ask_in_chat(question):
    """Send a question to the Safety Assistant view and run it there."""
    st.session_state.view='chat'
    st.session_state.pending=question
    st.rerun()

# ------------------------------------------------------------ vehicle data
# Direct structured-data access for the Lookup / Recalls / Features views.
# The UI runs the backend in-process by default, so this reads the same
# vehicles table the chat backend resolves against.

@st.cache_data(ttl=600, show_spinner=False)
def vehicle_makes():
    try:
        from sqlalchemy import select
        from db.models import Vehicle
        from db.session import get_sessionmaker
        with get_sessionmaker()() as db:
            return list(db.scalars(select(Vehicle.make).distinct().order_by(Vehicle.make)).all())
    except Exception as error:
        logging.getLogger(__name__).warning('Makes lookup failed (%s)', type(error).__name__)
        return []

@st.cache_data(ttl=600, show_spinner=False)
def vehicle_families(make):
    try:
        from sqlalchemy import select
        from db.models import Vehicle
        from db.session import get_sessionmaker
        with get_sessionmaker()() as db:
            return list(db.scalars(select(Vehicle.model_family).distinct()
                                   .where(Vehicle.make == make).order_by(Vehicle.model_family)).all())
    except Exception as error:
        logging.getLogger(__name__).warning('Models lookup failed (%s)', type(error).__name__)
        return []

@st.cache_data(ttl=600, show_spinner=False)
def vehicle_years(make, family):
    try:
        from sqlalchemy import select
        from db.models import Vehicle
        from db.session import get_sessionmaker
        with get_sessionmaker()() as db:
            years = db.scalars(select(Vehicle.model_year).distinct()
                               .where(Vehicle.make == make, Vehicle.model_family == family)).all()
            return sorted((int(y) for y in years), reverse=True)
    except Exception as error:
        logging.getLogger(__name__).warning('Years lookup failed (%s)', type(error).__name__)
        return []

@st.cache_data(ttl=600, show_spinner=False)
def vehicle_variants(make, family, year):
    try:
        from sqlalchemy import select
        from db.models import Vehicle
        from db.session import get_sessionmaker
        from structured.safercar import to_record
        with get_sessionmaker()() as db:
            rows = db.scalars(select(Vehicle).where(
                Vehicle.make == make, Vehicle.model_family == family,
                Vehicle.model_year == int(year)).order_by(Vehicle.variant_label)).all()
            return [to_record(v).model_dump() for v in rows]
    except Exception as error:
        logging.getLogger(__name__).warning('Variants lookup failed (%s)', type(error).__name__)
        return []

@st.cache_data(ttl=600, show_spinner=False)
def feature_stats():
    """Real availability counts per safety feature, from the vehicles table."""
    try:
        from collections import Counter
        from sqlalchemy import select
        from db.models import Vehicle
        from db.session import get_sessionmaker
        from structured.safercar import FEATURES
        counts = {key: Counter() for key in FEATURES}
        total = 0
        with get_sessionmaker()() as db:
            for features in db.scalars(select(Vehicle.features)).all():
                total += 1
                for key, status in (features or {}).items():
                    if key in counts:
                        counts[key][status] += 1
        return {'total': total, 'counts': {k: dict(v) for k, v in counts.items()}}
    except Exception as error:
        logging.getLogger(__name__).warning('Feature stats failed (%s)', type(error).__name__)
        return {'total': 0, 'counts': {}}

def pick_vehicle(prefix):
    """Cascading Make -> Model -> Year pickers backed by the vehicles table.
    Returns (make, family, year) or None when no vehicle data is available."""
    makes = vehicle_makes()
    if not makes:
        st.error('Vehicle data is not available right now. Please try again in a moment.')
        return None
    saved_make = st.session_state.get(f'{prefix}_make')
    make = st.selectbox('Make', makes,
                        index=makes.index(saved_make) if saved_make in makes else 0,
                        key=f'{prefix}_make_sel')
    st.session_state[f'{prefix}_make'] = make
    families = vehicle_families(make)
    if not families:
        st.error('No models were found for that make.')
        return None
    saved_family = st.session_state.get(f'{prefix}_family')
    family = st.selectbox('Model', families,
                          index=families.index(saved_family) if saved_family in families else 0,
                          key=f'{prefix}_family_sel')
    st.session_state[f'{prefix}_family'] = family
    years = vehicle_years(make, family)
    if not years:
        st.error('No model years were found for that vehicle.')
        return None
    saved_year = st.session_state.get(f'{prefix}_year')
    year = st.selectbox('Year', years,
                        index=years.index(saved_year) if saved_year in years else 0,
                        key=f'{prefix}_year_sel')
    st.session_state[f'{prefix}_year'] = year
    return make, family, year

VIEWS = {'chat': 'Safety Assistant', 'lookup': 'Vehicle Lookup', 'recalls': 'Vehicle Recalls',
         'features': 'Safety Features', 'resources': 'NHTSA Resources'}

with st.sidebar:
    st.markdown('## 🚘 AutoGuide AI')
    st.caption('SAFETY INTELLIGENCE PLATFORM')
    st.divider()
    st.markdown('**WORKSPACE**')
    for view_key, label, icon in [('chat', 'Safety Assistant', '▣'), ('lookup', 'Vehicle Lookup', '🔎'),
                                  ('resources', 'NHTSA Resources', '📄'), ('features', 'Safety Features', '🛡️'),
                                  ('recalls', 'Vehicle Recalls', '🚨')]:
        active = '● ' if st.session_state.view == view_key else ''
        if st.button(f'{icon} {label} {active}'.strip(), key=f'nav_{view_key}', **FULL_WIDTH):
            st.session_state.view = view_key
            st.rerun()
    st.caption(f'Current view: {VIEWS[st.session_state.view]}')
    st.divider()
    if st.button('＋ New Conversation',**FULL_WIDTH):
        st.session_state.messages=[];st.session_state.pending=None;st.session_state.session_id=None;st.rerun()
    transcript='\n\n'.join(f'{m["role"].upper()}: {m["content"]}' for m in st.session_state.messages)
    st.download_button('↓ Export Chat',transcript or 'No conversation yet','autoguide_chat.txt','text/plain',**FULL_WIDTH)
    st.divider()
    st.markdown('**SYSTEM DETAILS**')
    st.caption('Sources: NHTSA ratings/recalls + loaded AutoGuide, Wikipedia, FuelEconomy.gov and reference documents')
    st.caption(f'Configured generation model: {MODEL}')
    st.caption('Mode: DEMO — mock LLM' if DEMO else f'Configured provider: {_PROVIDER}')
    if _STATUS:
        st.caption(f'Vehicle records: {_STATUS["vehicles_loaded"]:,}')
        if _STATUS['checks']['vector_index']:
            st.caption(f'Article library: {_STATUS["chunks_loaded"]:,} passages')
            sources=_STATUS.get('indexed_sources',{})
            wiki_count=sources.get('wikipedia_article',0); fuel_count=sources.get('fueleconomy_page',0)
            st.caption(f'Website passages: Wikipedia {wiki_count:,} · FuelEconomy.gov {fuel_count:,}')
            if not wiki_count or not fuel_count:
                st.caption('History and fuel economy coverage depends on the corresponding website pages being collected and indexed.')
        else:
            st.warning('Article library unavailable. General safety answers need the article index.')
        st.caption('Backend setup complete' if _STATUS['ready'] else 'Backend setup incomplete; vehicle lookups may still work.')
        st.caption('External service availability has not been checked.')
    else:
        st.warning('Backend status unavailable. Check the backend connection.')
    st.divider()
    st.caption('MSAI 633 • Automotive Safety RAG')

# ================================================================ CHAT VIEW
def render_chat():
    st.caption('AUTOGUIDE  /  INTELLIGENCE WORKSPACE')
    st.title('Vehicle Safety Intelligence')
    st.caption('An evidence-grounded assistant for understanding automotive safety technologies.')

    hero_l,hero_r=st.columns([1.25,1],gap='medium')
    with hero_l:
        st.markdown('''<div class="hero-copy"><div class="eyebrow">DRIVE SMARTER. STAY SAFER.</div>
<h1>Auto<span>Guide</span> AI</h1><p>Ask about vehicle safety, recalls, history, specifications and fuel economy. Answers link to the available sources.</p>
<p>Safety Features &nbsp; • &nbsp; Recalls &nbsp; • &nbsp; Ratings</p></div>''',unsafe_allow_html=True)
    with hero_r:
        st.image(str(ROOT/'assets'/'vehicle_hero.jpg'),**FULL_WIDTH)

    main,side=st.columns([2.25,1],gap='medium')
    with main:
        st.markdown('<div class="sec-title">Explore vehicle safety</div>',unsafe_allow_html=True)
        topics=[
            ('braking.jpg','🛑','Braking Systems','ABS and emergency braking technologies','How does automatic emergency braking work?'),
            ('airbags.jpg','🛡️','Airbag Safety','Passenger protection and airbag systems','How do airbags protect passengers?'),
            ('driver_assistance.jpg','📡','Driver Assistance','Blind spots, lane alerts and driver support','What does blind spot monitoring do?'),
            ('safety_ratings.jpg','⭐','Safety Ratings','Crash tests and NHTSA safety scores','How are vehicle safety ratings determined?')
        ]
        cols=st.columns(4,gap='small')
        for i,(picture,emoji,title,desc,question) in enumerate(topics):
            with cols[i]:
                st.image(str(ROOT/'assets'/picture),**FULL_WIDTH)
                st.markdown(f'<div class="topic"><strong>{emoji} {html.escape(title)}</strong><small>{html.escape(desc)}</small></div>',unsafe_allow_html=True)
                if st.button('Explore →',key=f'topic{i}',**FULL_WIDTH):
                    st.session_state.pending=question;st.rerun()
        st.markdown('<div class="sec-title">Chat with AutoGuide AI</div>',unsafe_allow_html=True)
        if DEMO:
            st.info('Demo mode: no LLM API key configured, so answers are extractive placeholders. Retrieval, clarification and citations are real.')
        if not INDEX_READY:
            st.warning('Article library is not loaded in this demo, so general "how does it work?" questions '
                       'cannot be answered from articles yet. Vehicle ratings, safety-feature lookups, '
                       'comparisons and recalls work fully — try asking about a specific vehicle below, '
                       'or open Vehicle Lookup in the sidebar.')
        if not st.session_state.messages:
            with st.chat_message('assistant',avatar='🚘'):
                st.markdown('**Welcome to AutoGuide AI.** Ask about vehicle safety, history, specifications or fuel economy, or choose a topic above.')
            st.caption('TRY A QUESTION')
            if INDEX_READY:
                suggestions=[('Automatic braking','How does automatic emergency braking work?'),('Airbag operation','How do airbags protect passengers?'),('Blind spot monitoring','What does blind spot monitoring do?'),('Vehicle ratings','How are vehicle safety ratings determined?'),('Vehicle history','What is the history of the Toyota Camry?'),('Fuel economy','What is MPG?')]
            else:
                suggestions=[('2020 Camry rating','What is the overall NHTSA star rating of the 2020 Toyota Camry?'),('Civic blind spot detection','Does the 2022 Honda Civic have blind spot detection?'),('F-150 recalls','Are there any recalls for the 2020 Ford F-150?'),('Compare RAV4 vs CR-V','2024 Toyota RAV4 vs 2024 Honda CR-V'),('Is the CR-V safe?','Is the 2020 Honda CR-V safe?'),('Accord safety features','What safety features does the 2021 Honda Accord have?')]
            q1,q2=st.columns(2)
            for i,(label,prompt) in enumerate(suggestions):
                with (q1 if i%2==0 else q2):
                    if st.button(label,key=f'suggest{i}',**FULL_WIDTH):st.session_state.pending=prompt;st.rerun()
        for idx,m in enumerate(st.session_state.messages):
            with st.chat_message(m['role'],avatar='🚘' if m['role']=='assistant' else '👤'):
                st.markdown(m['content'])
                if m.get('sources'):show_sources(m['sources'])
                if m.get('disclaimer'):st.caption(m['disclaimer'])
                if m['role']=='assistant' and m.get('message_id'):
                    if m.get('feedback_rating'):
                        st.caption('Feedback saved. Thank you.')
                    else:
                        up,down=st.columns(2)
                        rating=None
                        if up.button('👍 Helpful',key=f'feedback_up_{m["message_id"]}'):
                            rating=1
                        if down.button('👎 Not helpful',key=f'feedback_down_{m["message_id"]}'):
                            rating=-1
                        if rating is not None:
                            try:
                                submit_feedback(m['message_id'],rating)
                                m['feedback_rating']=rating
                                st.rerun()
                            except Exception as error:
                                logging.getLogger(__name__).warning('Feedback save failed (%s)',type(error).__name__)
                                st.error('Feedback could not be saved. Please try again.')
                clar=m.get('clarification')
                if clar and clar.get('options') and idx==len(st.session_state.messages)-1:
                    opt_cols=st.columns(min(3,len(clar['options'])))
                    for j,option in enumerate(clar['options']):
                        if opt_cols[j%len(opt_cols)].button(option,key=f'clar{idx}_{j}',**FULL_WIDTH):
                            st.session_state.pending=option;st.rerun()
        typed=st.chat_input('Ask about vehicle safety, history, specifications or fuel economy...')
        question=typed or st.session_state.pending
        if question:
            st.session_state.pending=None
            history=st.session_state.messages.copy()
            st.session_state.messages.append({'role':'user','content':question})
            try:
                with st.spinner('Searching knowledge and preparing answer...'):
                    response=answer_question(question,history=history,session_id=st.session_state.session_id)
                st.session_state.session_id=response.get('session_id',st.session_state.session_id)
                st.session_state.messages.append({'role':'assistant','message_id':response.get('message_id'),'content':response.get('answer','No answer available.'),'sources':response.get('sources',[]),'trace':response.get('trace',[]),'refused':response.get('refused',False),'status':response.get('status'),'clarification':response.get('clarification'),'disclaimer':response.get('disclaimer'),'debug':response.get('debug')})
            except Exception as error:
                logging.getLogger(__name__).warning('Chat answer failed (%s)',type(error).__name__)
                st.session_state.messages.append({'role':'assistant','content':'Sorry — something went wrong while preparing that answer. Please try again in a moment.','sources':[],'trace':[],'refused':True})
            st.rerun()
    with side:
        st.markdown('<div class="sec-title">Session overview</div>',unsafe_allow_html=True)
        count=sum(m['role']=='user' for m in st.session_state.messages)
        cited=sum(bool(m.get('sources')) for m in st.session_state.messages if m['role']=='assistant')
        used=sum(len(m.get('sources',[])) for m in st.session_state.messages if m['role']=='assistant')
        a,b,c=st.columns(3)
        a.metric('Questions',count);b.metric('Cited',cited);c.metric('Sources',used)
        st.caption('NHTSA + loaded vehicle documents • model: '+MODEL)
        st.divider()
        st.markdown('**Evidence inspector**')
        latest=next((m for m in reversed(st.session_state.messages) if m['role']=='assistant'),None)
        if latest:
            if latest.get('refused'):st.warning('Insufficient evidence / answer unavailable')
            if latest.get('status')=='needs_clarification':st.info('Waiting for vehicle details')
            dbg=latest.get('debug') or {}
            if dbg:st.caption(f"Intent: {dbg.get('intent') or '-'}  •  Vehicle: {dbg.get('resolution') or '-'}  •  {dbg.get('timings_ms',{}).get('total','?')} ms")
            if (dbg.get('llm') or {}).get('error'):st.error('Something went wrong while generating that answer. Please try again.')
            if (dbg.get('llm') or {}).get('fallback_reason'):st.warning('AI model unreachable, showing source data: '+str(dbg['llm']['fallback_reason'])[:200])
            st.caption(f'Sources: {len(latest.get("sources",[]))}')
            show_sources(latest.get('sources',[]))
            trace=latest.get('trace',[])
            if trace:
                with st.expander(f'Retrieved passages ({len(trace)})'):
                    for c in trace:
                        if not isinstance(c,dict):continue
                        st.markdown('**'+str(c.get('title','Document'))+'**')
                        st.caption(f'Score: {c.get("score","N/A")}  •  Rerank: {c.get("rerank_score","N/A")}')
                        st.write(str(c.get('text',''))[:500])
                        st.divider()
        else:st.info('Ask a question to inspect retrieved evidence.')
        st.divider()
        st.markdown('**NHTSA Quick Links**')
        for label,url in [('Vehicle safety ratings','https://www.nhtsa.gov/ratings'),('Recall information','https://www.nhtsa.gov/recalls'),('Driver assistance technologies','https://www.nhtsa.gov/vehicle-safety/driver-assistance-technologies'),('Research & Reports','https://www.nhtsa.gov/research-data')]:
            st.link_button(label+' ↗',url,**FULL_WIDTH)
        st.caption('Educational information only; not professional mechanical advice.')

# ============================================================ LOOKUP VIEW
def _stars(value):
    return 'Not rated' if value is None else f'{value:g} ★'

def render_lookup():
    from structured.safercar import AVAILABILITY_TEXT, EVALUATIONS, FEATURES
    st.caption('AUTOGUIDE  /  VEHICLE LOOKUP')
    st.title('Vehicle Lookup')
    st.caption('Pick a vehicle to see its NHTSA 5-Star ratings and safety features, straight from the ratings data.')
    picked = pick_vehicle('lookup')
    if picked is None:
        return
    make, family, year = picked
    variants = vehicle_variants(make, family, year)
    if not variants:
        st.warning(f'NHTSA ratings data has no {year} {make.title()} {family.title()}. Try a different year.')
        return
    record = variants[0]
    if len(variants) > 1:
        labels = [v['variant_label'] for v in variants]
        saved = st.session_state.get('lookup_variant')
        choice = st.selectbox('Version — NHTSA rates each version separately', labels,
                              index=labels.index(saved) if saved in labels else 0, key='lookup_variant_sel')
        st.session_state['lookup_variant'] = choice
        record = variants[labels.index(choice)]
    st.markdown(f'<div class="sec-title">{record["model_year"]} {html.escape(record["make"].title())} {html.escape(record["model"])}</div>', unsafe_allow_html=True)
    st.caption(f'Version: {record["variant_label"]}' + (f'  •  Body: {record["body_style"]}' if record.get('body_style') else '') + (f'  •  Drive: {record["drive_train"]}' if record.get('drive_train') else ''))
    r1, r2, r3, r4 = st.columns(4)
    r1.metric('Overall rating', _stars(record.get('overall_stars')))
    r2.metric('Frontal crash', _stars(record.get('frontal_stars')))
    r3.metric('Side crash', _stars(record.get('side_stars')))
    r4.metric('Rollover', _stars(record.get('rollover_stars')))
    r5, r6 = st.columns(2)
    r5.metric('Side pole crash', _stars(record.get('side_pole_stars')))
    if record.get('rollover_risk_pct') is not None:
        r6.metric('Rollover risk', f'{record["rollover_risk_pct"]:g}%')
    st.markdown('**Safety features**')
    rows = [{'Feature': label, 'Availability': AVAILABILITY_TEXT.get(record.get('features', {}).get(key, 'unknown'), 'not reported')}
            for key, (label, _cols) in FEATURES.items()]
    st.dataframe(rows, hide_index=True, **FULL_WIDTH)
    if record.get('evaluations'):
        st.markdown('**NHTSA technology evaluations**')
        for key, result in record['evaluations'].items():
            label = EVALUATIONS.get(key, (key.replace('_', ' ').title(), ''))[0]
            st.markdown(f'- **{html.escape(label)}:** {html.escape(str(result))}')
    if record.get('safety_concerns'):
        st.markdown('**Safety concerns noted by NHTSA**')
        for concern in record['safety_concerns']:
            st.markdown(f'- {html.escape(str(concern))}')
    st.caption('Source: NHTSA 5-Star Safety Ratings data. Ratings apply to this exact version; other versions of the same model can differ.')
    b1, b2 = st.columns(2)
    with b1:
        if st.button('💬 Ask the assistant about this vehicle', key='lookup_ask', **FULL_WIDTH):
            ask_in_chat(f'What is the overall NHTSA star rating of the {record["model_year"]} {record["make"].title()} {record["model"].title()}?')
    with b2:
        if st.button('🚨 Check recalls for this vehicle', key='lookup_recalls', **FULL_WIDTH):
            st.session_state['recall_make'] = make
            st.session_state['recall_family'] = family
            st.session_state['recall_year'] = year
            st.session_state.view = 'recalls'
            st.rerun()

# ============================================================ RECALLS VIEW
def render_recalls():
    st.caption('AUTOGUIDE  /  VEHICLE RECALLS')
    st.title('Vehicle Recalls')
    st.caption('Check NHTSA recall campaigns for a vehicle by year, make and model.')
    picked = pick_vehicle('recall')
    if picked is None:
        return
    make, family, year = picked
    label = f'{year} {make.title()} {family.title()}'
    if st.button(f'🚨 Check recalls for the {label}', key='recall_submit', **FULL_WIDTH):
        try:
            with st.spinner('Checking NHTSA recall records...'):
                result = answer_question(f'Are there any recalls for the {year} {make.title()} {family.title()}?')
            st.session_state['recall_result'] = result
            st.session_state['recall_label'] = label
        except Exception as error:
            logging.getLogger(__name__).warning('Recall lookup failed (%s)', type(error).__name__)
            st.error('The recall check could not be completed. Please try again in a moment.')
    result = st.session_state.get('recall_result')
    if result and st.session_state.get('recall_label'):
        st.markdown(f'<div class="sec-title">Results for the {html.escape(st.session_state["recall_label"])}</div>', unsafe_allow_html=True)
        st.markdown(result.get('answer', 'No result was returned.'))
        if result.get('sources'):
            st.markdown('**Sources**')
            show_sources(result['sources'])
        if result.get('disclaimer'):
            st.caption(result['disclaimer'])
    else:
        st.info('Choose a vehicle above and press the check button. Results come from the NHTSA Recalls service through the same backend the chat uses.')
    st.divider()
    st.warning('Model-level results do not prove your specific car is affected. Always confirm with your VIN.')
    st.link_button('Check your VIN at NHTSA ↗', 'https://www.nhtsa.gov/recalls', **FULL_WIDTH)
    st.caption('If your VIN shows an open recall, contact a dealer for the free remedy. Recall information can change; NHTSA is the authoritative source.')

# =========================================================== FEATURES VIEW
FEATURE_INFO = {
    'abs': ('Prevents the wheels from locking during hard braking, so the driver can keep steering while stopping.', 'How does ABS work?'),
    'electronic_stability_control': ('Detects a loss of steering control and brakes individual wheels to help keep the vehicle on course.', 'How does electronic stability control work?'),
    'forward_collision_warning': ('Warns the driver when a crash with the vehicle ahead is likely, so they can brake or steer in time.', 'What does forward collision warning do?'),
    'automatic_emergency_braking': ('Detects a likely front crash and can apply the brakes automatically if the driver does not react.', 'How does automatic emergency braking work?'),
    'dynamic_brake_support': ('Adds extra braking force when the driver brakes for a likely crash but does not press hard enough.', 'What is dynamic brake support?'),
    'lane_departure_warning': ('Alerts the driver when the vehicle starts to drift out of its lane without a turn signal.', 'What does lane departure warning do?'),
    'blind_spot_detection': ('Watches the areas beside and behind the vehicle that mirrors may not show, and alerts the driver to vehicles there.', 'What does blind spot monitoring do?'),
    'adaptive_cruise_control': ('Keeps a set speed and following distance, speeding up and slowing down with traffic.', 'How does adaptive cruise control work?'),
    'backup_camera': ('Shows the area behind the vehicle on a screen while reversing.', 'How do backup cameras help safety?'),
    'daytime_running_lights': ('Front lights that stay on during the day so other drivers can see the vehicle sooner.', 'What are daytime running lights?'),
    'head_side_airbags': ('Curtain airbags that deploy along the side windows to protect occupants’ heads in a side crash.', 'How do airbags protect passengers?'),
    'torso_side_airbags': ('Side airbags that protect the chest and torso in a side-impact crash.', 'How do side airbags work?'),
    'pelvis_side_airbags': ('Side airbags positioned to protect the pelvis in a side-impact crash.', 'How do side airbags work?'),
    'knee_airbags': ('Airbags that deploy below the dashboard to protect knees and help position the occupant.', 'How do airbags protect passengers?'),
    'seat_belt_pretensioners': ('Pull the seat belt snug in the first moments of a crash to position the occupant correctly.', 'How do seat belt pretensioners work?'),
    'seat_belt_load_limiters': ('Let the belt spool out slightly after the initial crash force, reducing chest pressure.', 'How do seat belt load limiters work?'),
    'automatic_crash_notification': ('Notifies emergency services automatically after a serious crash.', 'What is automatic crash notification?'),
    'automatic_door_locks': ('Lock the doors automatically once the vehicle is moving.', 'What are automatic door locks for?'),
}

def render_features():
    from structured.safercar import FEATURES
    st.caption('AUTOGUIDE  /  SAFETY FEATURES')
    st.title('Safety Features Explorer')
    st.caption('What each technology does, how common it is in the NHTSA ratings data, and a shortcut to ask the assistant about it.')
    if not INDEX_READY:
        st.info('Article explanations are not loaded in this demo, so "Ask how it works" questions get an honest '
                'notice in the chat instead of an article answer. The availability counts below are live from the '
                '17,313-vehicle NHTSA ratings data and always work.')
    stats = feature_stats()
    total = stats.get('total', 0)
    keys = list(FEATURES.keys())
    for start in range(0, len(keys), 3):
        cols = st.columns(3, gap='small')
        for offset, key in enumerate(keys[start:start + 3]):
            label = FEATURES[key][0]
            desc, question = FEATURE_INFO.get(key, ('', f'How does {label} work?'))
            counts = stats.get('counts', {}).get(key, {})
            standard = counts.get('standard', 0)
            with cols[offset]:
                st.markdown(f'<div class="surface"><strong>{html.escape(label)}</strong>'
                            f'<p class="subtle">{html.escape(desc)}</p></div>', unsafe_allow_html=True)
                if total:
                    st.caption(f'Standard on {standard:,} of {total:,} rated vehicles ({standard / total:.0%}). '
                               f'Optional on {counts.get("optional", 0):,}.')
                if st.button('Ask how it works →', key=f'feature_{key}', **FULL_WIDTH):
                    ask_in_chat(question)

# ========================================================== RESOURCES VIEW
def render_resources():
    st.caption('AUTOGUIDE  /  NHTSA RESOURCES')
    st.title('NHTSA Resources')
    st.caption('Official National Highway Traffic Safety Administration pages and how to use them.')
    c1, c2 = st.columns(2, gap='medium')
    with c1:
        st.markdown('<div class="sec-title">Official links</div>', unsafe_allow_html=True)
        for label, url in [('Vehicle safety ratings', 'https://www.nhtsa.gov/ratings'),
                           ('Recall information & VIN check', 'https://www.nhtsa.gov/recalls'),
                           ('Driver assistance technologies', 'https://www.nhtsa.gov/vehicle-safety/driver-assistance-technologies'),
                           ('Research & data', 'https://www.nhtsa.gov/research-data'),
                           ('NHTSA home', 'https://www.nhtsa.gov')]:
            st.link_button(label + ' ↗', url, **FULL_WIDTH)
        st.markdown('<div class="sec-title">Check a recall by VIN</div>', unsafe_allow_html=True)
        st.markdown('1. Find the 17-character VIN on the driver-side dashboard (visible through the windshield) or on the driver’s door frame.\n'
                    '2. Open the NHTSA recall page above and enter the VIN.\n'
                    '3. If a recall is open, contact any dealer of the brand — recall repairs are free.\n'
                    '4. A model-level search (like the one in this app) is a starting point only; the VIN result is the one that applies to your car.')
    with c2:
        st.markdown('<div class="sec-title">How NHTSA 5-Star Ratings work</div>', unsafe_allow_html=True)
        st.markdown('NHTSA crash-tests vehicles and rates them from 1 to 5 stars. The overall rating combines '
                    'frontal crash, side crash and rollover resistance results — more stars mean the vehicle '
                    'performed better in those tests. Ratings apply to the exact version tested, so use the '
                    'Vehicle Lookup view to compare versions of the same model.')
        st.markdown('<div class="sec-title">About this demo</div>', unsafe_allow_html=True)
        if _STATUS:
            st.markdown(f'- **Vehicle records loaded:** {_STATUS["vehicles_loaded"]:,}\n'
                        f'- **Answer model:** {MODEL} (provider: {_PROVIDER})\n'
                        f'- **Article library:** {"loaded — " + format(_STATUS["chunks_loaded"], ",") + " passages" if INDEX_READY else "not loaded in this deployment, so article-based answers are unavailable; ratings, features and recalls are unaffected"}')
        else:
            st.markdown('Backend status is unavailable right now.')
        st.caption('Educational information only; not professional mechanical advice. For recall status of a specific car, NHTSA’s VIN check is authoritative.')

# ================================================================ DISPATCH
{'chat': render_chat, 'lookup': render_lookup, 'recalls': render_recalls,
 'features': render_features, 'resources': render_resources}[st.session_state.view]()
