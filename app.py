import html
import logging
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from packaging.version import Version

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

# Backend integration (see api/client.py). UI design: Sai Soujanya Yendrathi.
from api.client import answer_question, readiness, submit_feedback  # noqa: E402
from core.config import get_settings  # noqa: E402

# Streamlit >= 1.50 replaced use_container_width with width='stretch' (the old
# flag floods the log with deprecation warnings and will be removed).
FULL_WIDTH = {'width': 'stretch'} if Version(st.__version__) >= Version('1.50') else {'use_container_width': True}
st.set_page_config(page_title='AutoGuide AI | Safety Intelligence', page_icon='🚙', layout='wide', initial_sidebar_state='expanded')
_SETTINGS = get_settings()
_PROVIDER = _SETTINGS.resolved_llm_provider()
DEMO = _PROVIDER == 'mock'
MODEL = {'openai': _SETTINGS.openai_model, 'anthropic': _SETTINGS.anthropic_model, 'gemini': _SETTINGS.gemini_model}.get(_PROVIDER, 'mock (no API key)')
try:
    _STATUS = readiness()
    _PROVIDER = _STATUS['llm_provider']
    MODEL = _STATUS['llm_model']
    DEMO = _PROVIDER == 'mock'
except Exception as error:
    logging.getLogger(__name__).warning('Backend readiness check failed (%s)', type(error).__name__)
    _STATUS = None

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

with st.sidebar:
    st.markdown('## 🚘 AutoGuide AI')
    st.caption('SAFETY INTELLIGENCE PLATFORM')
    st.divider()
    st.markdown('**WORKSPACE**')
    st.markdown('**▣ Safety Assistant**')
    st.markdown('📄 NHTSA Resources')
    st.markdown('🛡️ Safety Features')
    st.markdown('🚨 Vehicle Recalls')
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
    if not st.session_state.messages:
        with st.chat_message('assistant',avatar='🚘'):
            st.markdown('**Welcome to AutoGuide AI.** Ask about vehicle safety, history, specifications or fuel economy, or choose a topic above.')
        st.caption('TRY A QUESTION')
        q1,q2=st.columns(2)
        for i,(label,prompt) in enumerate([('Automatic braking','How does automatic emergency braking work?'),('Airbag operation','How do airbags protect passengers?'),('Blind spot monitoring','What does blind spot monitoring do?'),('Vehicle ratings','How are vehicle safety ratings determined?'),('Vehicle history','What is the history of the Toyota Camry?'),('Fuel economy','What is MPG?')]):
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
            st.session_state.messages.append({'role':'assistant','content':f'Unable to answer. Check backend configuration. ({type(error).__name__})','sources':[],'trace':[],'refused':True})
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
        if (dbg.get('llm') or {}).get('error'):st.error('Backend error: '+str(dbg['llm']['error'])[:300])
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
