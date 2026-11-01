"""ממשק Streamlit לעוזר המידע של מרפאת השיניים "אלמוג". הרצה: streamlit run app.py"""
import streamlit as st
from chromadb.errors import NotFoundError

from rag_core import CATEGORIES, SPLITTERS, ask, configure, load_index

st.set_page_config(page_title="מרפאת השיניים אלמוג | עוזר מידע", page_icon="🦷", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700&display=swap');
.stApp, .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp textarea {font-family: 'Heebo', sans-serif;}
.stApp, [data-testid="stSidebar"], [data-testid="stChatInput"] textarea {direction: rtl; text-align: right;}
.stApp :is(ul, ol) {padding-right: 1.4rem; padding-left: 0;}
.hero {background: linear-gradient(135deg, #0b8a8f, #2cc4bd); color: #fff; padding: 1.5rem 1.8rem;
       border-radius: 18px; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(15,163,163,.25);}
.hero h1 {margin: 0; padding: 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: .4rem 0 0; font-size: 1.05rem;}
[data-testid="stExpanderDetails"] :is(h1, h2, h3) {font-size: 1.05rem; padding: .4rem 0 .1rem; color: #0b7f84;}
[data-testid="stExpanderDetails"] :is(p, li, td, th) {font-size: .92rem;}
</style>""", unsafe_allow_html=True)

try:
    configure()
except RuntimeError as e:
    st.error(str(e))
    st.stop()


@st.cache_resource(show_spinner="טוען את מאגר המידע...")
def get_index(strategy):
    return load_index(strategy)  # נטען פעם אחת לכל אסטרטגיה ונשמר בזיכרון


def answer_question(question, strategy, category):
    try:
        return ask(get_index(strategy), question, strategy, category)
    except NotFoundError:  # המאגר נבנה מחדש (ingest.py) בזמן שהממשק פתוח – טוענים אותו מחדש ומנסים שוב
        get_index.clear()
        return ask(get_index(strategy), question, strategy, category)


def show_sources(sources):
    """דרישה 6: הצגת המקטעים שעליהם התבססה התשובה."""
    if sources:
        with st.expander(f"📄 מקורות ({len(sources)})"):
            for s in sources:
                with st.container(border=True):
                    st.caption(f"**{s['file']}** · {CATEGORIES[s['category']]} · דמיון {s['score']}")
                    st.markdown(s["text"])  # טבלאות וכותרות מוצגות מעוצבות (גודל הכותרות מוקטן ב-CSS)


with st.sidebar:
    st.markdown("## 🦷 מרפאת אלמוג")
    label = st.selectbox("תחום השאלה", ["כל התחומים", *CATEGORIES.values()])  # בונוס: סינון לפי Metadata
    category = next((k for k, v in CATEGORIES.items() if v == label), None)
    strategy = st.radio("שיטת חלוקת המסמכים", list(SPLITTERS),  # בונוס: השוואת אסטרטגיות Chunking
                        format_func={"sentence": "לפי גודל (משפטים)", "markdown": "לפי כותרות"}.get)
    st.divider()
    st.markdown("**🕗 שעות פעילות**  \nא׳–ה׳: 08:00 עד 20:00  \nו׳: 08:00 עד 13:00")
    if st.button("🗑️ שיחה חדשה", use_container_width=True):
        st.session_state.history = []

st.markdown('<div class="hero"><h1>🦷 מרפאת השיניים אלמוג</h1><p>שלום! אני נציג המידע של המרפאה. '
            'אפשר לשאול אותי על טיפולים, מחירים, תורים ושעות פעילות.</p></div>', unsafe_allow_html=True)

history = st.session_state.setdefault("history", [])
avatars = {"user": "🙂", "assistant": "🦷"}
for msg in history:
    with st.chat_message(msg["role"], avatar=avatars[msg["role"]]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

if question := st.chat_input("הקלידו שאלה, למשל: כמה עולה הלבנת שיניים?"):
    history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=avatars["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        with st.spinner("מחפש במאגר המידע..."):
            try:
                answer, sources = answer_question(question, strategy, category)
            except (RuntimeError, NotFoundError) as e:  # אינדקס שלא נבנה, או שהטעינה מחדש נכשלה
                st.error("המאגר עודכן – יש להפעיל מחדש את האפליקציה." if isinstance(e, NotFoundError) else str(e))
                st.stop()
        st.markdown(answer)
        show_sources(sources)
    history.append({"role": "assistant", "content": answer, "sources": sources})
