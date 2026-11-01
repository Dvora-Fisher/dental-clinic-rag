"""ליבת מערכת ה-RAG: הגדרות, טעינת מסמכים, חיתוך, אינדקס ב-Chroma, אחזור ויצירת תשובה."""
import truststore

truststore.inject_into_ssl()  # אימות תעודות דרך Windows (פותר חסימה של אנטי-וירוס שסורק HTTPS, כמו ESET)

import os
import random
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google.genai import errors as genai_errors, types as genai_types
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).parent
DATA_DIR, DB_DIR = ROOT / "data", ROOT / "chroma_db"
# כמה מקטעים נשלחים ל-LLM. ב-markdown המקטעים קטנים, ומידע קשור מפוזר בכמה סעיפים – לכן 6
TOP_K = {"markdown": 6, "sentence": 4}
# סף דמיון לכל אסטרטגיה – כויל לפי evaluate.py. זהיר בכוונה: חוסם רק שאלות רחוקות בבירור מהמאגר;
# מה שעובר את הסף ואין לו תשובה – נעצר על ידי ההוראה בפרומפט.
MIN_SCORE = {"sentence": 0.52, "markdown": 0.54}
MIN_CHUNK_CHARS = 50  # מקטעים קצרים מזה (למשל כותרת בלי תוכן) לא נשמרים
NO_ANSWER_TOKEN = "[[NO_ANSWER]]"  # סימן פנימי שהמודל מחזיר כשאין תשובה; לא מוצג למשתמש
NO_ANSWER_MESSAGES = [
    "מצטער, את המידע הזה אין לי כרגע 🙏 אפשר להתקשר למרפאה ונשמח לעזור.",
    "שאלה טובה! לצערי אין לי עליה מידע. הצוות שלנו בטלפון ישמח לענות 😊",
    "על זה אני לא יודע לענות, מצטער. אולי אוכל לעזור בשאלה אחרת על המרפאה?",
    "אין לי מידע על כך במאגר שלי 🙏 מוזמנים לפנות לדלפק הקבלה ונבדוק בשבילכם.",
]
SERVICE_BUSY = "אוי, המערכת עמוסה כרגע 🙏 נסו שוב בעוד דקה."
CATEGORIES = {"general": "כללי", "treatments": "טיפולים", "pricing": "מחירים ותשלומים", "policies": "נהלים"}

# שתי אסטרטגיות חיתוך להשוואה (בונוס). הראשונה היא ברירת המחדל: markdown שומרת כל סעיף שלם,
# ואילו ב-sentence החפיפה בין מקטעים שכפלה שורה ברשימת הרופאים וגרמה לספירה שגויה.
SPLITTERS = {
    "markdown": lambda: MarkdownNodeParser(),                                # מקטע לכל סעיף לפי כותרות
    "sentence": lambda: SentenceSplitter(chunk_size=512, chunk_overlap=50),  # לפי גודל, בלי לשבור משפטים
}

QA_PROMPT = PromptTemplate(
    "אתה נציג השירות של מרפאת השיניים \"אלמוג\": חם, אדיב ולבבי.\n"
    "ענה בעברית, בקצרה (עד 3–4 משפטים), בניסוח טבעי ומגוון, ופנה לשואל בלשון ניטרלית. מותר אימוג'י אחד כשמתאים.\n"
    "אל תפתח כל תשובה באותה מילה; לפעמים פתח ישר בתשובה. הימנע מהגזמות.\n"
    "השתמש רק במידע שבין הקווים. מותר לספור, לסכם ולשלב פרטים מכמה מקומות במידע, אבל אסור להוסיף עובדות שלא מופיעות בו.\n"
    f"אם המידע לא מאפשר לענות על השאלה, החזר רק: {NO_ANSWER_TOKEN}\n"
    "אל תיתן אבחנה או ייעוץ רפואי אישי; במקום זה הצע לקבוע תור במרפאה.\n"
    "---------\n{context_str}\n---------\n"
    "שאלה: {query_str}\nתשובה: "
)


def configure():
    """מגדיר את מודלי Gemini. פונקציה נפרדת כדי שבבדיקות אפשר יהיה להחליף במודלי דמה."""
    load_dotenv(ROOT / ".env")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("חסר GOOGLE_API_KEY בקובץ .env")
    # שכבת ניסיונות חוזרים אחת בלבד (2 ניסיונות, עד 30 שניות לקריאה) – כדי לא להיתקע כשהמודל עמוס
    http = genai_types.HttpOptions(timeout=30_000, retry_options=genai_types.HttpRetryOptions(attempts=2))
    Settings.llm = GoogleGenAI(model=os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"), api_key=key, temperature=0.7,  # 0.7: ניסוח מגוון, צמוד לעובדות
                               max_retries=0, http_options=http)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=os.getenv("EMBED_MODEL", "gemini-embedding-001"), api_key=key)


def load_documents():
    """טוען את קבצי ה-Markdown. הקטגוריה = שם תת-התיקייה (הבסיס לסינון לפי Metadata)."""
    return SimpleDirectoryReader(
        str(DATA_DIR), recursive=True, required_exts=[".md"],
        file_metadata=lambda p: {"file_name": Path(p).name, "category": Path(p).parent.name},
    ).load_data()


def _collection(strategy, reset=False):
    """אוסף נפרד ב-Chroma לכל אסטרטגיית חיתוך. reset מוחק אותו, כדי שבנייה חוזרת לא תשכפל מקטעים.
    (ריקון במקום מחיקה נוסה ונפסל: ממשק פתוח בתהליך אחר המשיך לראות אוסף ריק וענה "אין מידע" בשקט.)"""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    name = f"clinic_{strategy}"
    if reset and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return client.get_or_create_collection(name)


def build_index(strategy):
    """חיתוך המסמכים, יצירת Embeddings ושמירה ב-Chroma. מחזיר את מספר המקטעים."""
    nodes = [n for n in SPLITTERS[strategy]().get_nodes_from_documents(load_documents())
             if len(n.get_content().strip()) >= MIN_CHUNK_CHARS]
    store = ChromaVectorStore(chroma_collection=_collection(strategy, reset=True))
    VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store))
    return len(nodes)


def load_index(strategy):
    """טוען אינדקס קיים מ-Chroma, בלי לחשב Embeddings מחדש."""
    collection = _collection(strategy)
    if collection.count() == 0:
        raise RuntimeError(f"האינדקס '{strategy}' ריק – יש להריץ קודם: python ingest.py")
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))


def ask(index, question, strategy, category=None):
    """אחזור + יצירת תשובה. מחזיר (תשובה, רשימת מקורות). category=None מחפש בכל המאגר."""
    filters = MetadataFilters(filters=[ExactMatchFilter(key="category", value=category)]) if category else None
    engine = index.as_query_engine(
        similarity_top_k=TOP_K[strategy], filters=filters, text_qa_template=QA_PROMPT,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=MIN_SCORE[strategy])],
    )
    try:
        response = engine.query(question)
    except genai_errors.APIError as e:
        if e.code not in (429, 500, 503, 504):  # עומס/מכסה/תקלה זמנית; שגיאות אחרות (כמו מפתח שגוי) נזרקות כרגיל
            raise
        return SERVICE_BUSY, []
    answer = str(response).strip()
    if not response.source_nodes or NO_ANSWER_TOKEN in answer:  # אין מקטע מעל הסף, או שהמודל לא מצא תשובה
        return random.choice(NO_ANSWER_MESSAGES), []  # לא מציגים מקורות שהתשובה לא התבססה עליהם
    sources = [
        {"file": n.metadata["file_name"], "category": n.metadata["category"],
         "score": round(n.score, 3), "text": n.get_content()}
        for n in response.source_nodes
    ]
    return answer, sources
