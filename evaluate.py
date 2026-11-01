"""בדיקות המערכת. שימוש: python evaluate.py [--retrieval-only]  →  results/eval_results.md (או eval_retrieval.md)
חלק 1 (בונוס): מדדי אחזור Hit-rate ו-MRR לשתי אסטרטגיות החיתוך, ציון הדמיון הגבוה ביותר לכל שאלה (לכיול הסף)
ומספר המקטעים הזעירים. חלק 2: תשובות מלאות לכל השאלות, כולל מקורות, זמן ומספר טוקנים."""
import sys
import time

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from rag_core import MIN_CHUNK_CHARS, ROOT, SPLITTERS, TOP_K, _collection, ask, configure, load_index

TESTS = [  # (שאלה, המסמך שמכיל את התשובה; None = התשובה לא נמצאת במאגר)
    ("כמה עולה הלבנת שיניים?", "price_list.md"),
    ("מה שעות הפעילות ביום שישי?", "hours_contact.md"),
    ("כמה זמן מראש צריך לבטל תור כדי לא לשלם דמי ביטול?", "appointments_cancellations.md"),
    ("מה מותר לאכול אחרי עקירה?", "care_instructions.md"),
    ("מאיזה גיל מומלץ להביא ילד לבדיקה ראשונה?", "children.md"),
    ("מה עושים אם שן נשברה בשבת?", "emergency.md"),
    ("האם אפשר לשלם בתשלומים?", "insurance_payments.md"),
    # שאלות "קשות": התשובה במאגר, אבל בניסוח שונה מהמסמך
    ("אפשר להגיע עם כלב נחייה?", "about.md"),
    ("מי עושה עקירה של שן בינה?", "treatments.md"),
    ("יש הנחה לפנסיונרים?", "price_list.md"),
    ("איפה מחנים?", "hours_contact.md"),
    ("כמה רופאים יש?", "about.md"),  # דורשת ספירה – המידע לא כתוב במפורש
    ("האם המרפאה מבצעת ניתוחי לייזר בעיניים?", None),
    ("כמה עולה השתלת שיער?", None),
]
QA_STRATEGY = "markdown"
RETRIEVAL_ONLY = "--retrieval-only" in sys.argv  # חלק 1 בלבד, בלי קריאות ל-LLM
DELAY = 4  # שניות בין קריאות ל-LLM, כדי לא לחרוג ממגבלת הקצב של השכבה החינמית

tokens = TokenCountingHandler()  # סופר טוקנים בקירוב (tokenizer של OpenAI, לא של Gemini)
Settings.callback_manager = CallbackManager([tokens])  # לפני configure(), כדי שהמודלים ייצרו איתו
configure()
indexes = {s: load_index(s) for s in SPLITTERS}

# ---- חלק 1: אחזור בלבד (Embeddings לשאלות, בלי LLM) ----
ranks, top_scores = {}, {}
tiny_retrieved = dict.fromkeys(SPLITTERS, 0)  # כמה מקטעים זעירים נכנסו ל-Top-K, בכל השאלות יחד
for s, index in indexes.items():
    retriever = index.as_retriever(similarity_top_k=TOP_K[s])
    for q, expected in TESTS:
        nodes = retriever.retrieve(q)
        files = [n.metadata["file_name"] for n in nodes]
        ranks[s, q] = files.index(expected) + 1 if expected in files else None
        top_scores[s, q] = nodes[0].score
        tiny_retrieved[s] += sum(len(n.get_content().strip()) < MIN_CHUNK_CHARS for n in nodes)
answerable = [q for q, f in TESTS if f]
lines = [f"# תוצאות בדיקה", "", f"## 1. מדדי אחזור (Top-K: {TOP_K}, {len(answerable)} שאלות שיש להן תשובה)", "",
         f"| אסטרטגיה | Hit-rate | MRR | מקטעים באינדקס | מקטעים זעירים (<{MIN_CHUNK_CHARS} תווים) באינדקס "
         f"| מקטעים זעירים שאוחזרו |", "|---|---|---|---|---|---|"]
for s in SPLITTERS:
    r = [ranks[s, q] for q in answerable]
    docs = _collection(s).get(include=["documents"])["documents"]
    lines.append(f"| {s} | {sum(x is not None for x in r) / len(r):.2f} | {sum(1 / x for x in r if x) / len(r):.2f} "
                 f"| {len(docs)} | {sum(len(d.strip()) < MIN_CHUNK_CHARS for d in docs)} | {tiny_retrieved[s]} |")
lines += ["", "### פירוט: מיקום המסמך הנכון וציון הדמיון הגבוה ביותר", "",
          "| שאלה | מסמך צפוי | " + " | ".join(f"{s}: מיקום / ציון" for s in SPLITTERS) + " |",
          "|---|---|" + "---|" * len(SPLITTERS)]
for q, expected in TESTS:
    cells = [f"{ranks[s, q] or '—'} / {top_scores[s, q]:.3f}" for s in SPLITTERS]
    lines.append(f"| {q} | {expected or 'לא במאגר'} | " + " | ".join(cells) + " |")

# ---- חלק 2: תשובות מלאות ----
if not RETRIEVAL_ONLY:
    lines += ["", f"## 2. שאלות ותשובות (אסטרטגיה: {QA_STRATEGY})"]
for i, (q, expected) in enumerate([] if RETRIEVAL_ONLY else TESTS, 1):
    print(f"[{i}/{len(TESTS)}] {q}")
    tokens.reset_counts()
    t = time.time()
    answer, sources = ask(indexes[QA_STRATEGY], q, QA_STRATEGY)
    lines += ["", f"### {i}. {q}", f"**תשובה:** {answer}", "",
              f"מסמך צפוי: `{expected or 'לא במאגר'}` · זמן: {time.time() - t:.1f} שניות · "
              f"טוקנים בפרומפט (קירוב): {tokens.prompt_llm_token_count}", "", "**מקורות:**"]
    lines += [f"- {s['file']} (דמיון {s['score']})" for s in sources] or ["- לא הוצגו מקורות"]
    time.sleep(DELAY)

out = ROOT / "results" / ("eval_retrieval.md" if RETRIEVAL_ONLY else "eval_results.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"נשמר: {out}")
