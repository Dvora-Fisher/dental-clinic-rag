"""בונה את האינדקסים ב-Chroma. שימוש: python ingest.py [sentence|markdown]  (ברירת מחדל: שניהם)"""
import sys

from rag_core import SPLITTERS, build_index, configure

configure()
for strategy in sys.argv[1:] or SPLITTERS:
    print(f"{strategy}: נשמרו {build_index(strategy)} מקטעים")
