import sys
from src.settings import settings
from pathlib import Path
from src.indexer import build_index
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if not settings.kb_path.exists():
    raise FileNotFoundError(f"База знаний не найдена: {settings.kb_path}")

if __name__ == "__main__":
    build_index()