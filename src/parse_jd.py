import zipfile
import xml.etree.ElementTree as ET
import os

STATIC_JD_WORDS = [
    "embeddings", "retrieval", "ranking", "llms", "fine-tuning", "vector", 
    "search", "infrastructure", "pinecone", "weaviate", "qdrant", "milvus", 
    "opensearch", "elasticsearch", "faiss", "ndcg", "mrr", "map", "eval", "evaluation",
    "learning-to-rank", "xgboost", "python", "pytorch", "tensorflow", "scikit-learn",
    "rag", "sentence-transformers"
]

def parse_docx_text(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespaces
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for p in root.findall('.//w:p', ns):
                texts = [t.text for t in p.findall('.//w:t', ns) if t.text]
                if texts:
                    paragraphs.append(''.join(texts))
            return '\n'.join(paragraphs)
    except Exception:
        return ""

def get_jd_keywords(jd_path):
    if not jd_path or not os.path.exists(jd_path):
        return STATIC_JD_WORDS
        
    text = ""
    if jd_path.endswith(".docx"):
        text = parse_docx_text(jd_path)
    elif jd_path.endswith(".txt") or jd_path.endswith(".md"):
        try:
            with open(jd_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            pass
            
    if not text:
        return STATIC_JD_WORDS
        
    # Extract matching keywords from text
    found_keywords = []
    text_lower = text.lower()
    for w in STATIC_JD_WORDS:
        if w in text_lower:
            found_keywords.append(w)
            
    if len(found_keywords) < 5:
        return STATIC_JD_WORDS
    return found_keywords
