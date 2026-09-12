import io
import os
import re
from typing import Dict, List
import uuid
import chromadb
from chromadb.utils import embedding_functions
import docx2txt
import ollama
from pypdf import PdfReader


class DocumentProcessor:

  def __init__(self, max_words: int = 80, overlap_words: int = 15):
    self.max_words = max_words
    self.overlap_words = overlap_words

  def _split_into_sentences(self, text: str) -> List[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    units = []
    for line in lines:
      if line.startswith(("-", "*", "+", "•")) or len(line.split()) <= 20:
        units.append(line)
      else:
        sentences = re.split(r"(?<=[.!?])\s+", line)
        units.extend([s.strip() for s in sentences if s.strip()])
    return units

  def chunk_text(self, content: str) -> List[str]:
    semantic_units = self._split_into_sentences(content)
    chunks = []
    current_chunk = []
    current_length = 0

    for unit in semantic_units:
      unit_words = unit.split()
      unit_length = len(unit_words)

      if current_length + unit_length > self.max_words and current_chunk:
        chunks.append(" ".join(current_chunk))
        overlap_pool = " ".join(current_chunk).split()
        if self.overlap_words > 0 and len(overlap_pool) > self.overlap_words:
          current_chunk = [
              " ".join(overlap_pool[-self.overlap_words :]) + "..."
          ]
          current_length = self.overlap_words
        else:
          current_chunk = []
          current_length = 0

      current_chunk.append(unit)
      current_length += unit_length

    if current_chunk:
      chunks.append(" ".join(current_chunk))
    return chunks

  def load_and_chunk(self, file_path: str) -> List[str]:
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"Không tìm thấy: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
      return self.chunk_text(f.read())


class VectorDatabase:

  def __init__(self, persist_dir: str = "./chroma_db"):
    self.client = chromadb.PersistentClient(path=persist_dir)
    self.embedding_fn = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    self.collection = self.client.get_or_create_collection(
        name="agritech_knowledge", embedding_function=self.embedding_fn
    )

  def add_chunks(self, chunks: List[str], source_name: str = "doc"):
    if not chunks:
      return
    ids = [
        f"{source_name}_{uuid.uuid4().hex[:6]}_{i}" for i in range(len(chunks))
    ]
    self.collection.add(documents=chunks, ids=ids)
    print(f"✅ Đã thêm {len(chunks)} chunks từ nguồn '{source_name}'")

  def search_context(self, query: str, top_k: int = 3) -> str:
    try:
      results = self.collection.query(query_texts=[query], n_results=top_k)
      retrieved_docs = results["documents"][0]
      return "\n\n".join(retrieved_docs)
    except Exception:
      return ""


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
  """Trích xuất text từ nhiều định dạng file khác nhau."""
  filename_lower = filename.lower()

  if filename_lower.endswith(".pdf"):
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    text_pages = [
        page.extract_text() or ""
        for page in pdf_reader.pages
        if page.extract_text()
    ]
    return "\n\n".join(text_pages)

  elif filename_lower.endswith(".docx"):
    return docx2txt.process(io.BytesIO(file_bytes))

  elif filename_lower.endswith((".txt", ".md")):
    return file_bytes.decode("utf-8")

  else:
    raise ValueError("Định dạng file không được hỗ trợ.")


class RAGChatbot:

  def __init__(self, model_name: str = "qwen2.5:7b"):
    self.model_name = model_name
    self.sessions: Dict[str, List[dict]] = {}

  def generate_answer_stream(
      self,
      query: str,
      product_name: str,
      context: str,
      session_id: str = "default",
  ):
    if session_id not in self.sessions:
      self.sessions[session_id] = []

    system_prompt = f"""Bạn là Chuyên gia Cố vấn Nông sản & Dinh dưỡng của sàn thương mại điện tử AgriTech.
Khách hàng đang xem sản phẩm: "{product_name}".

QUY TẮC:
1. Ưu tiên sử dụng thông tin trong "CẨM NANG THAM KHẢO" nếu có nội dung liên quan.
2. Nếu Cẩm nang không đề cập hoặc thiếu thông tin, hãy dùng kiến thức chuyên sâu của bạn về ẩm thực, dinh dưỡng, mẹo chọn quả và bảo quản để giải đáp tận tình.
3. Không trả lời hoặc can thiệp vào giá bán, số lượng tồn kho hay đặt hàng (đây là việc giữa người mua và người bán tự trao đổi).
4. Trả lời bằng tiếng Việt tự nhiên, thân thiện và mạch lạc.

CẨM NANG THAM KHẢO:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(self.sessions[session_id][-6:])
    messages.append({"role": "user", "content": query})

    response_stream = ollama.chat(
        model=self.model_name, messages=messages, stream=True
    )

    accumulated_text = []
    for chunk in response_stream:
      content = chunk.get("message", {}).get("content", "")
      if content:
        accumulated_text.append(content)
        yield content

    full_reply = "".join(accumulated_text)
    self.sessions[session_id].append({"role": "user", "content": query})
    self.sessions[session_id].append(
        {"role": "assistant", "content": full_reply}
    )

  def generate_answer(
      self,
      query: str,
      product_name: str,
      context: str,
      session_id: str = "default",
  ) -> str:
    return "".join(
        self.generate_answer_stream(query, product_name, context, session_id)
    )