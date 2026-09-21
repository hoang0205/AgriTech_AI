import io
import os
import re
import sqlite3
from typing import Dict, List
import uuid
import chromadb
from chromadb.utils import embedding_functions
import docx2txt
import ollama
from pypdf import PdfReader


class DocumentProcessor:

  def __init__(self, max_words: int = 350, overlap_words: int = 0):
    self.max_words = max_words

  def chunk_markdown(self, content: str) -> List[Dict[str, str]]:
      cleaned_content = (
          content.replace("\xa0", " ")
          .replace("\r\n", "\n")
          .replace("\r", "\n")
      )
      cleaned_content = cleaned_content.replace("\\", "")

      lines = cleaned_content.split("\n")
      chunks_data = []

      current_product = "Chung"
      current_heading = ""
      current_body = []

      for line in lines:
        stripped = line.strip()

        if not stripped or stripped == "---":
          continue

        if stripped.startswith("# ") and not stripped.startswith("## "):
          if current_heading and current_body:
            chunk_text = (
                f"# {current_product}\n{current_heading}\n"
                + "\n".join(current_body)
            )
            chunks_data.append({
                "text": chunk_text.strip(),
                "product": current_product.lower(),
            })
            current_heading = ""
            current_body = []

          current_product = stripped.replace("# ", "").strip()

        elif stripped.startswith("## "):
          if current_heading and current_body:
            chunk_text = (
                f"# {current_product}\n{current_heading}\n"
                + "\n".join(current_body)
            )
            chunks_data.append({
                "text": chunk_text.strip(),
                "product": current_product.lower(),
            })
            current_body = []

          current_heading = stripped
        else:
          current_body.append(stripped)

      if current_heading and current_body:
        chunk_text = (
            f"# {current_product}\n{current_heading}\n" + "\n".join(current_body)
        )
        chunks_data.append(
            {"text": chunk_text.strip(), "product": current_product.lower()}
        )

      return chunks_data


class ChatDatabase:

  def __init__(self, db_path: str = "chat_history.db"):
    self.db_path = db_path
    self._init_table()

  def _init_table(self):
    with sqlite3.connect(self.db_path) as conn:
      conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
      conn.commit()

  def add_message(self, session_id: str, role: str, content: str):
    with sqlite3.connect(self.db_path) as conn:
      conn.execute(
          "INSERT INTO chat_messages (session_id, role, content) VALUES (?,"
          " ?, ?)",
          (session_id, role, content),
      )
      conn.commit()

  def get_messages(self, session_id: str, limit: int = None) -> List[dict]:
    with sqlite3.connect(self.db_path) as conn:
      conn.row_factory = sqlite3.Row
      if limit:
        cursor = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {"role": r["role"], "content": r["content"]}
            for r in reversed(rows)
        ]
      else:
        cursor = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ?"
            " ORDER BY id ASC",
            (session_id,),
        )
        return [
            {"role": r["role"], "content": r["content"]}
            for r in cursor.fetchall()
        ]

  def clear_session(self, session_id: str):
    with sqlite3.connect(self.db_path) as conn:
      conn.execute(
          "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
      )
      conn.commit()


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

  def add_markdown_chunks(
      self, chunks_data: List[Dict[str, str]], source_name: str = "doc"
  ):
    if not chunks_data:
      return
    ids = [
        f"{source_name}_{uuid.uuid4().hex[:6]}_{i}"
        for i in range(len(chunks_data))
    ]
    documents = [c["text"] for c in chunks_data]
    metadatas = [
        {"source": source_name, "product": c["product"]} for c in chunks_data
    ]
    self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

  def search_context(
      self, query: str, product_name: str = None, top_k: int = 5
  ) -> str:
    try:
      if product_name:
        prod_kw = product_name.strip().lower()
        for prefix in [
            "quả ",
            "trái ",
            "củ ",
            "hạt ",
            "thịt ",
            "cá ",
            "con ",
            "rau ",
        ]:
          if prod_kw.startswith(prefix):
            prod_kw = prod_kw[len(prefix) :].strip()

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"product": {"$contains": prod_kw}},
        )
        if results["documents"] and results["documents"][0]:
          return "\n\n".join(results["documents"][0])

      results = self.collection.query(query_texts=[query], n_results=top_k)
      if results["documents"] and results["documents"][0]:
        return "\n\n".join(results["documents"][0])
      return ""
    except Exception as e:
      print(f"Lỗi Vector Search: {e}")
      return ""


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
  filename_lower = filename.lower()
  if filename_lower.endswith(".pdf"):
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    return "\n\n".join([p.extract_text() or "" for p in pdf_reader.pages])
  elif filename_lower.endswith(".docx"):
    return docx2txt.process(io.BytesIO(file_bytes))
  elif filename_lower.endswith((".txt", ".md")):
    return file_bytes.decode("utf-8")
  raise ValueError("Định dạng không được hỗ trợ.")


class RAGChatbot:

  def __init__(
      self, model_name: str = "qwen2.5:3b", db_path: str = "chat_history.db"
  ):
    self.model_name = model_name
    self.db = ChatDatabase(db_path)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    self.client = ollama.Client(host=ollama_host)

  def generate_answer(
      self,
      query: str,
      product_name: str,
      context: str,
      session_id: str = "default",
  ) -> str:
    system_prompt = f"""
Bạn là chuyên gia tư vấn nông sản và ẩm thực của AgriTech.
Nhiệm vụ: Trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO NGỮ CẢNH DƯỚI ĐÂY.

NGỮ CẢNH TÀI LIỆU:
{context}

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin có trong phần 'NGỮ CẢNH TÀI LIỆU' để trả lời.
2. Nếu trong ngữ cảnh KHÔNG nhắc đến các món ăn của sản phẩm '{product_name}', hoặc ngữ cảnh bị lệch sang sản phẩm khác (như thịt, cá...), TUYỆT ĐỐI KHÔNG tự bịa ra món ăn kết hợp phi lý.
3. Hãy trả lời thẳng thắn: 'Hiện tại dữ liệu của AgriTech chưa có thông tin công thức nấu ăn cho sản phẩm này.'
"""
    history_context = self.db.get_messages(session_id, limit=8)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_context)
    messages.append({"role": "user", "content": query})

    response = self.client.chat(
        model=self.model_name,
        messages=messages,
        stream=False,
        options={
            "num_gpu": 0,
            "temperature": 0.6,
            "repeat_penalty": 1.15,
        },
    )

    full_reply = response.get("message", {}).get("content", "")

    self.db.add_message(session_id, "user", query)
    self.db.add_message(session_id, "assistant", full_reply)
    return full_reply

  def generate_answer_stream(
      self,
      query: str,
      product_name: str,
      context: str,
      session_id: str = "default",
  ):
    system_prompt = f"""
Bạn là chuyên gia tư vấn nông sản và ẩm thực của AgriTech.
Nhiệm vụ: Trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO NGỮ CẢNH DƯỚI ĐÂY.

NGỮ CẢNH TÀI LIỆU:
{context}

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin có trong phần 'NGỮ CẢNH TÀI LIỆU' để trả lời.
2. Nếu trong ngữ cảnh KHÔNG nhắc đến các món ăn của sản phẩm '{product_name}', hoặc ngữ cảnh bị lệch sang sản phẩm khác (như thịt, cá...), TUYỆT ĐỐI KHÔNG tự bịa ra món ăn kết hợp phi lý.
3. Hãy trả lời thẳng thắn: 'Hiện tại dữ liệu của AgriTech chưa có thông tin công thức nấu ăn cho sản phẩm này.'
"""
    history_context = self.db.get_messages(session_id, limit=8)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_context)
    messages.append({"role": "user", "content": query})

    response_stream = self.client.chat(
        model=self.model_name,
        messages=messages,
        stream=True,
        options={
            "num_gpu": 0,
            "temperature": 0.6,
            "repeat_penalty": 1.15,
        },
    )

    accumulated = []
    for chunk in response_stream:
      content = chunk.get("message", {}).get("content", "")
      if content:
        accumulated.append(content)
        yield content

    full_reply = "".join(accumulated)

    self.db.add_message(session_id, "user", query)
    self.db.add_message(session_id, "assistant", full_reply)