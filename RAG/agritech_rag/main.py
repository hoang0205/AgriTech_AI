from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_pipeline import (
    DocumentProcessor,
    RAGChatbot,
    VectorDatabase,
    extract_text_from_file,
)

app = FastAPI(title="AgriTech AI Advisor API (Qwen 2.5)")

vector_db = VectorDatabase()
chatbot = RAGChatbot(model_name="qwen2.5:3b")
processor = DocumentProcessor(max_words=80, overlap_words=15)


class ProductAdviceRequest(BaseModel):
  query: str
  product_name: str
  user_id: str = "guest"
  session_id: str | None = None


@app.post("/api/chat")
async def chat_endpoint(request: ProductAdviceRequest):
  try:
    session_id = (
        request.session_id or f"{request.user_id}_{request.product_name}"
    )

    search_query = f"{request.product_name}: {request.query}"
    knowledge_context = vector_db.search_context(search_query, top_k=3)

    answer = chatbot.generate_answer(
        query=request.query,
        product_name=request.product_name,
        context=knowledge_context,
        session_id=session_id,
    )

    return {
        "status": "success",
        "session_id": session_id,
        "product_name": request.product_name,
        "answer": answer,
        "context_used": knowledge_context,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
  try:
    allowed_extensions = (".txt", ".md", ".pdf", ".docx")
    if not file.filename.lower().endswith(allowed_extensions):
      raise HTTPException(
          status_code=400,
          detail="Chỉ hỗ trợ các định dạng tài liệu: .txt, .md, .pdf, .docx",
      )

    raw_bytes = await file.read()

    try:
      text_content = extract_text_from_file(raw_bytes, file.filename)
    except UnicodeDecodeError:
      raise HTTPException(
          status_code=400,
          detail="Lỗi mã hóa: File văn bản cần được lưu dưới chuẩn UTF-8.",
      )
    except Exception as err:
      raise HTTPException(
          status_code=400, detail=f"Không thể đọc nội dung từ file: {str(err)}"
      )

    if not text_content.strip():
      raise HTTPException(
          status_code=400,
          detail="Không tìm thấy nội dung văn bản trong file đã tải lên.",
      )

    chunks = processor.chunk_text(text_content)

    clean_name = file.filename.rsplit(".", 1)[0]
    vector_db.add_chunks(chunks, source_name=clean_name)

    return {
        "status": "success",
        "message": (
            f"Đã nạp thành công tài liệu '{file.filename}' vào cơ sở dữ liệu!"
        ),
        "chunks_added": len(chunks),
    }

  except HTTPException:
    raise
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ProductAdviceRequest):
  session_id = request.session_id or f"{request.user_id}_{request.product_name}"

  search_query = f"{request.product_name}: {request.query}"
  knowledge_context = vector_db.search_context(search_query, top_k=3)

  def token_generator():
    for text_token in chatbot.generate_answer_stream(
        query=request.query,
        product_name=request.product_name,
        context=knowledge_context,
        session_id=session_id,
    ):
      yield text_token

  return StreamingResponse(
      token_generator(), media_type="text/plain; charset=utf-8"
  )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)