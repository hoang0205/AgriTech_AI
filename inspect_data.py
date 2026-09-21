from rag_pipeline import VectorDatabase

vdb = VectorDatabase()
collection = vdb.collection

# 1. Lấy thử 10 chunk đầu tiên xem nội dung
data = collection.get(limit=10, include=["documents", "metadatas"])

print(f"=== TỔNG SỐ CHUNKS HIỆN CÓ: {collection.count()} ===\n")
for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
    print(f"[{i+1}] Nguồn: {meta}")
    print(f"Nội dung: {doc[:150]}...\n" + "-"*40)

# 2. Thử tìm kiếm đúng từ khóa 'Dưa hấu' xem ChromaDB trả về cái gì
print("\n=== THỬ TÌM KIẾM 'Dưa hấu: Có thể làm những món gì' ===")
results = vdb.search_context("Dưa hấu: Có thể làm những món gì", top_k=3)
for r in results:
    print(f"- {r}\n")