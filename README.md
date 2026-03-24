# Vietnamese RAG QA System

Hệ thống hỏi–đáp tài liệu tiếng Việt sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)**.

## Kiến trúc

```
documents/ (PDF)
     │
     ▼
Document Loader          ← Trích xuất + chunking văn bản tiếng Việt
     │
     ▼
Embedding Model          ← keepitreal/vietnamese-sbert (PhoBERT-based)
     │  dense vectors
     ▼
Qdrant Vector DB  ◄──── Docker container (lưu trữ bền vững)
     │
     ▼
RAG Pipeline             ← Tìm top-K đoạn liên quan + sinh câu trả lời
     │
     ▼
LLM (Gemini/OpenAI)      ← Sinh câu trả lời cuối cùng
     │
     ▼
FastAPI Backend  ◄──────  REST API
     │
     ▼
Frontend Web UI          ← Giao diện hỏi đáp
```

## Yêu cầu

- **Docker Desktop** (Windows/Mac/Linux)
- **API Key**: [Google Gemini](https://aistudio.google.com/app/apikey) (miễn phí) hoặc OpenAI

## Cài đặt & Chạy

### 1. Tạo file `.env`

```bash
cp .env.example .env
```

Mở `.env` và điền API key:
```
GEMINI_API_KEY=your_key_here
```

### 2. Thêm tài liệu PDF

Đặt file PDF vào thư mục `documents/`. Hiện có sẵn:
- `Luật-109-2025-QH15.pdf`
- `Luật-67-2025-QH15.pdf`

### 3. Khởi động hệ thống

```bash
docker-compose up --build -d
```

Lần đầu chạy sẽ mất **10–20 phút** để:
- Build Docker image
- Tải embedding model (~270 MB từ HuggingFace)

### 4. Nhập tài liệu vào Vector DB

**Cách 1 – qua giao diện web:**
- Mở http://localhost:3000
- Click **"Nhập tài liệu vào DB"**

**Cách 2 – qua API:**
```bash
curl -X POST http://localhost:8000/ingest
```

**Cách 3 – qua CLI (trong container):**
```bash
docker exec rag-backend python -m scripts.ingest
```

### 5. Sử dụng

| Địa chỉ | Chức năng |
|---------|-----------|
| http://localhost:3000 | Giao diện web hỏi đáp |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:6333/dashboard | Qdrant dashboard |

## Sử dụng giao diện web

1. **Danh sách tài liệu** – sidebar trái hiển thị các file PDF
2. **Nhập tài liệu** – click "Nhập tài liệu vào DB" trước khi đặt câu hỏi
3. **Chế độ truy vấn**:
   - `RAG` – tìm kiếm tài liệu + sinh câu trả lời (chính xác)
   - `Không RAG` – LLM trả lời từ kiến thức chung (có thể "bịa")
   - `So sánh` – xem song song cả hai kết quả

4. **Đánh giá** – click "Chạy đánh giá" để chạy batch trên `QA_TEST.xlsx`

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/health` | Kiểm tra trạng thái |
| GET | `/info` | Thông tin hệ thống |
| GET | `/documents` | Danh sách tài liệu |
| POST | `/ingest` | Nhập tài liệu vào DB |
| POST | `/query` | Hỏi đáp có RAG |
| POST | `/query/no-rag` | Hỏi đáp không RAG |
| POST | `/compare` | So sánh RAG vs không RAG |
| POST | `/evaluate` | Đánh giá batch từ QA_TEST.xlsx |

### Ví dụ gọi API

```bash
# Hỏi đáp có RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Luật này quy định về điều gì?", "top_k": 5}'

# So sánh RAG vs không RAG
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "Điều kiện để được cấp phép là gì?"}'
```

## File QA_TEST.xlsx

File Excel để đánh giá hệ thống, cần có cột:
- `câu hỏi` hoặc `question`
- `câu trả lời` hoặc `reference_answer`

## Đổi embedding model

Sửa trong `.env`:
```
# PhoBERT (chất lượng cao hơn, ~540MB)
EMBEDDING_MODEL=VoVanPhuc/sup-SimCSE-VietNamese-phobert-base

# Hoặc Multilingual E5 (~1.1GB)
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

Sau đó rebuild và nhập lại tài liệu:
```bash
docker-compose up --build -d
curl -X POST "http://localhost:8000/ingest?recreate=true"
```

## Dừng hệ thống

```bash
docker-compose down        # Dừng (giữ data)
docker-compose down -v     # Dừng + xóa toàn bộ data Vector DB
```

## Cấu trúc thư mục

```
RAG-VNQA/
├── documents/              ← Đặt PDF tài liệu vào đây
├── QA_TEST.xlsx            ← Bộ câu hỏi kiểm tra
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py         ← FastAPI app
│   │   ├── config.py       ← Cấu hình
│   │   ├── document_loader.py   ← Đọc PDF + chunking
│   │   ├── embeddings.py   ← PhoBERT/SBERT encoding
│   │   ├── vector_store.py ← Qdrant client
│   │   ├── rag_pipeline.py ← RAG pipeline + LLM
│   │   └── evaluator.py    ← Đánh giá EM/F1/ROUGE
│   └── scripts/
│       └── ingest.py       ← CLI ingest script
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    ├── index.html          ← Giao diện web
    ├── style.css
    └── app.js
```
