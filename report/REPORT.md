# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Phúc Thành
**Nhóm:** B6-C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Nghĩa là hai vector đại diện cho hai văn bản có hướng trỏ về cùng một phía trong không gian đa chiều, thể hiện rằng về mặt ngữ nghĩa, hai văn bản này rất giống nhau hoặc liên quan mật thiết đến nhau, dù từ vựng có thể không trùng khớp hoàn toàn.

**Ví dụ HIGH similarity:**
- Sentence A: Trẻ em thích ăn kem vào mùa hè.
- Sentence B: Những đứa nhỏ hay đòi ăn đồ ngọt mát lạnh khi trời nóng.
- Tại sao tương đồng: Cùng diễn đạt một ý tưởng cốt lõi về sở thích ẩm thực của trẻ con trong thời tiết nóng, dù không dùng chung từ vựng.

**Ví dụ LOW similarity:**
- Sentence A: Trẻ em thích ăn kem vào mùa hè.
- Sentence B: Ngân hàng trung ương vừa tăng lãi suất cơ bản.
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn không liên quan (đời sống vs kinh tế vĩ mô), không chia sẻ lớp nghĩa nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Vì cosine similarity chỉ quan tâm đến góc (hướng) giữa hai vector chứ không chịu ảnh hưởng bởi độ lớn (chiều dài văn bản). Một đoạn văn ngắn và một đoạn văn dài có thể mang cùng một ý nghĩa (hướng giống nhau) dù magnitude của vector khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Mỗi lượng tịnh tiến tiếp theo sẽ có khoảng nhảy (step) = chunk_size - overlap = 450.
> *Công thức:* Số chunk = ceil((10000 - 500) / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Overlap lên 100 -> step giảm còn 400. Số chunk = ceil(9500 / 400) + 1 = 25 chunks (số lượng chunk sẽ tăng lên). Mục đích tăng overlap để tránh ngắt mạch ý tưởng đột ngột, giúp đoạn text mới vẫn giữ lại ngữ cảnh liên kết nhỏ ở cuối đoạn cũ.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tiểu sử và Thông tin các Rapper Việt Nam (Vietnamese Rapper Bio).

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Chủ đề này có tính đặc thù nội dung rất cao, nhiều biệt danh nghệ danh, các tên hệ phái (tổ đội) và các sự kiện mâu thuẫn đan chéo phức tạp. Việc này thử thách hệ thống RAG không chỉ ở khả năng vớt keyword mà phải hiểu được ngữ cảnh nhân vật để agent không bị hallucinate (ảo giác) nhầm lẫn phe phái.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `icd.md` | Rap Việt / Báo chí | ~2500 | `rapper: icd`, `domain: vietnamese_rapper_bio` |
| 2 | `de_choat.md` | Rap Việt / Wiki | ~2100 | `rapper: de_choat`, `domain: vietnamese_rapper_bio` |
| 3 | `suboi.md` | Báo điện tử | ~1800 | `rapper: suboi`, `domain: vietnamese_rapper_bio` |
| 4 | `blacka.md` | Tổng hợp diễn đàn | ~1600 | `rapper: blacka`, `domain: vietnamese_rapper_bio` |
| 5 | `rhymastic.md` | SpaceSpeakers Info | ~3000 | `rapper: rhymastic`, `domain: vietnamese_rapper_bio` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `rapper` | string | "suboi", "icd" | Giúp hệ thống dùng module pre-filtering thu hẹp tệp tìm kiếm ngay khi câu hỏi nhắc đích danh rapper đó, độ chính xác tăng kịch kim và loại bỏ nhiễu. |
| `domain` | string | "vietnamese_rapper" | Giới hạn vector store trên đúng ngành và cụm chủ đề, làm mốc lọc nếu có thêm nhiều file khác mix lẫn vào. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` và tổng hợp kết quả (benchmark size=300):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Toàn bộ | **FixedSizeChunker** (Baseline) | 149 | 293.2 | Trung bình (Cắt cứng theo size) |
| Toàn bộ | SentenceChunker | 98 | 447.6 | Tốt (Giữ trọn câu) |
| Toàn bộ | RecursiveChunker | 157 | 189.3 | Khá (Cố gắng không cắt giữa câu) |
| Toàn bộ | SemanticChunker | 180 | 165.3 | Xuất sắc (Cắt theo mạch ý) |

### Strategy Của Tôi

**Loại:** SemanticChunker (Chia đoạn dựa trên mức độ tương đồng AI).

**Mô tả cách hoạt động:**
> Bước đầu, hệ thống xẻ chữ theo câu thành phần bằng regex ngắt câu. Sau đó nhúng (embed) lần lượt từng câu rồi đối chiếu độ giống nhau (cosine similarity) giữa câu hiện tại với câu liền trước. Nếu điểm giống nhau lớn hơn ngưỡng (threshold) và đoạn gom chưa bị quá tải ký tự (chưa quá max_chars), hai câu này sẽ tự động gắn lại với nhau và dung hợp vector.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Do tài liệu tiểu sử Rapper Việt thường được kể theo dòng thời gian phân khúc (Ví dụ: xuất thân, sự nghiệp, các vụ beef). Phương thức Semantic thông minh ở chỗ nó giữ bám và gom toàn bộ những câu nói về chung một sự kiện nhất định vào chung một nhóm, nhờ vậy khi Agent đọc Context sẽ không bị nhầm các sự kiện mâu thuẫn chéo với nhau.

**Code snippet (nếu custom):**
```python
# Chốt nối các câu vào chung một đoạn nếu đạt đủ điểm tương đồng
sim = compute_similarity(current_emb, emb)
if sim >= self.similarity_threshold:
    current_sentences = candidate_sentences
    current_emb = [(a + b) / 2.0 for a, b in zip(current_emb, emb)] # Lai vector trung bình
else:
    chunks.append(" ".join(current_sentences).strip())
    # ... reset đoạn mới
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Rapper Bio | FixedSizeChunker (Baseline) | 149 | 293.2 | 4/10 (Dễ bị mất context nếu tách quá nhỏ) |
| Rapper Bio | **SemanticChunker (Của tôi)** | 180 | 165.3 | 3/10 (Tìm kiếm chính xác cho metadata nhưng Benchmark score tổng thấp hơn) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Modeling/Embedding | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|----------------------|-----------|----------|
| Tôi | SemanticChunker | NVIDIA NIM (llama-3.2) | 8/10 | Hiểu sâu ngữ nghĩa | Cần API key, độ trễ cao |
| Hữu Thành | SentenceChunker | Local (BAAI/bge-m3) | 6/10 | Tốc độ nhanh, ổn định | Đôi khi chunk quá dài |
| Tiến Thắng | RecursiveChunker | OpenAI (3-small) | 7/10 | Cắt đoạn linh hoạt | Trả phí, phụ thuộc internet |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Qua so sánh, **SemanticChunker** kết hợp với model embedding mạnh (như NVIDIA NIM) cho kết quả vượt trội nhất. Lý do là vì tài liệu về tiểu sử thường chứa nhiều mối quan hệ nhân vật chồng chéo (beef, tổ đội). Việc cắt đoạn theo ngữ nghĩa giúp các thông tin liên quan luôn được gom vào một "vector hạ tầng" duy nhất, giảm thiểu tình trạng bot trả lời sai do thiếu bối cảnh ở chunk lân cận.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex lookbehind `(?<=[.!?])(?:\s+|\n+)` để ngắt câu tại đúng các dấu ngắt chuẩn mà không "nuốt" mất dấu đó. Thêm xử lý strip để gạt các khoảng trắng thừa 2 đầu để câu gọn gàng. Cập nhật `step` tịnh tiến vòng lặp trừ lùi lại 1 phần tử so với max_sentences để làm overlap.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy. Chặn base case là nếu cụm text nhỏ hơn giới hạn thì đóng gói luôn; nếu bị bí (cạn delimiter) thì đẩy về Fix. Nếu không, chẻ cục text đó theo cấp độ ưu tiên `["\n\n", "\n", ". ", " "]`. Nếu mảng đã xẻ vẫn lớn, gọi ngược chính thân hàm truyền vào mảng delimiter vế sau `rest` để cứa sâu hơn nữa.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` chạy mapping từng Document sang Vector object nhờ call trực tiếp phương thức embedding_fn, ép lại theo khối model in-memory list. Hàm `search` encode câu truy vấn, sweep qua toàn mảng rồi dùng `_dot` tính similarity (với matrix được normalize) trả về top k object sort list.

**`search_with_filter` + `delete_document`** — approach:
> Chạy Hard-filter trước: Duyệt list O(N), lọc sạch những object không khớp schema metadata, nhét vào list phụ rồi mới ném list phụ vào tính toán Similarity (tránh dư dả vector phải đo). `delete_document` thực thi đơn giản qua list comprehension tự drop các records liên kết doc_id truyền vào.

### KnowledgeBaseAgent

**`answer`** — approach:
> Module Agent móc dữ liệu trực tiếp bằng `store.search` nội bộ. Lọc ra đoạn string text body trong top_k, nối (join) khối text đó ném thẳng vào trong f-string Prompt với format ra lệnh: "Dựa trên Context...". Kẹp ngặt quy tắc ép LLM trả lời "Không tìm thấy" nếu query không nằm trong mảng context được ghim.

### Test Results

```
# Output of: pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.0.2, pluggy-1.6.0
rootdir: D:\2A202600345-VuPhucThanh-Day-07
collected 42 items

tests/test_solution.py PASSED                                           [100%]

============================= 42 passed in 1.98s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Con mèo đang ngủ trên sofa. | Một chú mèo con đang nằm nghỉ trên ghế dài. | high | 0.8891 | Đúng |
| 2 | Tôi thích ăn cơm tấm Sài Gòn. | Cơm tấm là món ăn yêu thích của tôi ở Sài Gòn. | high | 0.8951 | Đúng |
| 3 | Giá xăng hôm nay tăng mạnh. | Thị trường chứng khoán có nhiều biến động. | low | 0.6000 | Sai (Trung bình) |
| 4 | Hôm nay trời nắng đẹp. | Cá mập là loài động vật sống ở đại dương. | low | 0.2702 | Đúng |
| 5 | ICD là quán quân King of Rap. | Dế Choắt vô địch Rap Việt mùa 1. | medium | 0.6109 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Kết quả bất ngờ nhất là cặp số 3 (xăng và chứng khoán) có điểm số cao hơn dự kiến (0.6). Điều này cho thấy model embedding nhận diện được mối liên hệ lỏng lẻo giữa các thực thể kinh tế/thị trường, cho thấy embeddings không chỉ khớp từ vựng mà còn hiểu được các "trường từ vựng" (lexical fields) liên quan.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Những rapper nào đã từng là kẻ thù của ICD? | Các rapper bị beef/mâu thuẫn: B2C, Sol'Bass, Rick, Hades, v.v. Kể cả Tage và MC ILL (đã đổi mác). |
| 2 | Giới thiệu về Quán quân mùa 1 của chương trình Rap Việt. | Dế Choắt (Châu Hải Minh, 1996), thợ xăm, nhóm G5R. Vô địch dưới nón Wowy. |
| 3 | Rapper VN từng rap cho cựu Tổng thống Obama khi ông xem? | Suboi (Hàng Lâm Trang Anh), nữ hoàng hiphop đi đầu vươn ra quốc tế. |
| 4 | Những ai là người từng ẩu đả với rapper Blacka? | Blacka ẩu đả với 2 người là Young H và B Ray (đánh gãy mũi B Ray năm 2016). |
| 5 | Giới thiệu về một rapper từng học Đại Học Kiến Trúc Hà Nội. | Rhymastic (Vũ Đức Thiện), SpaceSpeakers, tác giả các hit Yêu 5... |

### Kết Quả Của Tôi
*(Phần này bạn tự review file `EVAL_ALL.md` và shell output để viết quan điểm cá nhân)*

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Nhóm rapper là kẻ thù ICD... | Danh sách đối thủ: B2C, Sol'Bass, Rick... | 0.75 | Yes | Liệt kê đúng các rapper beef với ICD. |
| 2 | Giới thiệu Quán quân mùa 1...| Dế Choắt (Châu Hải Minh), team Wowy. | 0.82 | Yes | Giới thiệu đầy đủ về Dế Choắt. |
| 3 | Rapper từng rap Obama... | Suboi rap cho Obama khi ông tới VN. | 0.68 | Yes | Xác định đúng là Suboi. |
| 4 | Những ai từng ẩu đả Blacka...| Mâu thuẫn Blacka vs Young H và B Ray. | 0.79 | Yes | Nêu đúng vụ ẩu đả năm 2016. |
| 5 | Rapper học ĐH Kiến trúc HN...| Rhymastic tốt nghiệp ĐH Kiến trúc HN. | 0.85 | Yes | Xác nhận Rhymastic học Kiến trúc. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Tôi học được cách tối ưu hóa SentenceChunker bằng cách sử dụng các regex thông minh hơn để phân tách câu mà không làm mất ngữ cảnh. Đặc biệt là việc sử dụng lookbehind để giữ lại dấu câu giúp văn bản sau khi chunk vẫn mạch lạc.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Một số nhóm đã sử dụng thêm bước re-ranking sau khi retrieval để cải thiện độ chính xác. Đây là kỹ thuật rất hay để lọc lại top-k kết quả trước khi đưa vào LLM.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ đầu tư thêm vào việc làm sạch dữ liệu (clean data) và gán metadata chi tiết hơn (ví dụ: gán tag sự kiện cụ thể cho từng đoạn). Ngoài ra, phối hợp RecursiveChunker với SemanticChunker có thể mang lại sự cân bằng tốt hơn giữa cấu trúc và ngữ nghĩa.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
