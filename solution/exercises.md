# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Model dùng ở đây là `meta/llama-3.2-11b-vision-instruct`. Ở temperature 0.0, 0.5 và 1.0, model đều trả lời mạch lạc, đúng tiếng Việt, đúng format sự thật về Việt Nam, chỉ khác nhau về chủ đề được chọn, temp càng cao thì chủ đề càng lạ, cho thấy model đang lấy mẫu xa dần khỏi câu trả lời xác suất cao nhất. Ở temperature 1.5, output hoàn toàn không sử dụng được: câu chữ trộn lẫn tiếng Việt, tiếng Anh và Trung Hàn gì đó, không còn thành câu cho thấy 1.5 đã vượt quá ngưỡng dùng được của model.
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, đúng chính sách/quy trình công ty và ít rủi ro hallucination, temperature thấp giúp model bám sát thông tin đã cho (system prompt, tài liệu tra cứu) và trả lời giống nhau cho cùng một câu hỏi, dễ kiểm soát chất lượng và audit hơn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng token đầu ra/ngày = 10.000 người dùng × 3 lần gọi × 350 token ≈ 10.500.000 token.
> Chi phí GPT-4o = 10.500.000/1000 × $0.010 ≈ **$105/ngày**.
> Chi phí GPT-4o-mini = 10.500.000/1000 × $0.0006 ≈ **$6.30/ngày**.
> → GPT-4o đắt hơn khoảng **16.7 lần** (đúng bằng tỉ lệ giá output 0.010/0.0006 trong `PRICING_PER_1K_TOKENS`).
> GPT-4o xứng đáng chi phí cao khi task cần suy luận phức tạp, độ chính xác cao (tư vấn pháp lý/y tế, sinh code, phân tích nhiều bước). GPT-4o-mini phù hợp cho các tác vụ đơn giản, khối lượng lớn như trả lời FAQ, phân loại yêu cầu, tóm tắt ngắn — nơi chất lượng mini đã đủ tốt và tiết kiệm 16x chi phí có ý nghĩa lớn ở quy mô 10.000 người dùng/ngày.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, model xưng hô gần gũi, dùng từ vựng đơn giản, và minh họa bằng ví dụ cụ thể, dễ hình dung. Với persona chuyên gia tài chính, model bỏ lời chào, dùng thuật ngữ kỹ thuật tiếng Anh kèm chú thích, và trình bày thoe kiểu học thuật hơn (liệt kê tính chất bằng số thứ tự, in đậm).System prompt khác nhau, định hình vai của model khác đã làm thay đổi giọng văn, mức độ kỹ thuật và cách chọn ví dụ.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn 138 từ: ước lượng theo "số từ / 0.75" ra 184 token, còn đếm thật bằng tiktoken ra 156 token, chênh khoảng 17.9%. 
Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì ký tự có dấu không nằm trong bộ ký tự phổ biến mà bộ mã hóa BPE của tiktoken được huấn luyện chủ yếu trên tiếng Anh, nhiều từ tiếng Việt bị tách thành 2–3 token thay vì 1 token trọn vẹn như một từ tiếng Anh tương đương.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài và người dùng tương tác trực tiếp (chatbot CLI/web) — thấy chữ xuất hiện ngay giảm cảm giác chờ đợi dù tổng thời gian không đổi. Non-streaming phù hợp hơn khi cần xử lý toàn bộ output trước khi dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giãn dần thời gian chờ giữa các lần retry, cho server thời gian hồi phục thay vì bị dội thêm request ngay lập tức. Nếu hàng nghìn client cùng retry với delay cố định, tất cả sẽ đồng loạt gọi lại cùng một thời điểm, khiến server càng quá tải nặng hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> "Bạn là trợ giảng thân thiện của khóa AI-LLM, trả lời ngắn gọn (tối đa 3 câu) bằng tiếng Việt, dùng ví dụ đơn giản khi giải thích khái niệm kỹ thuật, và luôn hỏi lại nếu câu hỏi chưa rõ." "Trả lời ngắn gọn (tối đa 3 câu)" giúp giảm số token sinh ra mỗi lượt — vừa giảm chi phí/độ trễ vừa tránh model lan man trong một phiên chat nhiều lượt. "Chỉ định tiếng Việt" đảm bảo output nhất quán ngôn ngữ với học viên, tránh model tự chuyển sang tiếng Anh khi gặp thuật ngữ kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: history chỉ giữ 3 lượt nên trợ lý quên ngữ cảnh cũ. Cải thiện: khi history sắp vượt 6 message, dùng `call_openai_mini` tóm tắt các lượt cũ thành 1–2 câu rồi chèn vào đầu messages, giữ ngữ cảnh dài hạn mà không tốn nhiều token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
