"""
Script chạy thử Block 2 (exercises.md) — Câu 2.1 và Câu 2.2.
Chạy: python run_test2.py
Cần OPENAI_API_KEY (hoặc LAB_MODEL/LAB_MINI_MODEL nếu dùng NVIDIA NIM) trong .env.
"""

from template import chat_with_system_prompt, count_tokens

# ---------------------------------------------------------------------------
# Câu 2.1 — Sức mạnh của persona
# ---------------------------------------------------------------------------
def cau_2_1():
    print("=" * 70)
    print("CÂU 2.1 — Sức mạnh của persona")
    print("=" * 70)

    user_prompt = "Giải thích blockchain là gì?"

    persona_1 = "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
    persona_2 = "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

    resp1, latency1 = chat_with_system_prompt(persona_1, user_prompt)
    resp2, latency2 = chat_with_system_prompt(persona_2, user_prompt)

    print("\n--- Persona 1: Giáo viên tiểu học ---")
    print(f"(latency: {latency1:.2f}s, {len(resp1.split())} từ)")
    print(resp1)

    print("\n--- Persona 2: Chuyên gia tài chính ---")
    print(f"(latency: {latency2:.2f}s, {len(resp2.split())} từ)")
    print(resp2)


# ---------------------------------------------------------------------------
# Câu 2.2 — tiktoken vs đếm từ
# ---------------------------------------------------------------------------
def cau_2_2():
    print("\n" + "=" * 70)
    print("CÂU 2.2 — tiktoken vs đếm từ")
    print("=" * 70)

    # Đoạn văn tiếng Việt ~100 từ — có thể thay bằng đoạn văn của riêng bạn
    doan_van = (
        "Trí tuệ nhân tạo đang thay đổi cách con người làm việc và học tập "
        "trong nhiều lĩnh vực khác nhau. Từ y tế, giáo dục cho đến tài chính "
        "và sản xuất, các mô hình ngôn ngữ lớn giúp tự động hóa những công "
        "việc lặp đi lặp lại, đồng thời hỗ trợ con người đưa ra quyết định "
        "nhanh và chính xác hơn. Tuy nhiên, việc ứng dụng công nghệ này cũng "
        "đặt ra nhiều thách thức về đạo đức, quyền riêng tư và độ tin cậy của "
        "thông tin. Do đó, người dùng cần hiểu rõ giới hạn của các hệ thống "
        "trí tuệ nhân tạo trước khi áp dụng vào công việc thực tế hàng ngày "
        "của mình, đặc biệt là trong những quyết định quan trọng có ảnh "
        "hưởng lớn đến cuộc sống."
    )

    so_tu = len(doan_van.split())
    uoc_luong_tu = so_tu / 0.75
    so_token_that = count_tokens(doan_van)

    chenh_lech_pct = abs(so_token_that - uoc_luong_tu) / so_token_that * 100

    print(f"\nĐoạn văn ({so_tu} từ):\n{doan_van}\n")
    print(f"Số từ:                          {so_tu}")
    print(f"Ước lượng theo 'số từ / 0.75':   {uoc_luong_tu:.1f} token")
    print(f"Số token thật (tiktoken):        {so_token_that} token")
    print(f"Chênh lệch:                      {chenh_lech_pct:.1f}%")


if __name__ == "__main__":
    cau_2_1()
    cau_2_2()
