"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Ứng dụng mô phỏng prompt engineering và tool calling cho Lab #4.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import unicodedata
from typing import Dict, Any, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP

# ═══════════════════════════════════════════════════════════════════════════
# System prompt cấp sản xuất với quy tắc, phạm vi và hợp đồng đầu ra.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: Tư vấn sản phẩm và dịch vụ VinFast, Vinpearl.
- Phong cách: Chuyên nghiệp, thân thiện, rõ ràng và chính xác.

## AVAILABLE TOOLS
"""
    + json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
    + """

## CORE RULES
1. Không bịa giá, tính năng, tình trạng sản phẩm hoặc mã ticket.
2. Khi cần tra cứu sản phẩm, bắt buộc dùng search_product_catalog.
3. Khi khách hàng cần ghi nhận sự cố hoặc phản ánh, dùng submit_support_ticket.
4. Câu hỏi FAQ có thông tin xác thực thì trả lời trực tiếp; không gọi tool nếu không cần.
5. Nếu cần nhiều tool, gọi tuần tự và tổng hợp các kết quả quan sát được.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ nội dung thuộc hệ sinh thái Vingroup.
- Lịch sự từ chối yêu cầu ngoài phạm vi.
- Nếu dữ liệu không đủ, nói rõ điều chưa xác minh thay vì suy đoán.

## OUTPUT CONTRACT
- Luôn trả lời bằng tiếng Việt.
- Trình bày sản phẩm bằng danh sách hoặc bảng, nêu giá theo dữ liệu tool.
- Khi tạo ticket, xác nhận mã ticket và mức ưu tiên trả về từ tool.

## RESPONSE FORMAT
Thought: <mục đích của bước hiện tại>
Action: {"name": "<tên tool>", "args": {<tham số>}}
Observation: <kết quả trả về từ tool>
Final Answer: <câu trả lời hoàn chỉnh cho khách hàng>
"""
)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": (
                f"Câu hỏi: {user_input}\n"
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo tại showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân loại yêu cầu và trích xuất đối số cho các tool."""
        normalized_input = unicodedata.normalize("NFC", user_input)
        lower = normalized_input.lower()
        intents: Dict[str, Any] = {
            "needs_catalog": False,
            "needs_ticket": False,
            "is_faq": False,
            "catalog_args": {},
            "ticket_args": {}
        }

        faq_keywords = (
            "chính sách", "bảo hành", "đổi trả", "bao lâu", "điều kiện",
            "warranty", "refund", "return"
        )
        catalog_action = any(
            keyword in lower
            for keyword in ("xem", "tìm", "mua", "giá", "dưới", "bao nhiêu", "tư vấn")
        )
        ticket_action = any(
            keyword in lower
            for keyword in ("phản hồi", "khiếu nại", "ghi nhận", "ticket", "phản ánh", "hỗ trợ")
        )
        has_faq = any(keyword in lower for keyword in faq_keywords)
        if has_faq and not catalog_action and not ticket_action:
            intents["is_faq"] = True
            return intents

        is_vehicle_query = any(
            keyword in lower for keyword in ("xe điện", "vinfast", "vf ", "ô tô")
        )
        is_travel_query = any(
            keyword in lower
            for keyword in ("resort", "vinpearl", "du lịch", "nghỉ dưỡng", "khách sạn")
        )
        if catalog_action and (is_vehicle_query or is_travel_query):
            intents["catalog_args"]["category"] = (
                "du_lich" if is_travel_query and not is_vehicle_query else "xe_dien"
            )
            price_match = re.search(
                r"(\d+(?:[.,]\d+)?)\s*(triệu|tr|tỷ|tỉ)\b",
                lower
            )
            if price_match:
                amount = float(price_match.group(1).replace(",", "."))
                unit = price_match.group(2)
                multiplier = 1_000_000_000 if unit in ("tỷ", "tỉ") else 1_000_000
                intents["catalog_args"]["max_price"] = int(amount * multiplier)
            else:
                raw_price_match = re.search(
                    r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+|\d{7,})(?!\d)",
                    lower
                )
                if raw_price_match:
                    raw_price = re.sub(r"[.,]", "", raw_price_match.group(1))
                    intents["catalog_args"]["max_price"] = int(raw_price)
            intents["needs_catalog"] = True

        issue_keywords = ("lỗi", "hỏng", "bị", "ẩm mốc", "không hoạt động", "sự cố")
        if any(keyword in lower for keyword in issue_keywords) or ticket_action:
            name_match = re.search(
                r"(?:tôi\s+tên(?:\s+là)?|tên\s+tôi(?:\s+là)?|tôi\s+là)\s*([^,.;!?]+)",
                normalized_input,
                re.IGNORECASE
            )
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng ẩn danh"

            issue_match = re.search(
                r"\b(?:xe|phòng|sản phẩm|dịch vụ|pin)\b"
                r"[^,.!?]*?(?:bị|lỗi|hỏng|ẩm mốc|không hoạt động|sự cố)[^,.!?]*",
                normalized_input,
                re.IGNORECASE
            )
            if issue_match:
                issue_description = issue_match.group(0).strip()
            else:
                issue_match = re.search(
                    r"(?:bị|lỗi|hỏng|ẩm mốc|không hoạt động|sự cố)[^,.!?]*",
                    normalized_input,
                    re.IGNORECASE
                )
                issue_description = (
                    issue_match.group(0).strip()
                    if issue_match else "Khách hàng cần hỗ trợ"
                )
            issue_description = re.sub(
                r"\s+của tôi(?=\s+(?:bị|lỗi|hỏng|ẩm mốc|không hoạt động|sự cố))",
                "",
                issue_description,
                flags=re.IGNORECASE
            )
            issue_description = re.sub(
                r"\bphòng tôi đặt\b",
                "phòng đặt",
                issue_description,
                flags=re.IGNORECASE
            )
            issue_description = issue_description[0].upper() + issue_description[1:]

            if any(keyword in lower for keyword in ("khẩn cấp", "nghiêm trọng", "gấp", "cấp bách")):
                priority = "high"
            elif any(keyword in lower for keyword in ("nhẹ", "không gấp")):
                priority = "low"
            else:
                priority = "medium"

            intents["needs_ticket"] = True
            intents["ticket_args"] = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }

        return intents

    def _execute_step(
        self,
        intents: Dict[str, Any],
        iteration: int
    ) -> Tuple[str, bool]:
        """Thực thi một bước trong vòng lặp ReAct mô phỏng."""
        if intents["is_faq"]:
            final_answer = (
                "Theo dữ liệu sản phẩm hiện có, VinFast VF 5 Plus được ghi nhận "
                "có bảo hành pin 10 năm. Dữ liệu này chưa xác nhận thời hạn áp "
                "dụng chung cho tất cả mẫu xe; vui lòng liên hệ VinFast để xác "
                "nhận điều kiện bảo hành cụ thể."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Đây là câu hỏi FAQ; không cần gọi tool.",
                "final_answer": final_answer
            })
            return final_answer, True

        if intents["needs_catalog"] and iteration == 1:
            args = intents["catalog_args"]
            observation = TOOL_MAP["search_product_catalog"](**args)
            self.trace.append({
                "iteration": iteration,
                "thought": "Tra cứu danh mục sản phẩm theo yêu cầu và giới hạn giá.",
                "action": {"name": "search_product_catalog", "args": args},
                "observation": observation
            })
            if intents["needs_ticket"]:
                return f"Đã tra cứu được {len(observation)} kết quả.", False
            if not observation:
                maximum = args.get("max_price")
                limit = f" với giá tối đa {maximum:,} VNĐ" if maximum is not None else ""
                return f"Rất tiếc, không tìm thấy sản phẩm phù hợp{limit}.", True
            if isinstance(observation, list) and observation and "error" in observation[0]:
                return f"Lỗi hệ thống: {observation[0]['error']}", True

            product_lines = []
            for index, product in enumerate(observation, 1):
                availability = (
                    "Còn hàng" if product["availability"] == "in_stock" else "Đặt trước"
                )
                product_lines.append(
                    f"{index}. **{product['name']}** — {product['price_vnd']:,} VNĐ\n"
                    f"   {product['description']}\n"
                    f"   Tình trạng: {availability}"
                )
            return (
                f"Tìm thấy {len(observation)} sản phẩm phù hợp:\n\n"
                + "\n".join(product_lines),
                True
            )

        if intents["needs_ticket"] and (
            iteration == 2 or (iteration == 1 and not intents["needs_catalog"])
        ):
            args = intents["ticket_args"]
            observation = TOOL_MAP["submit_support_ticket"](**args)
            self.trace.append({
                "iteration": iteration,
                "thought": "Ghi nhận yêu cầu hỗ trợ vào hệ thống ticket.",
                "action": {"name": "submit_support_ticket", "args": args},
                "observation": observation
            })
            if not intents["needs_catalog"]:
                return (
                    f"Đã ghi nhận yêu cầu hỗ trợ của bạn:\n"
                    f"- **Mã ticket:** {observation['ticket_id']}\n"
                    f"- **Khách hàng:** {observation['customer_name']}\n"
                    f"- **Mức ưu tiên:** {observation['priority']}\n"
                    f"- **Trạng thái:** {observation['status']}",
                    True
                )
            return f"Đã tạo ticket {observation['ticket_id']}.", False

        if not intents["needs_catalog"] and not intents["needs_ticket"]:
            final_answer = (
                "Xin lỗi, tôi chỉ hỗ trợ thông tin về sản phẩm và dịch vụ thuộc "
                "hệ sinh thái Vingroup."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Yêu cầu nằm ngoài phạm vi hỗ trợ hoặc chưa đủ thông tin.",
                "final_answer": final_answer
            })
            return final_answer, True

        catalog_observation = next(
            (
                entry["observation"] for entry in self.trace
                if entry.get("action", {}).get("name") == "search_product_catalog"
            ),
            []
        )
        ticket_observation = next(
            (
                entry["observation"] for entry in self.trace
                if entry.get("action", {}).get("name") == "submit_support_ticket"
            ),
            {}
        )

        if catalog_observation and isinstance(catalog_observation, list):
            if catalog_observation and "error" in catalog_observation[0]:
                catalog_summary = f"Lỗi tra cứu: {catalog_observation[0]['error']}"
            else:
                catalog_summary = "\n".join(
                    f"{index}. **{product['name']}** — {product['price_vnd']:,} VNĐ"
                    for index, product in enumerate(catalog_observation, 1)
                ) or "Không tìm thấy sản phẩm phù hợp."
        else:
            catalog_summary = "Không tìm thấy sản phẩm phù hợp."

        ticket_summary = (
            f"- Mã ticket: {ticket_observation['ticket_id']}\n"
            f"- Khách hàng: {ticket_observation['customer_name']}\n"
            f"- Mức ưu tiên: {ticket_observation['priority']}\n"
            f"- Trạng thái: {ticket_observation['status']}"
        )
        final_answer = (
            f"Kết quả tra cứu sản phẩm:\n{catalog_summary}\n\n"
            f"Thông tin ticket hỗ trợ:\n{ticket_summary}"
        )
        self.trace.append({
            "iteration": iteration,
            "thought": "Tổng hợp kết quả từ các tool đã gọi.",
            "final_answer": final_answer
        })
        return final_answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        iteration = 1
        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(intents, iteration)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
