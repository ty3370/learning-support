import streamlit as st
import pymysql
import json
import re
import pandas as pd

# ===== LaTeX 텍스트 정리 함수 =====
def clean_inline_latex(text):
    text = re.sub(r",\s*\\text\{(.*?)\}", r" \1", text)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\ce\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"\1/\2", text)
    text = re.sub(r"\\sqrt\{(.*?)\}", r"√\1", text)
    text = re.sub(r"\\rightarrow", "→", text)
    text = re.sub(r"\\to", "→", text)
    text = re.sub(r"\^\{(.*?)\}", r"^\1", text)
    text = re.sub(r"_\{(.*?)\}", r"_\1", text)
    text = re.sub(r"\^([0-9])", r"^\1", text)
    text = re.sub(r"_([0-9])", r"\1", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\(\((.*?)\)\)", r"\1", text)
    text = re.sub(r"\(([^()]*\\[a-z]+[^()]*)\)", lambda m: clean_inline_latex(m.group(1)), text)
    text = re.sub(r"\b(times)\b", "×", text)
    text = re.sub(r"\b(div|divided by)\b", "÷", text)
    text = re.sub(r"\b(plus)\b", "+", text)
    text = re.sub(r"\b(minus)\b", "-", text)

    replacements = {
        r"\\perp": "⟂",
        r"\\angle": "∠",
        r"\\parallel": "∥",
        r"\\infty": "∞",
        r"\\approx": "≈",
        r"\\sim": "∼",
        r"\\neq": "≠",
        r"\\leq": "≤",
        r"\\geq": "≥",
        r"\\pm": "±",
        r"\\mp": "∓",
        r"\\cdot": "·",
        r"\\times": "×",
        r"\\div": "÷",
        r"\\propto": "∝",
        r"\\equiv": "≡",
        r"\\cong": "≅",
        r"\\subseteq": "⊆",
        r"\\supseteq": "⊇",
        r"\\subset": "⊂",
        r"\\supset": "⊃",
        r"\\in": "∈",
        r"\\notin": "∉",
        r"\\cup": "∪",
        r"\\cap": "∩",
        r"\\forall": "∀",
        r"\\exists": "∃",
        r"\\nabla": "∇",
        r"\\partial": "∂",
    }
    for pattern, symbol in replacements.items():
        text = re.sub(pattern, symbol, text)

    text = re.sub(r"\bperp\b", "⟂", text)
    text = re.sub(r"\bangle\b", "∠", text)

    return text

# ===== DB 연결 =====
def connect_to_db():
    return pymysql.connect(
        host=st.secrets["DB_HOST"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_DATABASE"],
        charset='utf8mb4'
    )

# ===== 데이터 조회 =====
def fetch_students_v3(subject, topic):
    try:
        db = connect_to_db()
        cursor = db.cursor()
        sql = """
        SELECT DISTINCT number, name, code
        FROM qna_unique_v3
        WHERE subject = %s AND topic = %s
        ORDER BY number
        """
        cursor.execute(sql, (subject, topic))
        students = cursor.fetchall()
        cursor.close(); db.close()
        return students
    except pymysql.MySQLError as e:
        st.error(f"DB 오류: {e}")
        return []

def fetch_chat_v3(number, name, code, subject, topic):
    try:
        db = connect_to_db()
        cursor = db.cursor()
        sql = """
        SELECT chat
        FROM qna_unique_v3
        WHERE number = %s AND name = %s AND code = %s
          AND subject = %s AND topic = %s
        """
        cursor.execute(sql, (number, name, code, subject, topic))
        result = cursor.fetchone()
        cursor.close(); db.close()
        return result[0] if result else None
    except pymysql.MySQLError as e:
        st.error(f"DB 오류: {e}")
        return None

# ===== 기본 UI =====
st.title("학생 AI 대화 이력 조회(교사용)")
password = st.text_input("비밀번호를 입력하세요", type="password")
if password != st.secrets["PASSWORD"]:
    st.stop()

# ===== 과목/단원 선택 =====
TOPIC_MAP = {
    "1학년 과학": [
        "Ⅳ. 물질의 상태 변화",
        "Ⅴ. 힘의 작용",
        "Ⅵ. 기체의 성질"
    ],
    "2학년 수학": [
        "Ⅳ. 도형의 성질"
    ],
    "3학년 과학": [
        "Ⅳ. 자극과 반응",
        "Ⅴ. 생식과 유전",
        "Ⅵ. 에너지 전환과 보존"
    ]
}

subject = st.selectbox("과목 선택", ["과목을 선택하세요"] + list(TOPIC_MAP.keys()))
if subject == "과목을 선택하세요":
    st.stop()

topic = st.selectbox("대단원 선택", ["대단원을 선택하세요"] + TOPIC_MAP.get(subject, []))
if topic == "대단원을 선택하세요":
    st.stop()

# ===== 학생 목록 조회 =====
students = fetch_students_v3(subject, topic)
if not students:
    st.warning("해당 단원에 대해 대화한 학생이 없습니다.")
    st.stop()

student_options = [f"{n} ({nm}) / 코드: {c}" for n, nm, c in students]
selected = st.selectbox("학생 선택", student_options)
idx = student_options.index(selected)
number, name, code = students[idx]

# ===== 대화 불러오기 =====
chat_data = fetch_chat_v3(number, name, code, subject, topic)
if not chat_data:
    st.warning("대화 기록이 없습니다.")
    st.stop()

# ===== 대화 출력 =====
try:
    chat = json.loads(chat_data)
    st.write("### 대화 내용")

    for msg in chat:
        role = "**You:**" if msg["role"] == "user" else "**학습 도우미:**"
        ts = f" ({msg['timestamp']})" if "timestamp" in msg else ""
        content = msg["content"]

        parts = re.split(r"(@@@@@.*?@@@@@)", content, flags=re.DOTALL)
        cleaned_parts = []

        for part in parts:
            if part.startswith("@@@@@") and part.endswith("@@@@@"):
                st.latex(part[5:-5].strip())
                cleaned_parts.append(part[5:-5].strip())

            else:
                txt = clean_inline_latex(part.strip())
                for link in re.findall(r"(https?://\S+\.(?:png|jpg))", txt):
                    st.image(link)
                    txt = txt.replace(link, "")

                if txt.strip():
                    for line in txt.splitlines():
                        if line.strip():
                            st.write(f"{role} {line.strip()}{ts}")
                            role = ""

                cleaned_parts.append(txt)

except json.JSONDecodeError:
    st.error("대화 JSON 형식 오류입니다.")
    st.stop()