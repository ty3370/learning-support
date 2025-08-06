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

# ===== 학생 목록 조회 =====
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

# ===== 대화 기록 조회 =====
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

# ===== 메인 UI =====
st.title("📚 학생 AI 대화 이력 조회 (qna_unique_v3 기반)")

# ===== 비밀번호 확인 =====
password = st.text_input("비밀번호", type="password")
if password != st.secrets["PASSWORD"]:
    st.stop()

# ===== 과목/단원 선택 =====
subject = st.selectbox("과목을 선택하세요", ["과학"])
topic = st.selectbox("대단원을 선택하세요", [
    "Ⅳ. 자극과 반응", "Ⅴ. 생식과 유전", "Ⅵ. 에너지 전환과 보존"
])
if not topic:
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
    st.write("### 💬 대화 내용 (LaTeX 포함)")
    chat_table = []

    for msg in chat:
        role = "**You:**" if msg["role"] == "user" else "**과학 도우미:**"
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
                if txt:
                    lines = txt.splitlines()
                    for line in lines:
                        imgs = re.findall(r"(https?://\S+\.(?:png|jpg|jpeg))", line)
                        for img in imgs:
                            st.image(img)
                            line = line.replace(img, "")
                        if line.strip():
                            st.write(f"{role} {line.strip()}{ts}")
                            role = ""  # 한 번만 출력
                    cleaned_parts.append(txt)

        chat_table.append({
            "말한 사람": "학생" if msg["role"] == "user" else "과학 도우미",
            "내용": " ".join(cleaned_parts),
            "시간": msg.get("timestamp", "")
        })

    # ===== 복사용 표 =====
    st.write("### 복사용 표")
    df = pd.DataFrame(chat_table)
    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

except json.JSONDecodeError:
    st.error("대화 JSON 형식 오류입니다.")