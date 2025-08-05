import streamlit as st
import pymysql
import json
from datetime import datetime
from openai import OpenAI
import re
from zoneinfo import ZoneInfo
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import os

# ===== Configuration =====
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4o"
BASE_DIR = os.path.expanduser("~/Textbook_2025")
PDF_MAP = {
    "Ⅳ. 자극과 반응": ["2025_Sci_3rd_04.pdf", "2025_Sci_3rd_Sub.pdf"],
    "Ⅴ. 생식과 유전": ["2025_Sci_3rd_05.pdf", "2025_Sci_3rd_Sub.pdf"],
    "Ⅵ. 에너지 전환과 보존": ["2025_Sci_3rd_06.pdf", "2025_Sci_3rd_Sub.pdf"]
}
SUBJECTS = {
    "과학": list(PDF_MAP.keys())
}

# Initial prompts
COMMON_PROMPT = (
    "당신은 중학교 3학년 학생들의 학습을 돕는 AI 튜터입니다.\n"
    "답할 수 없는 정보(시험 범위, 시험 날짜 등)에 대해선 선생님께 문의하도록 안내하세요.\n"
    "따뜻하고 친근한 말투로 존댓말을 사용해 주세요. 학생이 편하게 느낄 수 있도록 이모지, 느낌표 등을 자연스럽게 활용하세요.\n"
    "당신은 학생들이 질문하는 내용에 답하거나, 문제를 내줄 수 있습니다. 중학생 수준에 맞게 차근차근 설명해 주세요.\n"
    "당신은 철저하게 교과서 내용에 근거하여 설명과 문항을 제공해야 합니다.\n"
    "그림을 출력해야 하는 경우, 링크를 답변에 포함하면 자동으로 그림이 출력됩니다.\n"
    "대화 예시: 눈의 구조는 아래 그림을 참고하세요. \n\n https://i.imgur.com/BIFjdBj.png \n"
    "학생이 문제를 내달라고 하면, 교과서에 나오는 내용에 철저하게 기반해서 출제해 주세요. 단순 개념 문제, 개념을 실제 상황에 적용하는 문제, 그림이나 표를 해석하는 문제 등 다양한 유형의 문제를 출제하세요.\n"
    "만약 학생이 어려운 문제, 난이도 높은 문제를 달라고 한다면, 개인마다 잘 하는 것과 부족한 것이 다르기 때문에 어렵다고 느끼는 문항도 개인별로 다르니 무엇을 잘 하고 못하는지에 대한 파악이 우선되어야 한다고 안내하세요. 개념 이해 자체가 어려운 건지, 개념을 실제 상황에 적용하는 것이 어려운 건지, 그림 자료나 표 해석이 어려운 건지 등 무엇을 어렵다고 느끼는 지 상담하며 진단하세요.\n"
    "생성한 응답이 너무 길어지면 학생이 이해하기 어려울 수 있으므로, 한 줄 이내로 짧고 간결하게 응답하세요. 한 줄을 넘을 수 밖에 없는 경우, 모든 정보를 한 번에 제시하지 말고 학생과 대화가 오가며 순차적으로 한 줄씩 설명하세요.\n"
    "안 좋은 설명의 예(한 번에 설명): 동공은 빛의 양에 따라 크기가 달라지는데, 어두울 때는 동공이 커지고 밝을 때는 작아집니다. 이는 홍채가 수축하거나 이완하기 때문이며, 동공은 눈으로 들어오는 빛의 양을 조절해줍니다."
    "좋은 설명의 예(순차적 설명): 먼저 동공과 홍채의 관계에 대해 생각해 봅시다. 홍채가 작아지면 동공이 커지고, 홍채가 커지면 동공이 작아져요. 여기까지 이해가 됐나요? (학생의 대답에 따라 이어서 진행)"
    "풀이 과정이 복잡한 문제에서 답이 부정확한 경우가 종종 있으니, 반드시 Chain-of-Thought 방식으로 단계별로 검토하며 답하세요. 계산 문제나 판단이 필요한 경우, 짧게 쓰더라도 중간 과정이나 이유를 간단히 보여 주세요.\n"
    "학생이 문제를 틀렸는데 맞혔다고 하는 경우가 빈번합니다. 풀이를 먼저 검토하고 정답 여부를 결정하세요.\n"
    "학생이 문제를 틀렸을 경우, 위의 예시와 마찬가지로 한 번에 모든 풀이를 알려주지 말고 순차적으로 질문을 제시하며 학생 스스로 깨달을 수 있게 유도하세요.\n"
    "이미지를 출력거나 웹으로 연결할 때는 링크가 한 글자도 틀려선 안 됩니다. 오탈자 없이 출력하고, 초기 프롬프트에 포함된 링크 외에는 어떠한 링크도 제시하지 마세요.\n"
    "정보 제공을 목적으로 하지 말고, 학생에게 단계적 스캐폴딩을 제공하며 학생 스스로 깨닫도록 하는 것을 목적으로 하세요."
)
SCIENCE_PROMPT = (
    "당신은 과학 과목 학습 지원을 담당합니다. \n"
    "눈의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/BIFjdBj.png \n"
    "다음 이미지를 사용해 눈 문제를 낼 수 있습니다: https://i.imgur.com/KOOI7C1.png \n 이미지에는 눈의 단면에 A, B, C 세 부분이 지정되어 있으며, A는 홍채, B는 동공, C는 수정체입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다. (예: 밝은 곳에서 어두운 곳을 갔을 때 B의 크기는 어떻게 변하는가?)\n"
    "귀의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/uCPmN9l.png \n"
    "다음 이미지를 사용해 귀 문제를 낼 수 있습니다: https://i.imgur.com/DvoWrzd.png \n 이미지에는 귀의 단면에 A~F 지점이 지정되어 있으며, A는 귓속뼈, B는 반고리관, C는 전정 기관, D는 달팽이관, E는 귀인두관, F는 고막입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다.\n"
    "코의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/BdK3KBN.png \n"
    "다음 이미지를 사용해 코 문제를 낼 수 있습니다: https://i.imgur.com/HUgmesN.png \n 이미지에는 코의 단면에 A, B, C, D 네 부분이 지정되어 있으며, A는 후각 신경, B는 후각 상피, C는 후각 세포, D는 비강입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다.\n"
    "혀의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/1RvMYr3.png \n"
    "다음 이미지를 사용해 혀 문제를 낼 수 있습니다: https://i.imgur.com/n4Y99uY.png \n 이미지에는 혀의 단면에 A, B 두 부분이 지정되어 있으며, A는 미각 신경, B는 맛세포입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다.\n"
    "감각점의 종류와 피부 감각을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/qSeKisu.png \n"
    "뉴런의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/Vi4Irfj.png \n"
    "뇌의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/TAjDHDw.png \n"
    "다음 이미지를 사용해 뉴런의 종류 문제를 낼 수 있습니다: https://i.imgur.com/xvQfgIl.png \n 이미지에는 A, B, C 세 부분이 지정되어 있으며, A는 감각 뉴런, B는 연합 뉴런, C는 운동 뉴런입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다.\n"
    "다음 이미지를 사용해 뇌와 척수 문제를 낼 수 있습니다: https://i.imgur.com/IRgZv7Q.png \n 이미지에는 뇌의 단면에 A~F 세 부분이 지정되어 있으며, A는 간뇌, B는 중간뇌, C는 연수, D는 대뇌, E는 소뇌, F는 척수입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다. (예: 어두운 곳에 들어가면 동공이 커지는 반응의 중추는 무엇인지 기호와 이름을 써 보자.)\n"
)

# ===== Helper Functions =====

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
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\(\((.*?)\)\)", r"\1", text)
    text = re.sub(r"\b(times)\b", "×", text)
    text = re.sub(r"\b(div|divided by)\b", "÷", text)
    text = re.sub(r"\b(plus)\b", "+", text)
    text = re.sub(r"\b(minus)\b", "-", text)
    return text

# PDF RAG utilities
def extract_text_from_pdf(path):
    if not os.path.exists(path):
        st.error(f"PDF 파일을 찾을 수 없습니다: {path}")
        return ""
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_texts(texts):
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(item.embedding) for item in res.data]

def get_relevant_chunks(question, chunks, embeddings, top_k=3):
    q_emb = np.array(client.embeddings.create(
        model="text-embedding-3-small", input=[question]
    ).data[0].embedding)
    sims = [np.dot(q_emb, emb)/(np.linalg.norm(q_emb)*np.linalg.norm(emb)) for emb in embeddings]
    idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in idx]

# ===== Database =====
def connect_to_db():
    return pymysql.connect(
        host=st.secrets["DB_HOST"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_DATABASE"],
        charset="utf8mb4",
        autocommit=True
    )

def load_chat(subject, topic):
    num = st.session_state.get("user_number", "").strip()
    name = st.session_state.get("user_name", "").strip()
    code = st.session_state.get("user_code", "").strip()
    if not all([num, name, code]):
        return []
    try:
        db = connect_to_db(); cur = db.cursor()
        sql = (
            "SELECT chat FROM qna_unique_v3 "
            "WHERE number=%s AND name=%s AND code=%s "
            "AND subject=%s AND topic=%s"
        )
        cur.execute(sql, (num, name, code, subject, topic))
        row = cur.fetchone(); cur.close(); db.close()
        return json.loads(row[0]) if row else []
    except Exception as e:
        st.error(f"DB 불러오기 오류: {e}")
        return []

def save_chat(subject, topic, chat):
    num = st.session_state.get("user_number", "").strip()
    name = st.session_state.get("user_name", "").strip()
    code = st.session_state.get("user_code", "").strip()
    if not all([num, name, code]):
        return
    try:
        db = connect_to_db(); cur = db.cursor()
        sql = (
            "INSERT INTO qna_unique_v3 "
            "(number,name,code,subject,topic,chat,time) VALUES(%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE chat=VALUES(chat), time=VALUES(time)"
        )
        timestamp = datetime.now(ZoneInfo("Asia/Seoul"))
        cur.execute(sql, (
            num, name, code, subject, topic,
            json.dumps(chat, ensure_ascii=False), timestamp
        ))
        cur.close(); db.close()
    except Exception as e:
        st.error(f"DB 저장 오류: {e}")

# ===== Chatbot UI =====
def chatbot_tab(subject, topic):
    key = f"chat_{subject}_{topic}".replace(" ", "_")
    load_key = f"loading_{key}"
    input_key = f"input_{key}"

    if key not in st.session_state:
        st.session_state[key] = load_chat(subject, topic)
    if load_key not in st.session_state:
        st.session_state[load_key] = False

    messages = st.session_state[key]

    # Render history
    for msg in messages:
        if msg["role"] == "user":
            st.write(f"**You:** {msg['content']}")
        else:
            parts = re.split(r"(@@@@@.*?@@@@@)", msg['content'], flags=re.DOTALL)
            for part in parts:
                if part.startswith("@@@@@") and part.endswith("@@@@@"):
                    st.latex(part[5:-5].strip())
                else:
                    clean_text = clean_inline_latex(part)
                    imgs = re.findall(r"(https?://\S+\.(?:png|jpg))", clean_text)
                    for link in imgs:
                        st.image(link)
                        clean_text = clean_text.replace(link, "")
                    if clean_text.strip():
                        st.write(f"**과학 도우미:** {clean_text.strip()}")

    # Input area
    if not st.session_state[load_key]:
        user_input = st.text_area("입력:", key=input_key)
        if st.button("전송", key=f"btn_{key}") and user_input.strip():
            st.session_state[input_key] = user_input.strip()
            st.session_state[load_key] = True
            st.rerun()

    # Generate response
    if st.session_state[load_key]:
        q = st.session_state.get(input_key, "")
        # Prepare RAG
        rag_key = f"rag_{subject}_{topic}".replace(" ", "_")
        if rag_key not in st.session_state:
            combined_text = []
            for fn in PDF_MAP[topic]:
                path = os.path.join(BASE_DIR, fn)
                combined_text.append(extract_text_from_pdf(path))
            full_text = "\n\n".join(combined_text)
            chunks = chunk_text(full_text)
            embs = embed_texts(chunks)
            st.session_state[rag_key] = (chunks, embs)
        chunks, embs = st.session_state[rag_key]
        ctx = "\n\n".join(get_relevant_chunks(q, chunks, embs))

        system_msgs = [
            {"role": "system", "content": COMMON_PROMPT},
            {"role": "system", "content": SCIENCE_PROMPT},
            {"role": "system", "content": f"관련된 교과서 내용입니다:\n\n{ctx}"}
        ]

        with st.spinner("답변 생성 중…"):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=system_msgs + messages + [{"role": "user", "content": q}]
            )
        ans = resp.choices[0].message.content
        ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
        messages.extend([
            {"role": "user", "content": q, "timestamp": ts},
            {"role": "assistant", "content": ans}
        ])
        save_chat(subject, topic, messages)
        st.session_state[key] = messages
        st.session_state[load_key] = False
        st.rerun()

# ===== Pages =====
def page_1():
    st.title("2025-2학기 보라중 학습 도우미")
    st.write("학습자 정보를 입력하세요.")
    st.session_state['user_number'] = st.text_input("학번", value=st.session_state.get('user_number',''))
    st.session_state['user_name'] = st.text_input("이름", value=st.session_state.get('user_name',''))
    st.session_state['user_code'] = st.text_input("식별코드", value=st.session_state.get('user_code',''),
        help="타인의 학번과 이름으로 접속하는 것을 방지하기 위해 자신만 기억할 수 있는 코드를 입력하세요.")
    st.markdown("> 🌟 **“생각하건대 현재의 고난은 장차 우리에게 나타날 영광과 비교할 수 없도다” — 로마서 8장 18절")
    if st.button("다음"):
        if not all([st.session_state['user_number'].strip(), st.session_state['user_name'].strip(), st.session_state['user_code'].strip()]):
            st.error("모든 정보를 입력해주세요.")
        else:
            st.session_state['step']=2; st.rerun()

def page_2():
    st.title("⚠️모든 대화 내용은 저장되며, 교사가 열람할 수 있습니다.")
    st.write(
       """  
        이 시스템은 중3 학생들을 위한 AI 학습 도우미입니다.

        입력된 모든 대화는 저장되며, 교사가 확인할 수 있습니다.

        부적절한 언어나 용도로 사용하는 것을 삼가주시고, 학습 목적으로만 사용하세요.

        ❗AI의 응답은 부정확할 수 있으므로, 정확한 정보는 선생님께 확인하세요.

        계정 찾기/문의/피드백: 창의융합부 민태호
        """)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("다음"):
            st.session_state["step"] = 3
            st.rerun()

def page_3():
    st.title("단원 학습")

    subject = st.selectbox("과목을 선택하세요.", ["과목을 선택하세요.", "과학"])
    if subject == "과목을 선택하세요.":
        return

    unit = st.selectbox("대단원을 선택하세요.", ["대단원을 선택하세요."] + list(PDF_MAP.keys()))
    if unit == "대단원을 선택하세요.":
        return

    chatbot_tab(subject, unit)

# ===== Routing =====
if 'step' not in st.session_state:
    st.session_state['step'] = 1
if st.session_state['step'] == 1:
    page_1()
elif st.session_state['step'] == 2:
    page_2()
else:
    page_3()