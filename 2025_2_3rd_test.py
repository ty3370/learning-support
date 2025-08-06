import streamlit as st
import pymysql
import json
from datetime import datetime
from openai import OpenAI
import re
from zoneinfo import ZoneInfo
import fitz  # PyMuPDF
import numpy as np
import os
import hashlib
import time

# ===== Configuration =====
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4o"
BASE_DIR = os.path.join(os.getcwd(), "Textbook_2025")
PDF_MAP = {
    "Ⅳ. 자극과 반응": ["2025_Sci_3rd_04.pdf"],
    "Ⅴ. 생식과 유전": ["2025_Sci_3rd_05.pdf"],
    "Ⅵ. 에너지 전환과 보존": ["2025_Sci_3rd_06.pdf"]
}
SUBJECTS = {"과학": list(PDF_MAP.keys())}

# Initial prompts
COMMON_PROMPT = (
    "당신은 중학교 3학년 학생들의 학습을 돕는 AI 튜터입니다.\n"
    "답할 수 없는 정보(시험 범위, 시험 날짜 등)에 대해선 선생님께 문의하도록 안내하세요.\n"
    "따뜻하고 친근한 말투로 존댓말을 사용해 주세요. 학생이 편하게 느낄 수 있도록 이모지, 느낌표 등을 자연스럽게 활용하세요.\n"
    "당신은 학생들이 질문하는 내용에 답하거나, 문제를 내줄 수 있습니다. 중학생 수준에 맞게 차근차근 설명해 주세요.\n"
    "당신은 철저하게 교과서 내용에 근거하여 설명과 문항을 제공해야 합니다.\n"
    "모든 수식은 반드시 LaTeX 형식으로 작성하고 '@@@@@'로 감싸주세요. 수식 앞뒤에는 반드시 빈 줄로 구분해 주세요. 이 규칙은 어떤 경우에도 반드시 지켜야 합니다. 예시:\n\n@@@@@\nE_p = 9.8 \\times m \\times h\n@@@@@\n\n"
    "절대로 문장 중간에 LaTeX 형식이 들어가선 안 됩니다. LaTex 사용은 반드시 줄바꿈하고, LaTex 앞뒤를 각각 @ 기호 5개로 감싸야 합니다.\ "
    "틀린 표현 예시: 어떤 물체의 질량이 2kg이고 높이가 10m일 때 위치에너지는((E_p = 9.8 \\times m \\times h))입니다.\n"
    "맞는 표현 예시: 어떤 물체의 질량이 2kg이고 높이가 10m일 때 위치에너지는 다음과 같이 계산할 수 있습니다:\n\n@@@@@\nE_p = 9.8 \\times m \\times h\n@@@@@\n\n"
    "만약 LaTex를 줄바꿈 없이 사용해야만 하는 상황이라면, LaTex가 아닌 글로 쓰세요. \n틀린 표현 예시: 위치에너지는 9.8 \\times m \\times h입니다. \n맞는 표현 예시: 위치에너지는 9.8×m×h입니다. LaTex를 쓰려면 반드시 앞뒤로 줄바꿈해야 합니다.\n"
    "그림을 출력해야 하는 경우, 링크를 답변에 포함하면 자동으로 그림이 출력됩니다.\n"
    "대화 예시: 눈의 구조는 아래 그림을 참고하세요. \n\n https://i.imgur.com/BIFjdBj.png \n"
    "학생이 문제를 내달라고 하면, 교과서에 나오는 내용에 철저하게 기반해서 출제해 주세요. 단순 개념 문제, 개념을 실제 상황에 적용하는 문제, 그림이나 표를 해석하는 문제 등 다양한 유형의 문제를 출제하세요.\n"
    "만약 학생이 어려운 문제, 난이도 높은 문제를 달라고 한다면, 개인마다 잘 하는 것과 부족한 것이 다르기 때문에 어렵다고 느끼는 문항도 개인별로 다르니 무엇을 잘 하고 못하는지에 대한 파악이 우선되어야 한다고 안내하세요. 개념 이해 자체가 어려운 건지, 개념을 실제 상황에 적용하는 것이 어려운 건지, 그림 자료나 표 해석이 어려운 건지, 서술형 답을 쓰는 게 어려운 건지 등 무엇을 어렵다고 느끼는 지 상담하며 진단하세요.\n"
    "생성한 응답이 너무 길어지면 학생이 이해하기 어려울 수 있으므로, 한 줄 이내로 짧고 간결하게 응답하세요. 한 줄을 넘을 수 밖에 없는 경우, 모든 정보를 한 번에 제시하지 말고 학생과 대화가 오가며 순차적으로 한 줄씩 설명하세요.\n"
    "안 좋은 설명의 예(한 번에 설명): 동공은 빛의 양에 따라 크기가 달라지는데, 어두울 때는 동공이 커지고 밝을 때는 작아집니다. 이는 홍채가 수축하거나 이완하기 때문이며, 동공은 눈으로 들어오는 빛의 양을 조절해줍니다.\n"
    "좋은 설명의 예(순차적 설명): 먼저 동공과 홍채의 관계에 대해 생각해 봅시다. 홍채가 작아지면 동공이 커지고, 홍채가 커지면 동공이 작아져요. 여기까지 이해가 됐나요? (학생의 대답에 따라 이어서 진행)\n"
    "학생이 전반적인 내용을 요약해달라고 요청할 경우에도, 마찬가지로 일부 내용만 요약해 제시한 뒤 이어서 계속 요약하냐고 묻고, 학생이 계속해 달라고 하면 이어서 요약본을 제시하세요. 이런 방법으로 하나의 대화가 지나치게 길어지지 않도록 조절하세요.\n"
    "풀이 과정이 복잡한 문제에서 답이 부정확한 경우가 종종 있으니, 반드시 Chain-of-Thought 방식으로 단계별로 검토하며 답하세요. 계산 문제나 판단이 필요한 경우, 짧게 쓰더라도 중간 과정이나 이유를 간단히 보여 주세요.\n"
    "학생이 문제를 틀렸는데 맞혔다고 하는 경우가 빈번합니다. 풀이를 먼저 검토하고 정답 여부를 결정하세요.\n"
    "학생이 문제를 틀렸을 경우, 위의 예시와 마찬가지로 한 번에 모든 풀이를 알려주지 말고 순차적으로 질문을 제시하며 학생 스스로 깨달을 수 있게 유도하세요.\n"
    "이미지를 출력거나 웹으로 연결할 때는 링크가 한 글자도 틀려선 안 됩니다. 오탈자 없이 출력하고, 초기 프롬프트에 포함된 링크 외에는 어떠한 링크도 제시하지 마세요.\n"
    "정보 제공을 목적으로 하지 말고, 학생에게 단계적 스캐폴딩을 제공하며 학생 스스로 깨닫도록 하는 것을 목적으로 하세요."
)

SCIENCE_04_PROMPT = (
    "당신은 과학의 Ⅳ. 자극과 반응 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "눈의 구조와 기능: 공막, 맥락막, 홍채, 동공, 수정체, 섬모체, 유리체, 맹점, 각막, 망막, 시각 신경, 홍채와 동공·섬모체와 수정체의 조절\n"
    "귀의 구조와 기능: 귓바퀴, 외이도, 고막, 귓속뼈, 달팽이관, 청각 신경, 평형 감각(반고리관, 전정기관, 평형 감각 신경), 귀인두관\n"
    "코의 구조와 기능: 비강, 후각 상피, 후각 세포, 후각 신경\n"
    "혀의 구조와 기능: 유두, 맛봉오리, 맛세포, 미각 신경\n"
    "감각점의 종류: 통점, 압점, 냉점, 온점, 촉점\n"
    "뉴런의 구조와 기능(신경 세포체, 가지 돌기, 축삭 돌기), 감각 뉴런, 연합 뉴런, 운동 뉴런\n"
    "중추신경계(뇌와 척수), 말초 신경계(감각 신경과 운동 신경), 뇌의 구조와 기능(대뇌, 소뇌, 간뇌, 중간뇌, 연수, 척수), 자율 신경, 교감 신경\n"
    "무조건 반사, 뇌하수체 호르몬(생장 호르몬, 갑상샘 자극 호르몬, 항이뇨 호르몬), 갑상샘 호르몬(티록신), 부신 호르몬(아드레날린), 이자 호르몬(인슐린, 글루카곤), 난소 호르몬(에스트로젠), 정소 호르몬(테스토스테론), 호르몬 관련 질병\n"
    "항상성, 체온 조절(혈관 확장과 수축, 뇌하수체와 갑상샘 호르몬 및 세포 호흡 변화), 혈당량 조절(인슐린 또는 글루카곤 분비)\n\n"
    "2. 학습 지원 지침\n"
    "설명 시 이미지를 사용해도 되고, 이미지 없이 텍스트로만 설명해도 됩니다. 문제를 낼 때도 텍스트로만 이루어진 문제, 표로 정보가 제공되는 문제, 이미지를 해석하는 문제 등을 다양하게 출제하세요.\n"
    "하나의 대화에서는 하나의 그림만을 사용하세요.\n"
    "아래 언급된 링크 외에는 어떠한 링크도 사용하지 마세요.\n\n"
    "3. 사용 가능한 이미지 목록:\n"
    "눈의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/BIFjdBj.png \n"
    "다음 이미지를 사용해 눈 문제를 낼 수 있습니다: https://i.imgur.com/KOOI7C1.png \n 이미지에는 눈의 단면에 A, B, C 세 부분이 지정되어 있으며, A는 홍채, B는 동공, C는 수정체입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다. (예: 밝은 곳에서 어두운 곳을 갔을 때 B의 크기는 어떻게 변하는가?)\n"
    "귀의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/uCPmN9l.png \n"
    "다음 이미지를 사용해 귀 문제를 낼 수 있습니다: https://i.imgur.com/DvoWrzd.png \n 이미지에는 귀의 단면에 A~F 지점이 지정되어 있으며, A는 귓속뼈, B는 반고리관, C는 전정 기관, D는 달팽이관, E는 귀인두관, F는 고막입니다.\n"
    "코의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/BdK3KBN.png \n"
    "다음 이미지를 사용해 코 문제를 낼 수 있습니다: https://i.imgur.com/HUgmesN.png \n 이미지에는 코의 단면에 A, B, C, D 네 부분이 지정되어 있으며, A는 후각 신경, B는 후각 상피, C는 후각 세포, D는 비강입니다.\n"
    "혀의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/1RvMYr3.png \n"
    "다음 이미지를 사용해 혀 문제를 낼 수 있습니다: https://i.imgur.com/n4Y99uY.png \n 이미지에는 혀의 단면에 A, B 두 부분이 지정되어 있으며, A는 미각 신경, B는 맛세포입니다.\n"
    "감각점의 종류와 피부 감각을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/qSeKisu.png \n"
    "뉴런의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/Vi4Irfj.png \n"
    "뇌의 구조와 기능을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/TAjDHDw.png \n"
    "다음 이미지를 사용해 뉴런의 종류 문제를 낼 수 있습니다: https://i.imgur.com/xvQfgIl.png \n 이미지에는 A, B, C 세 부분이 지정되어 있으며, A는 감각 뉴런, B는 연합 뉴런, C는 운동 뉴런입니다.\n"
    "다음 이미지를 사용해 뇌와 척수 문제를 낼 수 있습니다: https://i.imgur.com/IRgZv7Q.png \n 이미지에는 뇌의 단면에 A~F 세 부분이 지정되어 있으며, A는 간뇌, B는 중간뇌, C는 연수, D는 대뇌, E는 소뇌, F는 척수입니다. 이 이미지를 활용한 문항을 제시할 수 있습니다. (예: 어두운 곳에 들어가면 동공이 커지는 반응의 중추는 무엇인지 기호와 이름을 써 보자.)\n"
    "뇌하수체 호르몬(생장 호르몬, 갑상샘 자극 호르몬, 항이뇨 호르몬), 갑상샘 호르몬(티록신), 부신 호르몬(아드레날린), 이자 호르몬(인슐린, 글루카곤), 난소 호르몬(에스트로젠), 정소 호르몬(테스토스테론)을 그림으로 보여줄 때 다음 링크를 사용할 수 있습니다: https://i.imgur.com/bY0Xne5.png"
    "다음 이미지를 사용해 추울 때 우리 몸에서 일어나는 변화 문제를 낼 수 있습니다: https://i.imgur.com/V7512QP.png \n 이미지에는 ㉠, ㉡, ㉢, ㉣, A, B 여섯 부분이 지정되어 있습니다. 그림은 다음과 같은 구조입니다: \n피부의 냉점 → ( ㉠ ) → 피부 근처 혈관의 ( ㉡ ) → 열 발산 ( ㉢ )\n 피부의 냉점 → 뇌하수체 A 분비 증가 → 갑상샘 B 분비 증가 → 세포 호흡 촉진 → 열 발산 ( ㉣ )\n ㉠은 간뇌, ㉡은 수축, ㉢은 감소, ㉣은 증가 입니다. A는 갑상샘 자극 호르몬, B는 티록신 입니다.\n"
    "다음 이미지를 사용해 혈당량 조절 과정 문제를 낼 수 있습니다: https://i.imgur.com/S6uxW4b.png \n 이미지에는 A, B 두 부분이 지정되어 있습니다. 이자에서 호르몬 A가 분비되면 간에서 포도당→글리코젠으로 전환됩니다. 이자에서 호르몬 B가 분비되면 간에서 글리코젠→포도당으로 전환됩니다. 이 이미지를 활용한 문항을 제시할 수 있습니다. (예: 식사를 하여 혈당량이 높아졌을 때 분비량이 증가하는 것은 무엇인가?)\n"
)

SCIENCE_05_PROMPT = (
    "당신은 과학의 Ⅴ. 생식과 유전 단원 학습 지원을 담당합니다. \n"
)

SCIENCE_06_PROMPT = (
    "당신은 과학의 Ⅵ. 에너지 전환과 보존 단원 학습 지원을 담당합니다. \n"
)

def summarize_chunks(chunks, science_prompt, max_chunks=5):
    summaries = []
    for chunk in chunks[:max_chunks]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": COMMON_PROMPT},
                {"role": "system", "content": science_prompt},
                {"role": "system",
                 "content": "아래 텍스트를 앞서 언급된 키워드 중심으로 정리해 주세요."},
                {"role": "user",   "content": chunk}
            ]
        )
        summaries.append(resp.choices[0].message.content)
    return "\n\n".join(summaries)

# ===== Helpers =====
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

# RAG pipelines
def extract_text_from_pdf(path):
    if not os.path.exists(path):
        return ""
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)

def chunk_text(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed_texts(texts):
    if not texts:
        return []
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(d.embedding) for d in res.data]

def get_relevant_chunks(question, chunks, embeddings, top_k=3):
    if not chunks:
        return []
    q_emb = np.array(
        client.embeddings.create(
            model="text-embedding-3-small", input=[question]
        ).data[0].embedding
    )
    sims = [np.dot(q_emb, emb)/(np.linalg.norm(q_emb)*np.linalg.norm(emb)) for emb in embeddings]
    idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in idx]

# DB

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
        row = cur.fetchone()
        cur.close(); db.close()
        return json.loads(row[0]) if row else []
    except Exception as e:
        st.error(f"DB 오류: {e}")
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
        ts = datetime.now(ZoneInfo("Asia/Seoul"))
        cur.execute(sql, (
            num, name, code, subject, topic,
            json.dumps(chat, ensure_ascii=False), ts
        ))
        cur.close(); db.close()
    except Exception as e:
        st.error(f"DB 오류: {e}")

# Spinner 아이콘 정의

def show_stage(message):
    st.markdown(f"""
    <div style='display: flex; align-items: center; font-size: 18px;'>
        <div class="loader" style="
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 16px;
            height: 16px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        "></div>
        <div>{message}</div>
    </div>

    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Chat UI

def chatbot_tab(subject, topic):
    key = f"chat_{subject}_{topic}".replace(" ", "_")
    load_key = f"loading_{key}"
    input_key = f"buffer_{key}"
    widget_key_base = f"textarea_{key}"

    # 1) 세션 초기화
    if key not in st.session_state:
        st.session_state[key] = load_chat(subject, topic)
    if load_key not in st.session_state:
        st.session_state[load_key] = False
    msgs = st.session_state[key]

    # Select the appropriate science prompt for this unit
    science_prompts = {
        "Ⅳ. 자극과 반응": SCIENCE_04_PROMPT,
        "Ⅴ. 생식과 유전": SCIENCE_05_PROMPT,
        "Ⅵ. 에너지 전환과 보존": SCIENCE_06_PROMPT
    }
    selected_science_prompt = science_prompts.get(topic, "")

    # 2) 기존 메시지 렌더링
    for msg in msgs:
        if msg["role"] == "user":
            st.write(f"**You:** {msg['content']}")
        else:
            parts = re.split(r"(@@@@@.*?@@@@@)", msg['content'], flags=re.DOTALL)
            for part in parts:
                if part.startswith("@@@@@") and part.endswith("@@@@@"):
                    st.latex(part[5:-5].strip())
                else:
                    txt = clean_inline_latex(part)
                    for link in re.findall(r"(https?://\S+\.(?:png|jpg))", txt):
                        st.image(link)
                        txt = txt.replace(link, "")
                    if txt.strip():
                        st.write(f"**과학 도우미:** {txt.strip()}")

    # 3) 입력창 & 버튼 (토글 방식)
    placeholder = st.empty()
    if not st.session_state[load_key]:
        with placeholder.container():
            user_input = st.text_area("입력:", key=f"{widget_key_base}_{len(msgs)}")
            if st.button("전송", key=f"send_{key}_{len(msgs)}") and user_input.strip():
                st.session_state[input_key] = user_input.strip()
                st.session_state[load_key] = True
                st.rerun()

    # 4) 로딩 상태일 때만 OpenAI 호출 (하이브리드 방식)
    if st.session_state[load_key]:
        q = st.session_state.pop(input_key, "")
        if q:

            stage = st.empty()

            # PDF 전체 텍스트 읽기
            stage.empty()
            stage = st.empty()
            show_stage("교과서 검색 중...")
            time.sleep(0.5)
            texts = [extract_text_from_pdf(os.path.join(BASE_DIR, fn))
                     for fn in PDF_MAP[topic]]
            full = "\n\n".join(texts)

            # 디버깅용
#            st.write("🧪 사용 중인 파일:", PDF_MAP[topic])
#            st.write("📄 full 길이:", len(full))
#            st.write("📄 내용 일부:", full[:300])
#            for fn in PDF_MAP[topic]:
#                path = os.path.join(BASE_DIR, fn)
#                st.write(path, "존재 여부:", os.path.exists(path))

            # 한번만: 전체 요약 + embedding 캐시
            full_hash = hashlib.md5(full.encode("utf-8")).hexdigest()
            sum_key = f"sum_{subject}_{topic}".replace(" ", "_")

            # 1) 청크·임베딩 캐시
#            if 'chunks_embs' not in st.session_state:
#                chunks = chunk_text(full)
#                embs   = embed_texts(chunks)
#                st.session_state['chunks_embs'] = (chunks, embs)
        
#            chunks, embs = st.session_state['chunks_embs']

            # 질문마다: RAG로 연관 청크 검색
            stage.empty()
            stage = st.empty()
            show_stage("내용 분석 중...")
            time.sleep(0.5)
            chunks = chunk_text(full)
            embs   = embed_texts(chunks)
            relevant = get_relevant_chunks(q, chunks, embs, top_k=3)
#            st.write("📎 관련 청크 개수:", len(relevant))
#            st.write("🔍 청크 미리보기:", relevant)

            # 2) 질문 시: 상위 5개 청크만 가져와 답변 생성
            relevant = relevant[:5]

            stage.empty()
            stage = st.empty()
            show_stage("답변 생성 중...")
            time.sleep(0.5)
            system_messages = [
                {"role": "system", "content": COMMON_PROMPT},
                {"role": "system", "content": selected_science_prompt},
            ]

            history = st.session_state.get("history", [])

            rag_system_message = {
                "role": "system",
                "content": (
                    "아래 청크들은 교과서에서 발췌한 내용입니다. "
                    "질문과 관련된 청크만 참고해 답변하세요. "
                    "답변시 교과서의 표현을 철저하게 반영하세요:\n\n"
                    + "\n\n".join(relevant)
                )
            }

            prompt = system_messages + history + [
                rag_system_message,
                {"role": "user", "content": q}
            ]

            resp = client.chat.completions.create(model=MODEL, messages=prompt)
            ans = resp.choices[0].message.content
#            rag_info = f"🔍 참고한 내용:\n\n{'\n\n'.join(relevant)}\n\n"
#            ans = rag_info + ans
            stage.empty()
            ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
            msgs.extend([
                {"role": "user", "content": q, "timestamp": ts},
                {"role": "assistant", "content": ans}
            ])
            save_chat(subject, topic, msgs)
            st.session_state[key] = msgs
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
    st.markdown("> 🌟 “생각하건대 현재의 고난은 장차 우리에게 나타날 영광과 비교할 수 없도다” — 로마서 8장 18절")
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

    default_subject = "과목을 선택하세요."
    subject = st.selectbox(
        "과목을 선택하세요.",
        [default_subject] + list(SUBJECTS.keys())
    )
    if subject == default_subject:
        return

    default_unit = "대단원을 선택하세요."
    units = SUBJECTS[subject]  # 과목별 대단원 리스트
    unit = st.selectbox(
        "대단원을 선택하세요.",
        [default_unit] + units
    )
    if unit == default_unit:
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