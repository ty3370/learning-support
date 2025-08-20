import streamlit as st
import pymysql
import json
from datetime import datetime
from openai import OpenAI
import re
from zoneinfo import ZoneInfo
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import hashlib
import time
import uuid
import base64

# ===== Configuration =====
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4o"
BASE_DIR = os.path.join(os.getcwd(), "Textbook_2025")
PDF_MAP = {
    "Ⅳ. 도형의 성질": ["2025_Math_2nd_04.pdf"]
}
SUBJECTS = {"2학년 수학": list(PDF_MAP.keys())}

# Initial prompts
COMMON_PROMPT = (
    "당신은 중학교 2학년 학생들의 학습을 돕는 AI 튜터입니다.\n"
    "답할 수 없는 정보(시험 범위, 시험 날짜 등)에 대해선 선생님께 문의하도록 안내하세요.\n"
    "따뜻하고 친근한 말투로 존댓말을 사용해 주세요. 학생이 편하게 느낄 수 있도록 상황에 맞는 다양한 이모지, 느낌표 등을 자연스럽게 활용하세요.\n"
    "당신은 학생들이 질문하는 내용에 답하거나, 문제를 내줄 수 있습니다. 중학생 수준에 맞게 차근차근 설명해 주세요.\n"
    "당신은 철저하게 교과서 내용에 근거하여 설명과 문항을 제공해야 합니다.\n"
    "모든 수식은 반드시 LaTeX 형식으로 작성하고 '@@@@@'로 감싸주세요. 수식 앞뒤에는 반드시 빈 줄로 구분해 주세요. 이 규칙은 어떤 경우에도 반드시 지켜야 합니다. 예시:\n\n@@@@@\nE_p = 9.8 \\times m \\times h\n@@@@@\n\n"
    "절대로 문장 중간에 LaTeX 형식이 들어가선 안 됩니다. LaTex 사용은 반드시 줄바꿈하고, LaTex 앞뒤를 각각 @ 기호 5개로 감싸야 합니다.\ "
    "틀린 표현 예시: 어떤 물체의 질량이 2kg이고 높이가 10m일 때 위치에너지는((E_p = 9.8 \\times m \\times h))입니다.\n"
    "맞는 표현 예시: 어떤 물체의 질량이 2kg이고 높이가 10m일 때 위치에너지는 다음과 같이 계산할 수 있습니다:\n\n@@@@@\nE_p = 9.8 \\times m \\times h\n@@@@@\n\n"
    "만약 LaTex를 줄바꿈 없이 사용해야만 하는 상황이라면, LaTex가 아닌 글로 쓰세요. \n틀린 표현 예시: 위치에너지는 9.8 \\times m \\times h입니다. \n맞는 표현 예시: 위치에너지는 9.8×m×h입니다. LaTex를 쓰려면 반드시 앞뒤로 줄바꿈해야 합니다.\n"
    "그림을 출력해야 하는 경우, 링크를 답변에 포함하면 자동으로 그림이 출력됩니다. 따로 하이퍼링크를 만들 필요가 없습니다.\n"
    "대화 예시: 눈의 구조는 아래 그림을 참고하세요. \n\n https://i.imgur.com/BIFjdBj.png \n"
    "학생이 문제를 내달라고 하면, 교과서에 나오는 내용에 철저하게 기반해서 출제해 주세요. 한 번에 여러 개의 문제를 달라는 명시적인 요청이 없다면, 하나의 대화에서는 한 문제만 내세요.\n"
    "만약 학생이 어려운 문제, 난이도 높은 문제를 달라고 한다면, 개인마다 잘 하는 것과 부족한 것이 다르기 때문에 어렵다고 느끼는 문항도 개인별로 다르니 무엇을 잘 하고 못하는지에 대한 파악이 우선되어야 한다고 안내하세요. 내용 자체가 이해되지 않는 것인지, 내용은 이해하지만 문제에 적용하는 것이 어려운 건지, 텍스트·그림·표·그래프 등의 자료 해석이 어려운 건지, 서술형 답을 쓰는 게 어려운 건지 등 무엇을 어렵다고 느끼는 지 상담하며 진단하세요.\n"
    "생성한 응답이 너무 길어지면 학생이 이해하기 어려울 수 있으므로, 한 줄 이내로 짧고 간결하게 응답하세요. 한 줄을 넘을 수 밖에 없는 경우, 모든 정보를 한 번에 제시하지 말고 학생과 대화가 오가며 순차적으로 한 줄씩 설명하세요.\n"
    "안 좋은 설명의 예(한 번에 설명): 동공은 빛의 양에 따라 크기가 달라지는데, 어두울 때는 동공이 커지고 밝을 때는 작아집니다. 이는 홍채가 수축하거나 이완하기 때문이며, 동공은 눈으로 들어오는 빛의 양을 조절해줍니다.\n"
    "좋은 설명의 예(순차적 설명): 먼저 동공과 홍채의 관계에 대해 생각해 봅시다. 홍채가 작아지면 동공이 커지고, 홍채가 커지면 동공이 작아져요. 여기까지 이해가 됐나요? (학생의 대답에 따라 이어서 진행)\n"
    "학생이 전반적인 내용을 요약해달라고 요청할 경우에도, 마찬가지로 일부 내용만 요약해 제시한 뒤 이어서 계속 요약하냐고 묻고, 학생이 계속해 달라고 하면 이어서 요약본을 제시하세요. 이런 방법으로 하나의 대화가 지나치게 길어지지 않도록 조절하세요.\n"
    "가독성이 좋도록 적절히 줄바꿈으로 하고 개조식으로 답변하세요."
    "풀이 과정이 복잡한 문제에서 답이 부정확한 경우가 종종 있으니, 반드시 Chain-of-Thought 방식으로 단계별로 검토하며 답하세요. 계산 문제나 판단이 필요한 경우, 짧게 쓰더라도 중간 과정이나 이유를 간단히 보여 주세요.\n"
    "학생이 문제를 틀렸는데 맞혔다고 하는 경우가 빈번합니다. 풀이를 먼저 검토하고 정답 여부를 결정하세요.\n"
    "학생이 문제를 틀렸을 경우, 위의 예시와 마찬가지로 한 번에 모든 풀이를 알려주지 말고 순차적으로 질문을 제시하며 학생 스스로 깨달을 수 있게 유도하세요.\n"
    "이미지를 출력거나 웹으로 연결할 때는 링크가 한 글자도 틀려선 안 됩니다. 오탈자 없이 출력하고, 초기 프롬프트에 포함된 링크 외에는 어떠한 링크도 제시하지 마세요.\n"
    "정보 제공을 목적으로 하지 말고, 학생에게 단계적 스캐폴딩을 제공하며 학생 스스로 깨닫도록 하는 것을 목적으로 하세요."
)

MATH_04_PROMPT = (
    "당신은 수학의 Ⅳ. 도형의 성질 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "이등변 삼각형, 이등변 삼각형의 성질, 꼭지각의 이등분선, 이등변 삼각형이 되는 조건 \n"
    "직각삼각형의 합동 조건, RHA합동, RHS합동 \n"
    "피타고라스 정리, \n\n@@@@@\na^2 + b^2 = c^2\n@@@@@\n\n"
    "접한다, 접선, 접점, 내접 내접원, 내심, 외접, 외접원, 외심 \n"
    "평행사변형, 평행사변형의 성질, 평행사변형이 되는 조건 \n"
    "여러 가지 사각형의 성질, 여러 가지 사각형 사이의 관계 \n"
    "직사각형, 직사각형의 성질, 마름모, 마름모의 성질, 정사각형, 정사각형의 성질 \n\n"
    "2. 학습 지원 지침\n"
    "설명 시 이미지를 사용해도 되고, 이미지 없이 텍스트로만 설명해도 됩니다. 문제를 낼 때도 텍스트로만 이루어진 문제, 표로 정보가 제공되는 문제, 이미지를 해석하는 문제, 선택형 문제, 서술형 문제 등을 다양하게 출제하세요. \n"
    "하나의 대화에서는 하나의 그림만을 사용하세요. \n\n"
    "3. 사용 가능한 이미지 목록:\n"
    "학생에게 내용을 설명하거나 문제를 내줄 때, 필요하다면 그 내용 또는 문제에 해당하는 도형을 자동 생성하여 함께 제시합니다. \n"
)

MATH_05_PROMPT = (
    "당신은 수학의 Ⅴ. XXX 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "2. 학습 지원 지침\n"
    "3. 사용 가능한 이미지 목록:\n"
)

MATH_06_PROMPT = (
    "당신은 수학의 Ⅴ. XXX 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "2. 학습 지원 지침\n"
    "3. 사용 가능한 이미지 목록:\n"
)

def summarize_chunks(chunks, math_prompt, max_chunks=3):
    summaries = []
    for chunk in chunks[:max_chunks]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": COMMON_PROMPT},
                {"role": "system", "content": math_prompt},
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

def _save_fig_return_path(fig, fname="diagram.png"):
    buf = BytesIO()
    fig.savefig(buf, bbox_inches="tight", dpi=160)
    buf.seek(0)
    path = os.path.join(os.getcwd(), fname)
    with open(path, "wb") as f:
        f.write(buf.read())
    plt.close(fig)
    return path

def generate_diagram_image(prompt: str, size: str = "auto") -> str:
    """
    LLM이 전달한 diagram_prompt로 도형 이미지를 생성하고, 로컬 파일 경로를 반환합니다.
    - OpenAI Images API를 사용 (model: gpt-image-1)
    - size: '1024x1024' | '1024x1536' | '1536x1024' | 'auto'
    """
    try:
        # 허용 크기 검증 및 폴백
        allowed = {"1024x1024", "1024x1536", "1536x1024"}
        if size == "auto":
            sz = "1024x1024"
        else:
            sz = size if size in allowed else "1024x1024"

        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=sz,
            n=1
        )
        b64 = result.data[0].b64_json
        filename = os.path.join(os.getcwd(), f"diagram_{uuid.uuid4().hex}.png")
        with open(filename, "wb") as f:
            f.write(base64.b64decode(b64))
        return filename
    except Exception as e:
        st.warning(f"도형 이미지 생성 실패: {e}")
        return ""

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

    # Select the appropriate math prompt for this unit
    math_prompts = {
        "Ⅳ. 도형의 성질": MATH_04_PROMPT
    }
    selected_math_prompt = math_prompts.get(topic, "")

    # 2) 기존 메시지 렌더링
    last_user_msg = None      # 직전 사용자 메시지
    last_assistant = None     # 직전 어시스턴트 메시지(답변 본문)

    for msg in msgs:
        if msg["role"] == "user":
            st.write(f"**You:** {msg['content']}")
            last_user_msg = msg["content"]
        else:
            # assistant 메시지 렌더링
            parts = re.split(r"(@@@@@.*?@@@@@)", msg['content'], flags=re.DOTALL)
            rendered_text = []
            for part in parts:
                if part.startswith("@@@@@") and part.endswith("@@@@@"):
                    st.latex(part[5:-5].strip())
                else:
                    txt = clean_inline_latex(part)
                    for link in re.findall(r"(https?://\S+\.(?:png|jpg))", txt):
                        st.image(link)
                        txt = txt.replace(link, "")
                    if txt.strip():
                        st.write(f"**학습 도우미:** {txt.strip()}")
                        rendered_text.append(txt.strip())
            last_assistant = "\n".join(rendered_text) if rendered_text else None

            # 조건부: 이 메시지에 '미리 생성된 도형' 정보가 있으면 표시
            if msg.get("need_diagram") and msg.get("diagram_image_path"):
                st.image(msg["diagram_image_path"], caption="AI가 생성한 그림으로, 부정확할 수 있습니다.")

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

            # 2) 질문 시: 상위 3개 청크만 가져와 답변 생성
            relevant = relevant[:3]

            stage.empty()
            stage = st.empty()
            show_stage("답변 생성 중...")
            time.sleep(0.5)

            # ── LLM 시스템 지시: JSON 형식 강제 ─────────────────────────────
            system_messages = [
                {"role": "system", "content": COMMON_PROMPT},
                {"role": "system", "content": selected_math_prompt},
                {"role": "system", "content":
                    (
                        "이후 모든 답변은 반드시 다음 JSON 형식 '한 개'로만 출력하세요.\n"
                        "{\n"
                        '  "answer": "<학생에게 보여줄 최종 답변 텍스트 — LaTeX 규칙(@@@@@) 준수>",\n'
                        '  "need_diagram": true | false,\n'
                        '  "diagram_prompt": "<그려야 할 도형을 한국어로 간결히 설명 — 한 장의 그림 기준, 필요한 보조선/표시 포함; 필요 없으면 빈 문자열>",\n'
                        '  "diagram_size": "auto | 1024x1024 | 1024x1536 | 1536x1024"\n'
                        "}\n"
                        "그림이 불필요하면 need_diagram=false로 하고, diagram_prompt는 빈 문자열로 두세요. "
                        "한 대화에서는 하나의 그림만 사용합니다."
                        "도형이나 위치관계 설명이 포함되면 가급적 need_diagram=true 로 설정합니다."
                    )
                },
            ]

            history = [{"role": msg["role"], "content": msg["content"]} for msg in msgs]

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
            raw = resp.choices[0].message.content

            # ── JSON 강제 추출: 코드펜스/서문 제거 후 첫 번째 {...}만 파싱 ─────────
            raw = raw.strip()

            # 코드펜스 제거
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            # 본문에서 첫 번째 JSON 객체만 추출
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                json_str = m.group(0)
                try:
                    parsed = json.loads(json_str)
                    ans = (parsed.get("answer") or "").strip()
                    need_diagram = bool(parsed.get("need_diagram", False))
                    diagram_prompt = (parsed.get("diagram_prompt") or "").strip()
                    diagram_size = (parsed.get("diagram_size") or "auto").strip()
                except Exception:
                    # JSON이 잡혔지만 파싱 실패 → 전체를 텍스트로 표기(최후 방어)
                    ans = raw
                    need_diagram = False
                    diagram_prompt = ""
                    diagram_size = "auto"
            else:
                # JSON 블록 자체가 없으면 전부 텍스트로 간주
                ans = raw
                need_diagram = False
                diagram_prompt = ""
                diagram_size = "auto"

            # ── (조건부) 도형 즉시 생성: show_stage 표시 후 이미지 생성 ────────
            diagram_image_path = None
            if need_diagram:
                # 🔒 청크 영향 제거: LLM이 준 diagram_prompt는 쓰지 않고, 질문(q)만 사용
                diagram_prompt = (
                    "다음 문제를 한 장의 단순한 도형으로 표현하세요. "
                    "필요한 보조선/각도/표시는 최소로 하고, 텍스트 표기는 최소화합니다. "
                    f"문제 요약: {q[:180]}"
                )

                stage.empty()
                stage = st.empty()
                show_stage("그림 생성 중...")
                time.sleep(0.3)
                diagram_image_path = generate_diagram_image(diagram_prompt, size=diagram_size)

            stage.empty()
            ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
            msgs.extend([
                {"role": "user", "content": q, "timestamp": ts},
                {
                    "role": "assistant",
                    "content": ans,
                    "need_diagram": need_diagram,
                    "diagram_prompt": diagram_prompt,
                    "diagram_image_path": diagram_image_path
                }
            ])
            save_chat(subject, topic, msgs)
            st.session_state[key] = msgs
            st.session_state[load_key] = False
            st.rerun()

# ===== Pages =====
def page_1():
    st.title("2025-2학기 보라중 학습 도우미")
    st.write("2학년용 학습 도우미 AI입니다. 아래에 학습자 정보를 입력하세요.")
    st.session_state['user_number'] = st.text_input("학번", value=st.session_state.get('user_number',''))
    st.session_state['user_name'] = st.text_input("이름", value=st.session_state.get('user_name',''))
    st.session_state['user_code'] = st.text_input("식별코드", value=st.session_state.get('user_code',''),
        help="타인의 학번과 이름으로 접속하는 것을 방지하기 위해 자신만 기억할 수 있는 코드를 입력하세요.")
    st.markdown("> 🌟 “생각하건대 현재의 고난은 장차 우리에게 나타날 영광과 비교할 수 없도다” — 로마서 8장 18절")
    if st.button("다음"):
        if not all([st.session_state['user_number'].strip(), st.session_state['user_name'].strip(), st.session_state['user_code'].strip()]):
            st.error("모든 정보를 입력해주세요.")
        else:
            st.session_state['step']=3; st.rerun()

def page_2(): # 현재 생략되어 있음
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
    st.markdown("❗ AI의 답변은 부정확할 수 있습니다. 의심스러운 정보는 반드시 교과 선생님께 직접 확인하세요.")

    default_subject = "과목을 선택하세요."
    subject = st.selectbox(
        "과목을 선택하세요.",
        [default_subject] + list(SUBJECTS.keys())
    )
    if subject == default_subject:
        return

    default_unit = "단원을 선택하세요."
    units = SUBJECTS[subject]  # 과목별 단원 리스트
    unit = st.selectbox(
        "단원을 선택하세요.",
        [default_unit] + units
    )
    if unit == default_unit:
        return

    # 단원이 바뀔 때 세션 상태 초기화
    if "prev_unit" not in st.session_state:
        st.session_state["prev_unit"] = unit

    if unit != st.session_state["prev_unit"]:
        for k in list(st.session_state.keys()):
            if k.startswith("chat_") or k.startswith("buffer_") or k.startswith("loading_") or k.startswith("textarea_"):
                del st.session_state[k]
        st.session_state["prev_unit"] = unit

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