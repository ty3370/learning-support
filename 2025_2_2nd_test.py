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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from io import BytesIO
import math
import json
import pathlib

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

DIAGRAM_PLANNER_PROMPT = (
    "당신은 '도형 선택 플래너'입니다. 아래 JSON 스키마로만 답하세요.\n"
    "질문이 순수 개념/정의 설명만으로 충분하면 need_diagram=false.\n"
    "그림이 학습에 유의미하면 need_diagram=true로 하고 하나의 도형만 고르세요.\n"
    "diagram.type은 다음 중 하나: "
    "['triangle_isosceles','triangle_right','parallelogram','quadrilateral_random','triangle_general']\n"
    "diagram.params는 길이/각 등 간단 파라미터, diagram.overlays는 보조선/원 표시를 부울로 포함합니다.\n"
    "가능한 overlays 키: ['angle_bisectors','perp_bisectors','altitudes','incircle','circumcircle','incenter','circumcenter','diagonals']\n"
    "출력은 반드시 아래 형식의 JSON 한 개만:\n"
    "{\n"
    '  "need_diagram": true|false,\n'
    '  "diagram": {\n'
    '    "type": "triangle_isosceles" | "triangle_right" | "parallelogram" | "quadrilateral_random" | "triangle_general",\n'
    '    "params": { /* 예: {"base":6, "side":5} 또는 {"a":3,"b":4} 또는 {"w":6,"h":3,"skew":2} */ },\n'
    '    "overlays": { "angle_bisectors":false, "perp_bisectors":false, "altitudes":false, "incircle":false, "circumcircle":false, "incenter":false, "circumcenter":false, "diagonals":false }\n'
    "  },\n"
    '  "caption": "학생용 간단 캡션 한 줄"\n'
    "}\n"
    "주의: need_diagram=false라면 'diagram'과 'caption'은 생략 가능합니다."
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
    text = re.sub(r"\^\s*\\circ", "°", text)
    text = re.sub(r"\^circ", "°", text)
    return text

def llm_plan_diagram(question, math_prompt, relevant_chunks):
    """LLM이 그림 필요 여부/종류를 JSON으로 결정하도록 1차 호출."""
    plan_messages = [
        {"role": "system", "content": COMMON_PROMPT},
        {"role": "system", "content": math_prompt},
        {"role": "system", "content": DIAGRAM_PLANNER_PROMPT},
        {"role": "system", "content": "다음은 교과서 관련 발췌입니다:\n" + "\n\n".join(relevant_chunks[:3])},
        {"role": "user", "content": question},
    ]
    try:
        resp = client.chat.completions.create(model=MODEL, messages=plan_messages)
        txt = resp.choices[0].message.content.strip()
        plan = json.loads(txt)
        if not isinstance(plan, dict): raise ValueError("planner JSON must be dict")
        return plan
    except Exception:
        return {"need_diagram": False}

def _facts_triangle_isosceles(A, B, C):
    # 기본 등식/수직이등분선 성질
    facts = {
        "type": "triangle_isosceles",
        "points": {"A": A, "B": B, "C": C},
        "equalities": ["AB=AC"], 
        "perpendicular": ["AD ⟂ BC"], 
        "bisectors": ["AD is angle bisector at A", "AD is perpendicular bisector of BC"]
    }
    return facts

def _facts_triangle_right(A, B, C, a, b):
    c = float(np.hypot(a, b))
    facts = {
        "type": "triangle_right",
        "points": {"A": A, "B": B, "C": C},
        "right_angle_at": "A",
        "lengths": {"AB": float(a), "AC": float(b), "BC": float(c)},
        "perpendicular": ["AB ⟂ AC"],
        "pythagoras": "AB^2 + AC^2 = BC^2"
    }
    return facts

def _facts_parallelogram(A, B, C, D):
    facts = {
        "type": "parallelogram",
        "points": {"A": A, "B": B, "C": C, "D": D},
        "parallel": ["AB ∥ CD", "AD ∥ BC"],
        "diagonals": "AC and BD bisect each other",
        "equal_sides": ["AB=CD", "AD=BC"]
    }
    return facts

def _facts_quad(A,B,C,D):
    facts = {
        "type": "quadrilateral",
        "points": {"A":A,"B":B,"C":C,"D":D},
        "note": "Diagonals drawn for comparison"
    }
    return facts

def render_diagram_from_spec(spec):
    """
    spec: {"type":..., "params":{...}, "overlays":{...}}
    반환: (img_path, facts_dict, spec_for_storage)
    """
    t = spec.get("type")
    params = spec.get("params", {}) or {}
    overlays = spec.get("overlays", {}) or {}
    # 기본 좌표는 각 그리기 함수 내부 정의와 일치시킵니다.

    if t == "triangle_isosceles":
        base = float(params.get("base", 6.0)); side = float(params.get("side", 5.0))
        img = draw_isosceles(base=base, side=side, overlays=overlays)
        A=(0.0,0.0); B=(base,0.0); C=(base/2.0, max(0.5, (side**2-(base/2.0)**2)**0.5) if side>base/2.0 else 3.0)
        facts = _facts_triangle_isosceles(A,B,C)

    elif t == "triangle_right":
        a = float(params.get("a", 3.0)); b = float(params.get("b", 4.0))
        seed = int(hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest(),16)%4
        img = draw_right_triangle(a=a, b=b, seed=seed, overlays=overlays, show_squares=bool(overlays.get("squares", False)))
        # 우리 구현의 기본 배치: A=(0,0), B=(a,0) 또는 회전/대칭. facts는 회전 무관 성질만 제공합니다.
        A=(0.0,0.0); B=(a,0.0); C=(0.0,b)
        facts = _facts_triangle_right(A,B,C,a,b)

    elif t == "parallelogram":
        w = float(params.get("w", 6.0)); h = float(params.get("h", 3.0)); skew=float(params.get("skew",2.0))
        img = draw_parallelogram(w=w,h=h,skew=skew,show_diagonals=bool(overlays.get("diagonals", True)))
        A=(0.0,0.0); B=(w,0.0); C=(w+skew,h); D=(skew,h)
        facts = _facts_parallelogram(A,B,C,D)

    elif t == "quadrilateral_random":
        img = draw_random_quadrilateral(show_diagonals=True)
        # 임의 사각형은 좌표를 런타임에 생성하므로, 여기서는 일반 설명만 남깁니다.
        facts = _facts_quad("A","B","C","D")

    else:  # triangle_general 등 → 등변 기본으로 fallback
        base = float(params.get("base", 6.0)); side = float(params.get("side", 5.0))
        img = draw_isosceles(base=base, side=side, overlays=overlays)
        A=(0.0,0.0); B=(base,0.0); C=(base/2.0, max(0.5, (side**2-(base/2.0)**2)**0.5) if side>base/2.0 else 3.0)
        facts = _facts_triangle_isosceles(A,B,C)

    return img, facts, spec

# ===== File-based persistence (No DB) =====
PERSIST_ROOT = pathlib.Path("./_persist")  # 프로젝트 폴더 내 저장소

def _safe(name: str) -> str:
    # 파일/폴더 이름에 쓸 수 있도록 최소 정제
    return "".join(ch for ch in name if ch.isalnum() or ch in ("-_",)).strip() or "unknown"

def _conv_dir(student_id: str, code: str, subject: str, topic: str) -> pathlib.Path:
    d = PERSIST_ROOT / _safe(student_id) / _safe(code) / _safe(subject) / _safe(topic)
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_conversation_file(student_id: str, code: str, subject: str, topic: str, msgs: list):
    d = _conv_dir(student_id, code, subject, topic)
    with open(d / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

def load_conversation_file(student_id: str, code: str, subject: str, topic: str) -> list:
    d = _conv_dir(student_id, code, subject, topic)
    p = d / "conversation.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

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
    if load_key not in st.session_state:
        st.session_state[load_key] = False

    if key not in st.session_state:
        st.session_state[key] = load_chat(subject, topic)

    msgs = st.session_state[key]

    math_prompts = {
        "Ⅳ. 도형의 성질": MATH_04_PROMPT
    }
    selected_math_prompt = math_prompts.get(topic, "")

    # 2) 기존 메시지 렌더링
    last_user_msg = None      # 직전 사용자 메시지
    for msg in msgs:
        if msg["role"] == "user":
            st.write(f"**You:** {msg['content']}")
            last_user_msg = msg["content"]
        else:
            raw = msg["content"]

            # 숨은 다이어그램 블록 추출(보여주지는 않음)
            spec_blocks = re.findall(r"```diagram_spec\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
            facts_blocks = re.findall(r"```diagram_facts\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
            # 실제 표시용 텍스트(숨은 블록 제거)
            visible = re.sub(r"```diagram_spec\s*\{.*?\}\s*```", "", raw, flags=re.DOTALL)
            visible = re.sub(r"```diagram_facts\s*\{.*?\}\s*```", "", visible, flags=re.DOTALL)

            # LaTeX 블록/이미지 링크/일반 텍스트 렌더
            parts = re.split(r"(@@@@@.*?@@@@@)", visible, flags=re.DOTALL)
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

            # 다이어그램 스펙이 저장되어 있으면, 여기서만 그림 생성/표시 (필요한 경우에만)
            if spec_blocks:
                try:
                    spec = json.loads(spec_blocks[-1])
                    img_path, _, _ = render_diagram_from_spec(spec)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption="도형(자동 생성)", use_container_width=True)
                except Exception:
                    pass

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
            system_messages = [
                {"role": "system", "content": COMMON_PROMPT},
                {"role": "system", "content": selected_math_prompt},
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

            # 1차 응답(텍스트) 생성: 교과서 RAG 컨텍스트 + 대화 기록 반영
            prompt = system_messages + history + [
                rag_system_message,
                {"role": "user", "content": q}
            ]
            resp_main = client.chat.completions.create(model=MODEL, messages=prompt)
            ans_text = resp_main.choices[0].message.content

            # 도식 필요성/유형 '계획' 프롬프트: LLM이 스스로 판단하게 함
            planner_system = {
                "role": "system",
                "content": (
                    "너는 수학 문제/설명에 필요한 도형을 '계획'하는 설계자다. "
                    "출력은 JSON 한 줄이어야 한다. 키: "
                    "{need_diagram: bool, reason: str, "
                    "shape: 'isosceles_triangle'|'right_triangle'|'incenter'|'circumcenter'|'parallelogram'|'quadrilateral', "
                    "labels: {A:[x,y],B:[x,y],C:[x,y],D:[x,y]}, "
                    "helpers: [{type:'altitude'|'bisector'|'median'|'perpendicular'|'parallel', from:'A'|'B'|'C'|'D', to:'B'|'C'|'D'|'side:AB'}, "
                    "…], caption: str}"
                )
            }
            planner_user = {
                "role": "user",
                "content": (
                    "학생 질문: " + q + "\n\n"
                    "관련 교과서 요약 청크:\n" + "\n\n".join(relevant) + "\n\n"
                    "위 맥락을 반영하여, 도형이 **필요하면** need_diagram=true로 하고, "
                    "필요 없으면 false로 하라. 좌표는 [0,1] 범위에서 대략 배치하면 된다."
                )
            }

            resp_plan = client.chat.completions.create(
                model=MODEL,
                messages=[planner_system, planner_user]
            )
            _raw_plan = resp_plan.choices[0].message.content.strip()

            # JSON 안전 파서
            def _extract_json(s: str):
                m = re.search(r"\{.*\}", s, flags=re.DOTALL)
                return json.loads(m.group(0)) if m else {"need_diagram": False}
            try:
                plan = _extract_json(_raw_plan)
            except Exception:
                plan = {"need_diagram": False}

            # 도형 렌더 함수(간단 버전) - 필요한 최소 유형만 지원
            from io import BytesIO
            import math
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            from matplotlib.lines import Line2D

            def _pt(name, labels, fallback):
                return tuple(labels.get(name, fallback))

            def render_diagram_from_plan(plan_dict):
                """plan_dict 기준으로 간단한 도형을 PNG 바이트로 반환"""
                fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
                ax.set_aspect("equal")
                ax.axis("off")

                labels = plan_dict.get("labels", {})
                shape  = plan_dict.get("shape", "quadrilateral")
                helpers = plan_dict.get("helpers", [])

                if shape in ("isosceles_triangle", "right_triangle", "incenter", "circumcenter"):
                    A = _pt("A", labels, (0.15, 0.2))
                    B = _pt("B", labels, (0.85, 0.2))
                    if shape == "isosceles_triangle":
                        C = _pt("C", labels, (0.5, 0.8))
                    elif shape == "right_triangle":
                        C = _pt("C", labels, (0.15, 0.8))
                    else:
                        C = _pt("C", labels, (0.55, 0.75))
                    poly = Polygon([A, B, C], fill=False)
                    ax.add_patch(poly)
                    for name, (x, y) in {"A": A, "B": B, "C": C}.items():
                        ax.plot([x], [y], marker="o")
                        ax.text(x, y, f" {name}", va="bottom", ha="left")

                    # 보조선 처리(아주 기본형)
                    for h in helpers:
                        t = h.get("type")
                        if t == "altitude" and h.get("from") == "A":
                            # A에서 BC에 내린 높이(근사)
                            x1, y1 = A; x2, y2 = B; x3, y3 = C
                            # BC 중점 근사
                            xm, ym = ( (x2+x3)/2, (y2+y3)/2 )
                            ax.add_line(Line2D([x1, xm], [y1, ym], linestyle="--"))
                        if t == "bisector" and h.get("from") in ("A","B","C"):
                            # 각의 이등분선(근사)
                            which = h.get("from")
                            P = {"A":A,"B":B,"C":C}[which]
                            ax.add_line(Line2D([P[0], 0.5], [P[1], 0.5], linestyle="--"))

                elif shape in ("parallelogram", "quadrilateral"):
                    A = _pt("A", labels, (0.2, 0.2))
                    B = _pt("B", labels, (0.8, 0.25))
                    D = _pt("D", labels, (0.35, 0.8))
                    if shape == "parallelogram":
                        # 평행사변형: B-A와 D-A 벡터를 이용
                        vx, vy = (B[0]-A[0], B[1]-A[1])
                        wx, wy = (D[0]-A[0], D[1]-A[1])
                        C = (A[0]+vx+wx, A[1]+vy+wy)
                    else:
                        C = _pt("C", labels, (0.85, 0.75))
                    poly = Polygon([A, B, C, D], fill=False)
                    ax.add_patch(poly)
                    for name, (x, y) in {"A":A,"B":B,"C":C,"D":D}.items():
                        ax.plot([x], [y], marker="o")
                        ax.text(x, y, f" {name}", va="bottom", ha="left")

                    # 보조선
                    for h in helpers:
                        if h.get("type") == "diagonal" and h.get("from") in ("A","B","C","D") and h.get("to") in ("A","B","C","D"):
                            P = {"A":A,"B":B,"C":C,"D":D}[h["from"]]
                            Q = {"A":A,"B":B,"C":C,"D":D}[h["to"]]
                            ax.add_line(Line2D([P[0], Q[0]],[P[1], Q[1]], linestyle="--"))

                buf = BytesIO()
                fig.tight_layout(pad=0.3)
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                return buf

            # 필요 시 도형 생성 및 화면 표시(+ LLM이 이해할 수 있도록 spec도 메시지에 포함)
            image_caption = ""
            if plan.get("need_diagram"):
                buf = render_diagram_from_plan(plan)
                image_caption = plan.get("caption", "")
                st.image(buf, caption=image_caption, use_container_width=True)

                # LLM이 도형을 '이해'하도록, 도형 설계 사양을 함께 저장(텍스트로)
                spec_text = "```diagram_spec\n" + json.dumps(plan, ensure_ascii=False) + "\n```"
                ans_text = spec_text + "\n\n" + ans_text

            # 6) 메시지 반영 및 저장
            stage.empty()
            ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
            msgs.extend([
                {"role": "user", "content": q, "timestamp": ts},
                {"role": "assistant", "content": ans_text}
            ])

            # 세션에 반영
            st.session_state[key] = msgs

            # 저장
            save_chat(subject, topic, msgs)

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
        이 시스템은 중2 학생들을 위한 AI 학습 도우미입니다.

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