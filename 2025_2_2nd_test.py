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
    # 수학
    "Ⅰ. 수와 식의 계산": ["2025_Math_2nd_01.pdf"],
    "Ⅳ. 도형의 성질": ["2025_Math_2nd_04.pdf"],
    "Ⅴ. 도형의 닮음": ["2025_Math_2nd_05.pdf"],
    "Ⅵ. 확률": ["2025_Math_2nd_06.pdf"],

    # 과학
    "Ⅳ. 식물과 에너지": ["2025_Sci_2nd_04.pdf"],
    "Ⅴ. 동물과 에너지": ["2025_Sci_2nd_05.pdf"],
    "Ⅵ. 물질의 특성": ["2025_Sci_2nd_06.pdf"],
    "Ⅶ. 수권과 해수의 순환": ["2025_Sci_2nd_07.pdf"],
}
SUBJECTS = {
    "2학년 과학": ["Ⅳ. 식물과 에너지", "Ⅴ. 동물과 에너지", "Ⅵ. 물질의 특성", "Ⅶ. 수권과 해수의 순환"],
    "2학년 수학": ["Ⅰ. 수와 식의 계산", "Ⅳ. 도형의 성질", "Ⅴ. 도형의 닮음", "Ⅵ. 확률"],
}

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

SCIENCE_04_PROMPT = (
    "당신은 과학의 Ⅳ. 식물과 에너지 단원 학습 지원을 담당합니다. 아래 1~2를 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "광합성, 엽록체, 광합성에 영향을 미치는 요인(빛의 세기, 이산화 탄소의 농도, 온도), 증산 작용\n"
    "식물의 호흡, 광합성 산물의 이동·저장·사용\n"
    "2. 학습 지원 지침\n"
    "가능한 한 교과서 표현을 그대로 사용하세요.\n"
    "문제를 낼 때 단순 개념 문제, 개념을 일상생활 상황에 적용해 해석하는 문제, 표를 해석하는 문제, 선택형 문제, 서술형 문제 등을 다양하게 출제하세요.\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이 단원에서는 사용 가능한 이미지가 없습니다. 이미지를 사용하지 마세요. \n"
)

SCIENCE_05_PROMPT = (
    "당신은 과학의 Ⅴ. 동물과 에너지 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "식물의 구성 단계(세포-조직-조직계-기관-개체), 동물의 구성 단계(세포-조직-기관-기관계-개체), 영양소의 종류와 기능(탄수화물, 단백질, 지방, 무기염류, 비타민, 물)\n"
    "소화계, 소화기관(입, 식도, 위, 소장, 대장, 침샘, 간, 쓸개, 이자), 녹말-(침샘의 아밀레이스, 이자액의 아밀레이스)-엿당-(소장의 탄수화물 분해 효소)-포도당, 단백질-(위액의 펩신, 이자액의 트립신)-중간 산물-(소장의 단백질 분해 효소)-아미노산, 지방-(쓸개즙)-지방-(이자액의 라이페이스)-모노글리세리드와 지방산, 수용성 영양소, 지용성 영양소\n"
    "순환계, 혈액의 구성(적혈구, 백혈구, 혈소판, 혈장), 동맥, 모세 혈관, 정맥, 심장의 구조(우심방, 좌심방, 우심실, 좌심실, 대정맥, 대동맥, 폐동맥, 폐정맥), 심장 박동의 원리(심방과 심실 이완, 심방 수축, 심실 수축), 폐순환, 온몸 순환\n"
    "호흡계, 호흡 기관(코, 기관, 기관지, 폐), 폐포, 갈비뼈(늑골), 흉강, 횡격막(가로막), 호흡 운동(들숨, 날숨)\n"
    "배설계, 콩팥, 오줌관, 방광, 요도, 네프론(사구체, 보먼주머니, 세뇨관), 여과, 재흡수, 분비\n"
    "세포 호흡\n"
    "2. 학습 지원 지침\n"
    "가능한 한 교과서 표현을 그대로 사용하세요.\n"
    "문제를 낼 때 단순 개념 문제, 개념을 일상생활 상황에 적용해 해석하는 문제, 표를 해석하는 문제, 선택형 문제, 서술형 문제 등을 다양하게 출제하세요.\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이 단원에서는 사용 가능한 이미지가 없습니다. 이미지를 사용하지 마세요. \n"
)

SCIENCE_06_PROMPT = (
    "당신은 과학의 Ⅵ. 물질의 특성 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "순물질, 혼합물, 물질의 특성, 밀도, 용해, 포화 용액, 불포화 용액, 용해도, 끓는점, 녹는점, 어는점\n"
    "증류, 밀도 차를 이용하는 분리, 재결정, 크로마토그래피\n"
    "2. 학습 지원 지침\n"
    "가능한 한 교과서 표현을 그대로 사용하세요.\n"
    "문제를 낼 때 단순 개념 문제, 개념을 일상생활 상황에 적용해 해석하는 문제, 표를 해석하는 문제, 선택형 문제, 서술형 문제 등을 다양하게 출제하세요.\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이 단원에서는 사용 가능한 이미지가 없습니다. 이미지를 사용하지 마세요. \n"
)

SCIENCE_07_PROMPT = (
    "당신은 과학의 Ⅶ. 수권과 해수의 순환 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "수권, 수자원, 해수의 연직 수온 분포(혼합층, 수온 약층, 심해층), 염류, 염분, 염분비 일정 법칙\n"
    "해류(난류, 한류), 우리나라 주변의 해류(쿠로시오 해류, 북한 한류, 황해 난류, 서한 연안류, 동한 난류, 중국 연안류), 조석, 조류, 만조, 간조, 조차\n"
    "2. 학습 지원 지침\n"
    "가능한 한 교과서 표현을 그대로 사용하세요.\n"
    "문제를 낼 때 단순 개념 문제, 개념을 일상생활 상황에 적용해 해석하는 문제, 표를 해석하는 문제, 선택형 문제, 서술형 문제 등을 다양하게 출제하세요.\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이 단원에서는 사용 가능한 이미지가 없습니다. 이미지를 사용하지 마세요. \n"
)

MATH_01_PROMPT = (
    "당신은 수학의 Ⅰ. 수와 식의 계산 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "소수, 합성수, 밑, 지수, 거듭제곱\n"
    "소인수분해, 최대공약수, 최소공배수\n"
    "정수와 유리수, 양의정수, 음의정수, 양수, 음수\n"
    "절댓값, 두 수의 크기비교, 수의 대소관계\n"
    "정수와 유리수의 덧셈, 뺄셈, 곱셈, 나눗셈, 혼합계산\n\n"
    "2. 학습 지원 지침\n"
    "학생의 답이 틀렸는데 맞았다고 하는 경우가 있으니, 문제의 정답은 반드시 신중하게 검토하세요.\n"
    "수학 교과에서는 LaTex 작성 규칙(수식 앞뒤를 줄바꿈하고 @@@@@로 감싸는 것)을 지키는 것이 매우 중요합니다. LaTex 규칙을 절대적으로 지키세요.\n"
    "학생에게 문제를 낼 때는 하나의 대화에 한 문제만 내세요.\n"
    "정수 계산식 문제를 낼 경우, 괄호 표현을 제대로 하세요.\n 잘못된 예시: -5) × (3 + 2 \n 올바른 예시: (-5) × (3 + 2)\n"
    "문제를 낸 뒤에는 정답과 풀이까지 제대로 확인을 하고 다음 대화를 진행하세요.\n\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이 단원에서는 사용 가능한 이미지가 없습니다. 이미지를 사용하지 마세요. \n"
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
    "하나의 대화에서는 하나의 그림만을 사용하세요. \n"
    "학생의 답이 틀렸는데 맞았다고 하는 경우가 있으니, 문제의 정답은 반드시 신중하게 검토하세요.\n"
    "수학 교과에서는 LaTex 작성 규칙(수식 앞뒤를 줄바꿈하고 @@@@@로 감싸는 것)을 지키는 것이 매우 중요합니다. LaTex 규칙을 절대적으로 지키세요.\n\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이등변삼각형의 성질을 설명하거나, 이등변삼각형의 성질 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/eTNB9WQ.png \n 이미지에는 AB=AC인 이등변삼각형 ABC(AB=AC)에서 ∠A의 이등분선을 그어 밑변 BC와의 교점을 D로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: ∠ABD가 65º일 때 ∠BAC의 크기는? 답: 50º)\n"
    "직각삼각형의 합동 조건을 설명하거나, 직각삼각형의 합동 조건 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/ePZvXJy.png \n 이미지에는 ∠C=∠F=90º인 두 직각삼각형 ABC와 DEF가 제시되어 있습니다. ∠B=∠E>∠A=∠D, AB=DE(빗변), BC=FE<AC=DF입니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: 특정 각도나 길이를 제시하고, 두 삼각형이 합동인지 아닌지 파악하는 문제)\n"
    "피타고라스 정리를 설명하거나, 피타고라스 정리 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/NcgWK9P.png \n 이미지에는 ∠C=90º인 직각삼각형 ABC가 제시되어 있으며, a^2+b^2=c^2입니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: a와 b의 길이를 제시하고 c의 길이를 구하도록 하는 문제)\n"
    "삼각형의 내심을 설명하거나, 삼각형의 내심 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/5whl3n8.png \n 이미지에는 △ABC에서 ∠A와 ∠B의 이등분선의 교점을 I라 하고, 점 I에서 삼각형의 각 변에 내린 수선의 발을 각각 D, E, F로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: ∠BAI=40º, ∠CBI=30º일 때, ∠ACI=? 답: 20º)\n"
    "삼각형의 외심을 설명하거나, 삼각형의 외심 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/9HioMmp.png \n 이미지에는 △ABC에서 변 AB와 변 AC의 수직이등분선의 교점을 O라 하고, 점 O에서 변 BC에 내린 수선의 발을 D로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: OC의 길이가 7cm일 때, OA의 길이는? 답: 7cm)\n"
    "평행사변형을 설명하거나, 평행사변형 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/VZ58iNo.png \n 이미지에는 AB//DC이고 AD//BC인 평행사변형이 제시되어 있습니다. ∠A=∠C>∠B=∠D입니다. 변의 길이는 AD=BC>AB=DC입니다. 대각선 AC와 BD의 교점은 O로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: ∠BAD=105º일 때, ∠ABC는? 답: 75º)\n"
    "직사각형을 설명하거나, 직사각형 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/UfG0xij.png \n 이미지에는 AB//DC이고 AD//BC인 직사각형이 제시되어 있습니다. 변의 길이는 AD=BC>AB=DC입니다. 대각선 AC와 BD의 교점은 O로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: BD=16cm일 때, AC의 길이는? 답: 8cm)\n"
    "마름모를 설명하거나, 마름모 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/N6R1kej.png \n 이미지에는 AB//DC이고 AD//BC인 마름모가 제시되어 있습니다. ∠A=∠C>∠B=∠D입니다. 대각선 AC와 BD의 교점은 O로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: ∠OAD=60º일 때, ∠ABO는? 답: 30º)\n"
    "정사각형을 설명하거나, 정사각형 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/8T8OAeH.png \n 이미지에는 AB//DC이고 AD//BC인 정사각형이 제시되어 있습니다. 대각선 AC와 BD의 교점은 O로 표시했습니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. \n"
)

MATH_05_PROMPT = (
    "당신은 수학의 Ⅴ. 도형의 닮음 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "닮음, 닮음비 \n\n"
    "@@@@@\n△ABC \\backsim △DEF\n@@@@@\n\n"
    "평면도형에서의 닮음의 성질, 입체도형에서의 닮음의 성질 \n\n"
    "닮은 두 입체도형의 넓이의 비 \n\n@@@@@\nm^2 : n^2\n@@@@@\n\n"
    "부피의 비 \n\n@@@@@\nm^3 : n^3\n@@@@@\n\n"
    "삼각형의 닮음 조건, SSS닮음, SAS닮음, AA닮음 \n"
    "평행선 사이의 선분의 길이의 비 \n"
    "중선, 무게중심, 삼각형의 무게중심 \n\n"
    "2. 학습 지원 지침\n"
    "설명 시 이미지를 사용해도 되고, 이미지 없이 텍스트로만 설명해도 됩니다. 문제를 낼 때도 텍스트로만 이루어진 문제, 표로 정보가 제공되는 문제, 이미지를 해석하는 문제, 선택형 문제, 서술형 문제 등을 다양하게 출제하세요. \n"
    "하나의 대화에서는 하나의 그림만을 사용하세요. \n"
    "학생의 답이 틀렸는데 맞았다고 하는 경우가 있으니, 문제의 정답은 반드시 신중하게 검토하세요.\n"
    "수학 교과에서는 LaTex 작성 규칙(수식 앞뒤를 줄바꿈하고 @@@@@로 감싸는 것)을 지키는 것이 매우 중요합니다. LaTex 규칙을 절대적으로 지키세요.\n\n"
    "3. 사용 가능한 이미지 목록:\n"
    "삼각형을 닮음을 설명하거나, 삼각형의 닮음 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/7ts4M8I.png \n 이미지에는 닮은 도형인 두 삼각형 △ABC와 △DEF가 있습니다. AB:DE=BC:EF=CA:FD이고 ∠A=∠D, ∠B=∠E, ∠C=∠F입니다. △DEF가 더 큰 삼각형입니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: AB가 5cm, AC가 3cm, DE가 10cm 라면 DF의 길이는? 답: 6cm)\n"
    "평행선에서 선분의 길이비를 설명하거나, 평행선에서 선분의 길이비 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/JBbYjSR.png \n 이미지에는 세 개의 평행선 l, m, n이 다른 두 직선과 만나서 생긴 선분을 보여줍니다. 그림에서 l // m // n이고, a:b=c:d 입니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: a가 1cm, b가 2cm, c가 2cm일 때 d 길이는? 답: 4cm)\n"
    "삼각형의 무게중심을 설명하거나, 삼각형의 무게중심 관련 문제를 낼 때, 다음 링크를 사용할 수 있습니다: https://i.imgur.com/zlwLnJc.png \n 이미지에는 삼각형 ABC에서 BC의 중점 D, AC의 중점 E, AB의 중점 F가, 무게중심 G가 표시되어 있습니다. AG:GD=BG:GE=CG:GF=2:1입니다. 이 이미지를 활용한 문항도 제시할 수 있습니다. (예: AG가 10cm라면 GD의 길이는? 답: 5cm)\n"
)

MATH_06_PROMPT = (
    "당신은 수학의 Ⅵ. 확률 단원 학습 지원을 담당합니다. 아래 1~3을 고려해 학습을 지원하세요. \n"
    "1. 단원의 주요 키워드\n"
    "경우의 수, 사건\n"
    "사건 A 또는 사건 B가 일어나는 경우의 수\n"
    "사건 A와 사건 B가 동시에 일어나는 경우의 수\n"
    "확률, 확률의 정의, 확률의 기본성질\n"
    "어떤 사건이 반드시 일어날 확률과 일어나지 않을 확률\n"
    "사건 A 또는 사건 B가 일어날 확률\n"
    "사건 A와 사건 B가 동시에 일어날 확률\n"
    "2. 학습 지원 지침\n"
    "학생의 답이 틀렸는데 맞았다고 하는 경우가 있으니, 문제의 정답은 반드시 신중하게 검토하세요.\n"
    "수학 교과에서는 LaTex 작성 규칙(수식 앞뒤를 줄바꿈하고 @@@@@로 감싸는 것)을 지키는 것이 매우 중요합니다. LaTex 규칙을 절대적으로 지키세요.\n\n"
    "3. 사용 가능한 이미지 목록:\n"
    "이 단원에서는 사용 가능한 이미지가 없습니다. 이미지를 사용하지 마세요. \n"
)

def summarize_chunks(chunks, unit_prompt, max_chunks=3):
    summaries = []
    for chunk in chunks[:max_chunks]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": COMMON_PROMPT},
                {"role": "system", "content": unit_prompt},
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

    replacements = {
        r"\\perp": "⟂",
        r"\\angle": "∠",
        r"\\parallel": "∥",
        r"\\infty": "∞",
        r"\\approx": "≈",
        r"\\sim": "∼",
        r"\\backsim": "∽",
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

    unit_prompts = {
        #수학
        "Ⅰ. 수와 식의 계산": MATH_01_PROMPT,
        "Ⅳ. 도형의 성질": MATH_04_PROMPT,
        "Ⅴ. 도형의 닮음": MATH_05_PROMPT,
        "Ⅵ. 확률": MATH_06_PROMPT,

        # 과학
        "Ⅳ. 식물과 에너지": SCIENCE_04_PROMPT,
        "Ⅴ. 동물과 에너지": SCIENCE_05_PROMPT,
        "Ⅵ. 물질의 특성": SCIENCE_06_PROMPT,
        "Ⅶ. 수권과 해수의 순환": SCIENCE_07_PROMPT,
    }
    selected_unit_prompt = unit_prompts.get(topic, "")

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
                        st.write(f"**학습 도우미:** {txt.strip()}")

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
                {"role": "system", "content": selected_unit_prompt},
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