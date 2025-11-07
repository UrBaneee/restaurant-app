import os
import re
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# -------------------- Env & Basic --------------------
load_dotenv()  # local dev: load .env
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip() or None

st.set_page_config(
    page_title="Restaurant Name Generator",
    page_icon="🍽️",
    layout="wide",
)

HERO_TITLE = "Restaurant Name Generator"
HERO_SUB = "Pick a cuisine on the left. Get a brandable name and a ready-to-use menu."

# -------------------- UI – Header --------------------
with st.container():
    col_logo, col_title = st.columns([1, 6])
    with col_title:
        st.markdown(
            f"<h1 style='margin-bottom:0'>{HERO_TITLE}</h1>"
            f"<p style='color:#6b7280;margin-top:0'>{HERO_SUB}</p>",
            unsafe_allow_html=True,
        )

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Pick a Cuisine")
    CUISINES = [
        "Indian", "Italian", "Mexican", "Arabic", "American", "Chinese", "Japanese",
        "Thai", "Korean", "French", "Spanish", "Greek", "Turkish", "Vietnamese",
    ]
    cuisine = st.selectbox("Cuisine", CUISINES, index=2)

    st.divider()
    st.caption("Generation Controls")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.7, 0.1)

    st.divider()
    st.caption("Options")
    bullet_style = st.selectbox(
        "Menu style", ["Bullets", "Numbered", "Plain lines"])
    export_fmt = st.selectbox("Export format", ["Text", "Markdown"])

    st.divider()
    st.caption("API Key")
    # 支持：Streamlit Cloud 的 Secrets / 环境变量 / 手动输入
    key_input = st.text_input(
        "OPENAI_API_KEY",
        type="password",
        placeholder="You may leave this empty if env var is set",
    )
    effective_key = key_input.strip() or OPENAI_API_KEY

# -------------------- LangChain – Chains --------------------


@st.cache_resource(show_spinner=False)
def get_llm(temp: float, api_key: str) -> ChatOpenAI:
    # 轻量、便于复用
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    return ChatOpenAI(
        temperature=temp,
        openai_api_key=api_key,
        model="gpt-4o-mini",
    )


parser = StrOutputParser()

name_prompt = PromptTemplate(
    input_variables=["cuisine"],
    template=(
        "You are a brand consultant. Give a short, catchy, brandable restaurant name "
        "for {cuisine} cuisine. Return ONLY the name, no quotes or extra text."
    ),
)
menu_prompt = PromptTemplate(
    input_variables=["cuisine", "restaurant_name"],
    template=(
        "List 6 popular menu items for a {cuisine} restaurant named {restaurant_name}. "
        "Return one item per line, no numbering."
    ),
)

# --- New prompts ---
drinks_prompt = PromptTemplate(
    input_variables=["cuisine", "restaurant_name"],
    template=(
        "List 6 popular drink items for a {cuisine} restaurant named {restaurant_name}. "
        "Include at least 2 non-alcoholic options. "
        "Return one item per line, no numbering."
    ),
)

slogan_prompt = PromptTemplate(
    input_variables=["restaurant_name", "cuisine"],
    template=(
        "You are a brand copywriter. Create a short, catchy slogan (max 6 words) "
        "for a {cuisine} restaurant named {restaurant_name}. Return ONLY the slogan."
    ),
)

description_prompt = PromptTemplate(
    input_variables=["restaurant_name", "cuisine"],
    template=(
        "Write a warm, vivid, 2–3 sentence description for a {cuisine} restaurant "
        "named {restaurant_name}. Avoid clichés. No markdown or extra headings."
    ),
)


def build_chain(llm: ChatOpenAI):
    # existing sub-chains
    name_chain = name_prompt | llm | parser
    menu_chain = menu_prompt | llm | parser

    # new sub-chains
    drinks_chain = drinks_prompt | llm | parser
    slogan_chain = slogan_prompt | llm | parser
    description_chain = description_prompt | llm | parser

    def build_menu(inputs):
        prompt_inputs = {
            "cuisine": inputs.get("cuisine", ""),
            "restaurant_name": inputs.get("restaurant_name", ""),
        }
        return menu_chain.invoke(prompt_inputs)

    def build_drinks(inputs):
        prompt_inputs = {
            "cuisine": inputs.get("cuisine", ""),
            "restaurant_name": inputs.get("restaurant_name", ""),
        }
        return drinks_chain.invoke(prompt_inputs)

    full_chain = (
        RunnablePassthrough
        .assign(restaurant_name=name_chain)
        .assign(menu_items=RunnableLambda(build_menu))
        .assign(drink_items=RunnableLambda(build_drinks))
        .assign(slogan=slogan_chain)
        .assign(description=description_chain)
    )
    return full_chain

# -------------------- Helpers --------------------


LINE_CLEAN_RE = re.compile(
    r"^\s*(?:[\u2022\-\*\u2013\u2014]+|\d+[.)\-\u2022]*)?\s*"
)
QUOTE_CLEAN_CHARS = "\"'“”‘’`"
MAX_MENU_ITEMS = 6


def normalize_lines(text: str | None) -> List[str]:
    if not text:
        return []

    cleaned: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        normalized = LINE_CLEAN_RE.sub("", stripped, count=1).strip()
        if normalized:
            cleaned.append(normalized)
    deduped = list(dict.fromkeys(cleaned))
    # 保持简洁：最多展示指定数量的菜单
    return deduped[:MAX_MENU_ITEMS]


def clean_restaurant_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = cleaned.strip(QUOTE_CLEAN_CHARS)
    return cleaned.strip()


def to_display_list(items: List[str], style: str) -> str:
    if style == "Bullets":
        return "\n".join([f"- {x}" for x in items])
    elif style == "Numbered":
        return "\n".join([f"{i+1}. {x}" for i, x in enumerate(items)])
    return "\n".join(items)


def to_export_v2(
    name: str,
    items: List[str],
    fmt: str,
    slogan: str = "",
    description: str = "",
    drinks: List[str] | None = None,
) -> str:
    drinks = drinks or []
    if fmt == "Markdown":
        parts = []
        parts.append(f"## {name}")
        if slogan:
            parts.append(f"*{slogan}*")
        if description:
            parts.append(description)

        parts.append("\n### Menu Items")
        parts.append("\n".join([f"- {x}" for x in items]))

        parts.append("\n### Drinks")
        if drinks:
            parts.append("\n".join([f"- {x}" for x in drinks]))
        else:
            parts.append("_(none)_")
        return "\n\n".join(parts).strip()
    else:
        parts = [name]
        if slogan:
            parts.append(slogan)
        if description:
            parts.append(description)
        parts.append("\nMenu Items")
        parts.extend(items)
        parts.append("\nDrinks")
        parts.extend(drinks if drinks else ["(none)"])
        return "\n".join(parts).strip()


# -------------------- Main Panel --------------------
col_left, col_right = st.columns([2.2, 1])

with col_left:
    st.subheader("Generate")
    run_btn = st.button("Generate Name & Menu", use_container_width=True)

    # 体验：按下按钮才运行，避免每次交互都打 API
    if run_btn:
        if not effective_key:
            st.error(
                "Please provide OPENAI_API_KEY (environment/secrets or sidebar input).")
        else:
            llm = get_llm(temperature, effective_key)
            chain = build_chain(llm)
            with st.spinner("Cooking..."):
                try:
                    res = chain.invoke({"cuisine": cuisine})
                    rest_name = clean_restaurant_name(res.get("restaurant_name"))
                    items = normalize_lines(res.get("menu_items"))

                    # NEW: drinks, slogan, description
                    drinks = normalize_lines(res.get("drink_items"))
                    slogan = (res.get("slogan") or "").strip()
                    description = (res.get("description") or "").strip()

                    if not rest_name:
                        raise ValueError("No restaurant name was returned by the model.")
                    if not items:
                        raise ValueError("No menu items were returned by the model.")
                    
                    st.success("Generated!")
                    # Title + Slogan
                    st.markdown(f"## {rest_name}")
                    if slogan:
                         st.markdown(f"*{slogan}*")
                    # Description
                    if description:
                         st.markdown(description)
                    # Food Menu
                    st.markdown("### Menu Items")
                    st.markdown(to_display_list(items, bullet_style))
                    # Drinks
                    st.markdown("### Drinks")
                    if drinks:
                         st.markdown(to_display_list(drinks, bullet_style))
                    else:
                        st.caption("No drinks returned. Try again or adjust temperature.")

                    # 复制到剪贴板（以文本框提供）
                    st.caption("Copy / Export")
                    content = to_export_v2(
                        rest_name, items, export_fmt,
                        slogan=slogan, description=description, drinks=drinks
                    )
                    st.text_area("Output", value=content, height=180)

                    # 下载按钮
                    st.download_button(
                        "Download",
                        data=content.encode("utf-8"),
                        file_name=f"{rest_name.replace(' ','_')}.{'md' if export_fmt=='Markdown' else 'txt'}",
                        mime="text/markdown" if export_fmt == "Markdown" else "text/plain",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.exception(e)
    else:
        st.info("Choose a cuisine and click **Generate Name & Menu**.")

with col_right:
    st.subheader("Preview")
    st.image(
        "https://images.unsplash.com/photo-1526318472351-c75fcf070305?q=80&w=1200&auto=format&fit=crop",
        caption="Fresh from the kitchen",
        use_container_width=True,
    )

st.caption("© 2025 Skye Yin – Built with Streamlit + LangChain (LCEL)")
