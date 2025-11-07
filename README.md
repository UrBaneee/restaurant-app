# 🍽️ Restaurant Name Generator (LangChain + Streamlit)

An interactive **LLM-powered restaurant name and menu generator**, built with  
**LangChain (v1 / Runnable API)** and **Streamlit**.  

This project is adapted and improved from the YouTube tutorial  
👉 _[LangChain Crash Course for Beginners | LangChain Tutorial](https://www.youtube.com/watch?v=nAmC7SoVLd8)_,  
with updated code for the latest LangChain 1.x syntax and OpenAI API integration.

---

## ✨ Features

- 🧠 **AI-generated restaurant names & menus** based on your selected cuisine  
- ⚙️ Built on the **latest LangChain Runnable interface** (`prompt | llm | parser`)  
- 🎛️ Adjustable **temperature** to control creativity  
- 🧩 Clean **Streamlit UI** with instant generation  
- 🔑 Optional in-app API key input for easy testing  

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **LangChain v1** | LLM orchestration (prompt + model + output parser) |
| **OpenAI GPT-4o-mini** | Language model for generation |
| **Streamlit** | Front-end interface |
| **Python 3.11+** | Core runtime environment |

---

## 🚀 Demo Overview

| UI Section | Function |
|-------------|-----------|
| **Cuisine Selector** | Choose from Chinese, Indian, Italian, etc. |
| **Generate Button** | Calls LLM pipeline to produce restaurant name & menu |
| **Preview Panel** | Displays AI-generated content in Markdown |


<p align="center">
  <img width="1237" height="786" alt="IMG_5223" src="https://github.com/user-attachments/assets/9f32537d-80ca-4252-86bb-fa494c176e50" />
</p>

---

## 🧑‍💻 How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/UrBaneee/restaurant_app.git
   cd restaurant_app

2. **Create and activate a virtual environment**
   ```bash
   conda create -n lc1 python=3.11 -y
   conda activate lc1

3. **Install dependencies**
   ```bash
   pip install -U langchain langchain-core langchain-community langchain-openai streamlit

4. **Add your OpenAI API key**
   Option A: Put it in a .env file:
   ```bash
   OPENAI_API_KEY=sk-xxxxxx

   OR: Enter it in the Streamlit sidebar at runtime

4. **Run the app**
   ```bash
   streamlit run app.py
