import os
import io
import re
import sys
import json
import base64
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from e2b_code_interpreter import Sandbox

# LLMs
import openai
import google.generativeai as genai

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    return match.group(1) if match else ""


def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner("Executing code..."):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results


def chat_with_openai(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    system_prompt = f"""You're a Python data scientist and visualization expert. Dataset is at '{dataset_path}'. Always read it using this path. Respond with analysis and Python code to solve the user's question."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    response = openai.ChatCompletion.create(
        model=st.session_state.openai_model,
        messages=messages,
        api_key=st.session_state.openai_api_key
    )
    reply = response["choices"][0]["message"]["content"]
    python_code = match_code_blocks(reply)
    return (code_interpret(e2b_code_interpreter, python_code), reply) if python_code else (None, reply)


def chat_with_gemini(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    prompt = f"""You're a Python data scientist and visualization expert. Dataset is at '{dataset_path}'. Always read it using this path. Respond with analysis and Python code to solve the user's question.

User Query:
{user_message}"""

    genai.configure(api_key=st.session_state.gemini_api_key)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    reply = response.text
    python_code = match_code_blocks(reply)
    return (code_interpret(e2b_code_interpreter, python_code), reply) if python_code else (None, reply)


def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Upload failed: {error}")
        raise error


def main():
    st.set_page_config(page_title="📊 AI Data Visualization Agent", layout="wide")
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask AI to analyze it!")

    # Sidebar for API Keys & Model Selection
    with st.sidebar:
        st.header("🔐 API & Model Config")

        provider = st.selectbox("Select Model Provider", ["OpenAI", "Gemini"])
        st.session_state.provider = provider

        if provider == "OpenAI":
            st.session_state.openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
            st.session_state.openai_model = st.selectbox("OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])
            st.sidebar.markdown("[Get OpenAI Key](https://platform.openai.com/account/api-keys)")
        else:
            st.session_state.gemini_api_key = st.text_input("🔑 Gemini API Key", type="password")
            st.sidebar.markdown("[Get Gemini Key](https://aistudio.google.com/app/apikey)")

        st.session_state.e2b_api_key = st.text_input("🧠 E2B API Key", type="password")
        st.sidebar.markdown("[Get E2B Key](https://e2b.dev/docs/legacy/getting-started/api-key)")

    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Data Preview")
        st.dataframe(df.head(10))

        query = st.text_area("🧠 Ask a question about your data", "What is the average cost by category?")

        if st.button("🚀 Analyze"):
            if not st.session_state.e2b_api_key or (
                provider == "OpenAI" and not st.session_state.openai_api_key
            ) or (provider == "Gemini" and not st.session_state.gemini_api_key):
                st.error("Please enter all required API keys.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)

                    if provider == "OpenAI":
                        results, reply = chat_with_openai(code_interpreter, query, dataset_path)
                    else:
                        results, reply = chat_with_gemini(code_interpreter, query, dataset_path)

                    st.markdown("### 🤖 AI Response")
                    st.write(reply)

                    if results:
                        for result in results:
                            if hasattr(result, "png") and result.png:
                                image = Image.open(BytesIO(base64.b64decode(result.png)))
                                st.image(image, caption="Visualization", use_container_width=True)
                            elif hasattr(result, "figure"):
                                st.pyplot(result.figure)
                            elif hasattr(result, "show"):
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)


if __name__ == "__main__":
    main()
