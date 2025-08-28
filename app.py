import os
import io
import time
import tempfile
from datetime import datetime

import streamlit as st
import mlflow
from mlflow.models.signature import infer_signature

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLMs
from langchain_openai import ChatOpenAI  

from prompts import BASE_SYSTEM, SHORT_SUMMARY, STRUCTURED_SUMMARY, ABSTRACT_FOCUS



# ---------- Helpers ----------
def extract_text_from_pdf(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages), len(pages)

def chunk_text(text: str, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def count_tokens_rough(text: str) -> int:
    return max(1, int(len(text) / 4))

def summarize_chunks_llm(chunks, model_name: str, temperature: float, prompt_variant: str):
    if prompt_variant == "Short bullets":
        user_prompt = SHORT_SUMMARY
    elif prompt_variant == "Structured":
        user_prompt = STRUCTURED_SUMMARY
    else:
        user_prompt = ABSTRACT_FOCUS

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Summarize each chunk briefly
    partial_summaries = []
    for i, ch in enumerate(chunks):
        resp = llm.invoke([
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": f"{user_prompt}\n\nCHUNK {i+1}:\n{ch}"}
        ])
        partial_summaries.append(resp.content.strip())

    # Final merge step
    merged_prompt = f"""Combine the following chunk summaries into a single cohesive summary.
Avoid repetition; keep it self-contained and faithful.

CHUNK SUMMARIES:
{chr(10).join(f"- {s}" for s in partial_summaries)}
"""
    final = llm.invoke([
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": merged_prompt}
    ]).content.strip()

    return partial_summaries, final, user_prompt

def compute_metrics(src_text: str, final_summary: str):
    src_tokens = count_tokens_rough(src_text)
    sum_tokens = count_tokens_rough(final_summary)
    compression = round(sum_tokens / src_tokens, 4)
    return {
        "source_tokens_approx": src_tokens,
        "summary_tokens_approx": sum_tokens,
        "compression_ratio": compression
    }

def save_artifacts(run_dir, partial_summaries, final_summary):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "partial_summaries.txt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(partial_summaries):
            f.write(f"### Chunk {i+1}\n{s}\n\n")
    with open(os.path.join(run_dir, "final_summary.txt"), "w", encoding="utf-8") as f:
        f.write(final_summary)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Paper Summarizer (LangChain + MLflow)", layout="wide")
st.title("📄 Research Paper Summarizer")

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    prompt_variant = st.selectbox("Prompt", ["Short bullets", "Structured", "Abstract/Conclusion focus"])
    chunk_size = st.slider("Chunk size", 500, 4000, 2000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 1000, 200, 50)
    experiment_name = st.text_input("MLflow experiment", value="paper-summarizer")
    tracking_uri = st.text_input("MLflow tracking URI", value="file:./mlruns")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
run_button = st.button("Summarize & Log to MLflow", type="primary", disabled=not uploaded_pdf)

if run_button and uploaded_pdf:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with st.status("Processing…", expanded=False) as status:
        start = time.time()
        try:
            text, page_count = extract_text_from_pdf(uploaded_pdf)
            status.update(label=f"Extracted text from {page_count} page(s).")

            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            status.update(label=f"Split into {len(chunks)} chunk(s). Summarizing…")

            partial_summaries, final_summary, used_prompt = summarize_chunks_llm(
                chunks, model_name, temperature, prompt_variant
            )
            metrics = compute_metrics(text, final_summary)
            latency = round(time.time() - start, 2)
            metrics["end_to_end_seconds"] = latency

            # MLflow logging
            with mlflow.start_run(run_name=f"{os.path.splitext(uploaded_pdf.name)[0]}-{datetime.now().strftime('%H:%M:%S')}"):
                mlflow.set_tags({
                    "task": "paper_summarization",
                    "model_name": model_name,
                    "prompt_variant": prompt_variant,
                    "pages": page_count
                })
                mlflow.log_params({
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "temperature": temperature
                })
                mlflow.log_metrics(metrics)

                # Save artifacts
                with tempfile.TemporaryDirectory() as tmpdir:
                    save_artifacts(tmpdir, partial_summaries, final_summary)
                    mlflow.log_artifacts(tmpdir, artifact_path="summaries")


                signature = infer_signature(
                    model_input={"text_len": len(text)},
                    model_output={"summary_len": len(final_summary)}
                )
                mlflow.log_dict(signature.to_dict(), "signature.json")

            status.update(label="Done! Logged to MLflow.", state="complete")

            st.success("Summary created ✅  (compare runs in MLflow UI).")
            st.subheader("Final Summary")
            st.write(final_summary)

            with st.expander("Partial chunk summaries"):
                for i, s in enumerate(partial_summaries, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(s)

            # st.info("Tip: run `mlflow ui` and open http://127.0.0.1:5000 to compare runs.")

        except Exception as e:
            st.error(f"Error: {e}")
