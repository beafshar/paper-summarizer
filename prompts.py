# prompts.py
BASE_SYSTEM = """You are a concise, reliable research paper summarizer."""

SHORT_SUMMARY = """Summarize the paper in 6-8 bullet points for a technical audience.
Highlight: problem, method, key results, limitations, and future work."""

STRUCTURED_SUMMARY = """Produce a structured summary with these sections:
1) Problem & Motivation
2) Method (2-4 sentences)
3) Data/Setup (if applicable)
4) Results (with concrete numbers if present)
5) Limitations
6) Takeaways for practitioners"""

ABSTRACT_FOCUS = """Summarize the paper focusing ONLY on the abstract + conclusion.
Keep it under 150 words with 3 bullets of takeaways."""
