"""Streamlit UI for the AI Resume Analyzer.

This module provides a minimal Streamlit interface that mirrors the
style and organization used elsewhere in the project: imports at the top,
small helper functions, and a guarded `main()` entry point.
"""

import tempfile
from typing import List, Optional

import matplotlib.pyplot as plt
import streamlit as st

from analyzer import (
    detect_job_field,
    extract_skills,
    extract_text_from_pdf,
    skill_keywords,
)


def build_skill_chart(detected_skills: List[str]) -> Optional[plt.Figure]:
    """Build a horizontal bar chart summarizing matched skills per field.

    Returns a Matplotlib Figure or `None` if there are no matches.
    """
    # Count keyword matches per field
    field_counts = {
        field: sum(1 for kw in keywords if kw in detected_skills)
        for field, keywords in skill_keywords.items()
    }

    # Keep only fields with at least one match
    field_counts = {k: v for k, v in field_counts.items() if v > 0}

    if not field_counts:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(field_counts.keys()), list(field_counts.values()))
    ax.set_xlabel("Number of Matched Skills")
    ax.set_ylabel("Job Fields")
    ax.set_title("Resume Skill Match Overview")
    plt.tight_layout()
    return fig


def main() -> None:
    # Match the app's primary color to the project's stylesheet (styles.css)
    primary_color = "#0a66c2"  # --blue from project's `styles.css`
    st.markdown(
        f"""
        <style>
        h1 {{ color: {primary_color} !important; }}
        h2 {{ color: {primary_color} !important; }}
        .stButton>button {{
            background: linear-gradient(90deg, {primary_color}, #155bd6) !important;
            color: white !important;
            border: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI Resume Analyzer by SkillSync")
    st.write("Upload your resume PDF and get detected skills + predicted career field.")

    uploaded_file = st.file_uploader("Choose your resume (PDF format)", type=["pdf"])

    if uploaded_file is None:
        return

    # Persist uploaded PDF to a temporary file and hand its path to the analyzer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        uploaded_file_path = tmp_file.name

    text = extract_text_from_pdf(uploaded_file_path)
    skills = extract_skills(text)
    field = detect_job_field(text)

    st.subheader("Extracted Skills:")
    st.write(", ".join(skills) if skills else "(no skills detected)")

    st.subheader("Predicted Career Field:")
    st.success(field)

    fig = build_skill_chart(skills)
    if fig:
        st.subheader("📊 Skill Distribution")
        st.pyplot(fig)
    else:
        st.info("No specific field matches strong enough for charting.")


if __name__ == "__main__":
    main()
