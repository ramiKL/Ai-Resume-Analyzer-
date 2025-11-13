import fitz  # PyMuPDF
import spacy
import re
import nltk
from nltk.corpus import stopwords
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from datetime import datetime
import numpy as np

os.makedirs("Uploaded_Resumes", exist_ok=True)

nltk.download('stopwords', quiet=True)
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

# Expanded skill bank grouped by job fields (add or refine terms as needed)
SKILL_MAP = {
    "Technology": [
        "python", "java", "c++", "javascript", "html", "css", "react", "angular",
        "node.js", "node", "django", "flask", "sql", "postgresql", "mysql",
        "aws", "azure", "gcp", "docker", "kubernetes", "pandas", "numpy",
        "tensorflow", "pytorch", "machine learning", "data science"
    ],
    "Healthcare": [
        "patient care", "medical", "nursing", "rn", "registered nurse", "emr",
        "electronic medical record", "phlebotomy", "clinical", "icu", "pediatrics",
        "radiology", "lab", "pharmacy", "healthcare", "medical assistant"
    ],
    "Finance & Accounting": [
        "accounting", "bookkeeping", "auditing", "tax", "financial analysis",
        "quickbooks", "accounts payable", "accounts receivable", "excel", "forecasting",
        "financial reporting", "investment", "cpa"
    ],
    "Marketing & Sales": [
        "seo", "sem", "content marketing", "social media", "google analytics",
        "campaign", "lead generation", "sales", "crm", "hubspot", "salesforce",
        "copywriting", "branding", "paid ads", "email marketing"
    ],
    "Design & Creative": [
        "photoshop", "illustrator", "indesign", "ux", "ui", "graphic design",
        "prototyping", "sketch", "figma", "animation", "video editing", "after effects"
    ],
    "Education & Training": [
        "teaching", "curriculum", "lesson plan", "elearning", "instructional design",
        "tutor", "classroom", "certification", "k-12", "higher education"
    ],
    "Operations & Admin": [
        "project management", "pm", "administrative", "scheduling", "logistics",
        "supply chain", "office management", "ms office", "customer service"
    ],
    "Hospitality & Service": [
        "hospitality", "guest services", "food service", "bartender", "front desk",
        "hotel", "catering", "events", "housekeeping"
    ],
    "Construction & Manufacturing": [
        "welding", "osha", "blueprint", "machinery", "cnc", "fabrication",
        "quality control", "assembly", "plumbing", "electrical"
    ],
    "Legal & Compliance": [
        "compliance", "legal research", "paralegal", "contract", "litigation",
        "case management", "regulatory", "nda"
    ]
}

# simple helper to find whole-word matches (case-insensitive)
def _find_keywords(text, keywords):
    found = set()
    for kw in keywords:
        # escape regex special characters in kw and match word boundaries
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, text.lower()):
            found.add(kw)
    return found

def extract_skills_and_fields(text):
    text = text.lower()
    # optional: remove some punctuation to help matching multi-word phrases
    text = re.sub(r"[\r\n\t]+", " ", text)
    field_hits = {}
    skills_found = set()

    for field, keywords in SKILL_MAP.items():
        matches = _find_keywords(text, keywords)
        if matches:
            field_hits[field] = len(matches)
            skills_found.update(matches)

    # If nothing found by exact keywords, try to extract nouns/phrases as backup
    if not skills_found:
        doc = nlp(text)
        for ent in doc.noun_chunks:
            token = ent.text.strip().lower()
            if len(token) > 2 and token not in stopwords.words('english'):
                skills_found.add(token)

    # sort fields by hits
    sorted_fields = sorted(field_hits.items(), key=lambda x: x[1], reverse=True)
    detected_fields = [f for f, _ in sorted_fields]
    return list(skills_found), detected_fields, field_hits

# Basic mapping for recommended job titles and courses per field
RECOMMENDED_JOBS = {
    "Technology": ["Software Engineer", "Data Analyst", "DevOps Engineer"],
    "Healthcare": ["Registered Nurse", "Clinical Assistant", "Healthcare Administrator"],
    "Finance & Accounting": ["Accountant", "Financial Analyst", "Bookkeeper"],
    "Marketing & Sales": ["Digital Marketer", "Sales Representative", "Content Strategist"],
    "Design & Creative": ["Graphic Designer", "UX/UI Designer", "Video Editor"],
    "Education & Training": ["Teacher", "Instructional Designer", "Tutor"],
    "Operations & Admin": ["Project Coordinator", "Operations Manager", "Office Administrator"],
    "Hospitality & Service": ["Hotel Manager", "Event Coordinator", "Server"],
    "Construction & Manufacturing": ["Welder", "Site Supervisor", "CNC Operator"],
    "Legal & Compliance": ["Paralegal", "Compliance Officer", "Legal Assistant"]
}

RECOMMENDED_COURSES = {
    "Technology": ["Intro to Programming (Python)", "Data Analysis with Pandas", "Cloud Fundamentals"],
    "Healthcare": ["Basic Life Support (BLS)", "Medical Terminology", "EMR Training"],
    "Finance & Accounting": ["Accounting Basics", "Excel for Finance", "Financial Modeling"],
    "Marketing & Sales": ["SEO Fundamentals", "Social Media Marketing", "Sales Techniques"],
    "Design & Creative": ["Adobe Photoshop Essentials", "UX Design Fundamentals", "Figma for Beginners"],
    "Education & Training": ["Classroom Management", "Instructional Design Basics", "TESOL Prep"],
    "Operations & Admin": ["Project Management Basics", "Supply Chain Fundamentals", "Office 365 Essentials"],
    "Hospitality & Service": ["Customer Service Excellence", "Event Management", "Food Safety"],
    "Construction & Manufacturing": ["OSHA Safety", "Blueprint Reading", "CNC Basics"],
    "Legal & Compliance": ["Legal Research", "Contract Law Basics", "Compliance Essentials"]
}

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get skill, job-field and course recommendations across many industries.")

# Inject project-wide CSS to better match the site's blue/white theme (uses Inter font)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Base app background and font */
    .stApp, .main, .block-container {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial !important;
        color: #063b73 !important;
        background: linear-gradient(135deg, #e8f2ff 0%, #f0e7ff 50%, #ffffff 100%) !important;
    }

    /* Make primary buttons look like project primary style */
    .stButton>button, .css-18ni7ap.egzxvld2 {
        border-radius: 10px !important;
        padding: 8px 14px !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg,#0a66c2,#155bd6) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 20px rgba(10,100,200,0.08) !important;
    }

    /* Headings and accents */
    h1, h2, h3, .stHeading {
        color: #0a66c2 !important;
    }

    /* Card-like containers */
    .stContainer, .stBlock, .css-1d391kg {
        background: #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 18px rgba(8,30,53,0.04) !important;
        padding: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Main content
if True:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])

    if uploaded_file:
        dest_path = f"Uploaded_Resumes/{uploaded_file.name}"
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_pdf(dest_path)
        skills, fields, field_hits = extract_skills_and_fields(text)

        # compute skill counts (frequency of keyword occurrences) for visualization
        skill_counts = {}
        low_text = text.lower()
        for field, keywords in SKILL_MAP.items():
            for kw in keywords:
                # count occurrences of keyword (simple whole-word match)
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                c = len(re.findall(pattern, low_text))
                if c:
                    skill_counts[kw] = skill_counts.get(kw, 0) + c

        # UI: two-column layout for modern presentation
        left, right = st.columns([2, 1])

        with left:
            if skills:
                st.subheader("Detected Skills / Keywords")
                st.write(", ".join(sorted(skills)))
            else:
                st.info("No direct keyword matches found — showing extracted noun phrases as hints.")
                st.write(", ".join(sorted(skills[:20])))

            # Field ranking and charts
            if fields:
                st.subheader("Likely Job Fields (ranked)")
                st.write(", ".join(fields))

                # Prepare DataFrame for plotting
                df = pd.DataFrame({"Field": list(field_hits.keys()), "Hits": list(field_hits.values())})
                df = df.sort_values("Hits", ascending=True)

                # Horizontal bar (modern style)
                fig_bar = px.bar(
                    df,
                    x="Hits",
                    y="Field",
                    orientation="h",
                    color="Hits",
                    color_continuous_scale="Blues",
                    template="plotly_white",
                    text="Hits",
                    height=320
                )
                fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
                fig_bar.update_traces(marker=dict(line=dict(width=0)), textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)

                # Treemap for quick overview across fields
                fig_tree = px.treemap(
                    df[df["Hits"] > 0],
                    path=["Field"],
                    values="Hits",
                    color="Hits",
                    color_continuous_scale="Blues",
                    template="plotly_white",
                    height=360
                )
                fig_tree.update_layout(margin=dict(l=5, r=5, t=30, b=5))
                st.plotly_chart(fig_tree, use_container_width=True)

            else:
                st.info("Could not confidently determine a job field from resume content.")

        with right:
            st.subheader("Skill Cloud")
            if skill_counts:
                # Build a word-cloud like scatter (no extra package required)
                sk_df = pd.DataFrame(list(skill_counts.items()), columns=["skill", "count"]).sort_values("count", ascending=False)
                # pick top N to avoid clutter
                sk_df = sk_df.head(30)
                # normalize font sizes
                min_size, max_size = 12, 48
                counts = sk_df["count"].astype(float)
                if counts.max() == counts.min():
                    sizes = np.full(len(counts), (min_size + max_size) / 2)
                else:
                    sizes = min_size + (counts - counts.min()) / (counts.max() - counts.min()) * (max_size - min_size)

                # random positions for a scattered "cloud"
                np.random.seed(42)
                x = np.random.rand(len(sk_df))
                y = np.random.rand(len(sk_df))

                fig_cloud = px.scatter(
                    sk_df,
                    x=x,
                    y=y,
                    text=sk_df["skill"],
                    size=[s/10 for s in sizes],  # adjust marker sizes
                    color=sk_df["count"],
                    color_continuous_scale="thermal",
                    template="plotly_white",
                    height=420
                )
                fig_cloud.update_traces(textposition="middle center", marker=dict(opacity=0))
                fig_cloud.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=10, b=0))
                # set text sizes manually
                for i, t in enumerate(fig_cloud.data[0]['text']):
                    fig_cloud.data[0]['textfont'] = dict(size=list(sizes))
                st.plotly_chart(fig_cloud, use_container_width=True)
            else:
                st.info("No keyword frequencies available for skill cloud.")

            # Recommended jobs & courses for top field
            if fields:
                top_field = fields[0]
                st.subheader(f"Suggested job titles")
                st.write(", ".join(RECOMMENDED_JOBS.get(top_field, [])[:5]))

                st.subheader(f"Recommended courses")
                st.write(", ".join(RECOMMENDED_COURSES.get(top_field, [])[:5]))

        # Fallback message if nothing to chart
        if not field_hits:
            st.info("No field keyword matches to chart.")


