import fitz  # PyMuPDF
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text


def extract_skills(text):
    """Return a list of recognized skills across multiple job domains"""
    text = text.lower()

    domain_skills = {
        "Data Science": [
            "python", "machine learning", "deep learning", "statistics",
            "sql", "data visualization", "power bi", "tableau", "pandas", "numpy"
        ],
        "Web Development": [
            "html", "css", "javascript", "react", "vue", "node", "django", "flask", "php"
        ],
        "Marketing": [
            "seo", "social media", "email marketing", "copywriting", "google ads",
            "content creation", "digital marketing", "branding", "market research"
        ],
        "Sales": [
            "negotiation", "customer relationship", "crm", "salesforce",
            "cold calling", "lead generation", "closing deals", "b2b sales", "b2c sales"
        ],
        "Graphic Design": [
            "photoshop", "illustrator", "indesign", "figma", "canva", "ux", "ui", "branding", "motion graphics"
        ],
        "Healthcare": [
            "patient care", "medical records", "diagnosis", "treatment planning",
            "nursing", "phlebotomy", "first aid", "emergency care", "surgery", "clinical research"
        ],
        "Education": [
            "teaching", "curriculum development", "lesson planning", "classroom management",
            "assessment", "tutoring", "student engagement"
        ],
        "Finance": [
            "budgeting", "forecasting", "accounting", "financial analysis",
            "bookkeeping", "excel", "taxation", "auditing", "investment"
        ],
        "Human Resources": [
            "recruitment", "onboarding", "performance management", "training",
            "employee engagement", "hr policies", "talent acquisition", "payroll"
        ],
        "Customer Service": [
            "communication", "problem solving", "complaint resolution",
            "support tickets", "empathy", "customer satisfaction"
        ]
    }

    found_skills = []
    for category, keywords in domain_skills.items():
        for kw in keywords:
            if kw in text:
                found_skills.append(kw)

    return sorted(list(set(found_skills)))

def detect_job_field(text):
    """Return the job category most matching the resume content"""
    text = text.lower()

    domain_skills = {
        "Data Science": [
            "python", "machine learning", "deep learning", "statistics",
            "sql", "data visualization", "power bi", "tableau", "pandas", "numpy"
        ],
        "Web Development": [
            "html", "css", "javascript", "react", "vue", "node", "django", "flask", "php"
        ],
        "Marketing": [
            "seo", "social media", "email marketing", "copywriting", "google ads",
            "content creation", "digital marketing", "branding", "market research"
        ],
        "Sales": [
            "negotiation", "customer relationship", "crm", "salesforce",
            "cold calling", "lead generation", "closing deals", "b2b sales", "b2c sales"
        ],
        "Graphic Design": [
            "photoshop", "illustrator", "indesign", "figma", "canva", "ux", "ui", "branding", "motion graphics"
        ],
        "Healthcare": [
            "patient care", "medical records", "diagnosis", "treatment planning",
            "nursing", "phlebotomy", "first aid", "emergency care", "surgery", "clinical research"
        ],
        "Education": [
            "teaching", "curriculum development", "lesson planning", "classroom management",
            "assessment", "tutoring", "student engagement"
        ],
        "Finance": [
            "budgeting", "forecasting", "accounting", "financial analysis",
            "bookkeeping", "excel", "taxation", "auditing", "investment"
        ],
        "Human Resources": [
            "recruitment", "onboarding", "performance management", "training",
            "employee engagement", "hr policies", "talent acquisition", "payroll"
        ],
        "Customer Service": [
            "communication", "problem solving", "complaint resolution",
            "support tickets", "empathy", "customer satisfaction"
        ]
    }

    domain_hits = {}
    for category, keywords in domain_skills.items():
        hits = sum(kw in text for kw in keywords)
        if hits:
            domain_hits[category] = hits

    if not domain_hits:
        return "General / Undefined"

    return max(domain_hits, key=domain_hits.get)
