import re
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


stop_words = set(stopwords.words('english'))

skill_keywords = {
        "data science": ["python", "machine learning", "data", "analytics", "sql", "ai", "deep learning"],
        "web development": ["html", "css", "javascript", "react", "node", "django", "flask", "typescript"],
        "marketing": ["marketing", "seo", "branding", "advertising", "social media", "content", "campaign"],
        "sales": ["sales", "negotiation", "crm", "lead generation", "b2b", "cold calling", "closing deals"],
        "graphic design": ["photoshop", "illustrator", "figma", "ui", "ux", "adobe", "canva", "indesign"],
        "medicine": ["doctor", "nurse", "clinical", "medical", "surgery", "diagnosis", "patient", "treatment"],
        "finance": ["accounting", "finance", "budgeting", "forecasting", "excel", "auditing", "tax"],
        "education": ["teacher", "curriculum", "lesson", "training", "instruction", "student", "classroom"],
        "engineering": ["cad", "mechanical", "electrical", "civil", "autocad", "manufacturing", "maintenance"],
        "law": ["law", "legal", "contract", "compliance", "litigation", "attorney", "regulation"],
        "logistics": ["supply chain", "logistics", "inventory", "warehouse", "procurement", "shipment", "transport"],
        "hospitality": ["hotel", "restaurant", "chef", "guest service", "reservation", "hospitality", "barista"],
        "human resources": ["recruitment", "hiring", "onboarding", "hr", "talent", "interview", "employee relations"],
        "project management": ["project", "planning", "scheduling", "deadline", "budget", "stakeholder", "agile"],
        "real estate": ["property", "real estate", "broker", "tenant", "lease", "valuation", "mortgage"],
        "customer service": ["customer", "support", "complaint", "communication", "helpdesk", "service", "call center"],
        "architecture": ["architecture", "design", "blueprint", "urban", "construction", "autocad", "3d modeling"],
        "agriculture": ["farming", "agriculture", "crop", "harvest", "livestock", "irrigation"],
        "media": ["journalism", "writing", "editor", "broadcast", "reporting", "publishing", "content"],
        "information technology": ["it", "network", "cybersecurity", "system admin", "database", "troubleshooting"]
    }

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_skills(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Expanded skill keywords across many fields
   
    detected_skills = []
    for field, keywords in skill_keywords.items():
        for kw in keywords:
            if kw in tokens:
                detected_skills.append(kw)

    return sorted(set(detected_skills))


def detect_job_field(text):
    text = text.lower()
    field_weights = {
        "Data Science": ["python", "machine learning", "data", "ai", "analytics"],
        "Web Development": ["html", "css", "javascript", "react", "flask"],
        "Marketing": ["seo", "marketing", "branding", "advertising", "campaign"],
        "Sales": ["sales", "customer", "crm", "b2b"],
        "Graphic Design": ["photoshop", "illustrator", "figma", "adobe"],
        "Medicine": ["doctor", "clinical", "medical", "surgery"],
        "Finance": ["accounting", "budget", "finance", "tax"],
        "Education": ["teacher", "student", "lesson", "school"],
        "Engineering": ["mechanical", "electrical", "civil", "manufacturing"],
        "Law": ["law", "legal", "contract", "litigation"],
        "Logistics": ["logistics", "supply chain", "shipment"],
        "Hospitality": ["hotel", "guest", "chef", "service"],
        "Human Resources": ["recruitment", "hr", "talent"],
        "Project Management": ["project", "deadline", "planning", "agile"],
        "Real Estate": ["property", "real estate", "lease"],
        "Customer Service": ["customer", "support", "helpdesk"],
        "Architecture": ["architecture", "design", "blueprint"],
        "Agriculture": ["farming", "crop", "harvest"],
        "Media": ["journalism", "writing", "editor"],
        "Information Technology": ["network", "system", "security", "database"]
    }

    scores = {}
    for field, keywords in field_weights.items():
        score = sum(text.count(k) for k in keywords)
        if score > 0:
            scores[field] = score

    if not scores:
        return "General / Other"

    return max(scores, key=scores.get)
