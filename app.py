import spacy
import subprocess
import sys

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

# Load spaCy small English model (make sure to install: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer('all-MiniLM-L6-v2')

THRESHOLDS = {
    'sentence': {'low': 0.60, 'ideal': (0.70, 0.90), 'high': 0.95},
    'paragraph': {'low': 0.55, 'ideal': (0.65, 0.85), 'high': 0.90},
    'section': {'low': 0.45, 'ideal': (0.50, 0.80), 'high': 0.85}
}

def check_similarity(text_list, level):
    results = []
    embeddings = model.encode(text_list)
    pairs = list(zip(text_list, text_list[1:], embeddings, embeddings[1:]))
    for idx, (a, b, vec_a, vec_b) in enumerate(pairs):
        score = cosine_similarity([vec_a], [vec_b])[0][0]
        threshold = THRESHOLDS[level]
        if score > threshold['high']:
            comment = "Redundant. Rephrase or remove repetition."
        elif score < threshold['low']:
            comment = "Too disconnected. Add bridging sentence or transition."
        elif threshold['ideal'][0] <= score <= threshold['ideal'][1]:
            comment = "Good semantic flow."
        else:
            comment = "Slight deviation. Consider reworking the transition."
        results.append((a, b, score, comment))
    return results

def split_into_paragraphs(text):
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def parse_headings(headings_text):
    headings = []
    for line in headings_text.splitlines():
        line = line.strip()
        m = re.match(r'<H([1-3])>\s*(.*)', line, re.I)
        if m:
            level = int(m.group(1))
            title = m.group(2).strip()
            headings.append({'level': level, 'title': title})
    return headings

def assign_content_to_headings(headings, content, top_level=2):
    top_headings = [h for h in headings if h['level'] == top_level]
    paragraphs = split_into_paragraphs(content)
    assigned = []
    n = len(paragraphs)
    m = len(top_headings)
    if m == 0:
        # fallback: assign all paragraphs under a dummy heading
        assigned.append({'heading': {'level': top_level, 'title': 'Content'}, 'paragraphs': paragraphs})
        return assigned
    approx_chunk_size = max(1, n // m)
    idx = 0
    for i, heading in enumerate(top_headings):
        if i < m -1:
            chunk = paragraphs[idx: idx + approx_chunk_size]
            idx += approx_chunk_size
        else:
            chunk = paragraphs[idx:]
        assigned.append({'heading': heading, 'paragraphs': chunk})
    return assigned

def analyze_assigned_content(assigned):
    output_lines = []
    titles = [item['heading']['title'] for item in assigned]
    if len(titles) > 1:
        sims = check_similarity(titles, 'section')
        output_lines.append("=== Section-to-Section Flow ===")
        for a, b, score, comment in sims:
            output_lines.append(f"Between '{a}' and '{b}': {comment} (score: {score:.3f})")
        output_lines.append("")
    output_lines.append("=== Paragraph-to-Paragraph Flow ===")
    for item in assigned:
        heading = item['heading']
        paras = item['paragraphs']
        output_lines.append(f"Under {heading['title']}:")
        if len(paras) < 2:
            output_lines.append("  Not enough paragraphs for analysis.")
            continue
        sims = check_similarity(paras, 'paragraph')
        for i, (a, b, score, comment) in enumerate(sims):
            output_lines.append(f"  Paragraph {i+1} to {i+2}: {comment} (score: {score:.3f})")
        output_lines.append("")
    return '\n'.join(output_lines)

# --- New functions for context and suggestions ---

def extract_key_concepts(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        # Filter short chunks
        if len(chunk.text.strip()) > 2:
            keywords.add(chunk.text.lower())
    return list(keywords)

def paragraph_concept_coverage(paragraph, context_keywords):
    para_doc = nlp(paragraph.lower())
    para_text = para_doc.text
    missing_concepts = []
    for concept in context_keywords:
        # simple substring check, can be improved to fuzzy matching
        if concept not in para_text:
            missing_concepts.append(concept)
    return missing_concepts

def generate_specific_suggestions(paragraphs, context_text):
    context_keywords = extract_key_concepts(context_text)
    suggestions = []
    for idx, para in enumerate(paragraphs):
        missing = paragraph_concept_coverage(para, context_keywords)
        if missing:
            missing_preview = ', '.join(missing[:5])  # limit to 5 concepts
            suggestion = (
                f"Paragraph {idx+1} misses key concepts: {missing_preview}. "
                "Consider adding explanations, examples, or clarifications about these topics."
            )
        else:
            suggestion = f"Paragraph {idx+1} covers main concepts well."
        suggestions.append((idx, para, suggestion))
    return suggestions

# --- Streamlit UI ---

st.title("Semantic Flow Checker with Context-Aware Suggestions")

headings_input = st.text_area(
    "Paste Headings (with <H1>, <H2>, <H3> tags, one per line):",
    height=200,
    value='''<H1> What is an IT Service Provider? Types, Importance, and Requirements
<H2> What Service Does IT Service Provider Offer?
<H2> What Does the IT Service Provider Do?
<H2> Types of IT Service Provider
<H2> Why Do Companies Require IT Service Providers?
<H2> How to Choose the Right IT Service Provider?'''
)

context_input = st.text_area(
    "Enter Main Context / Summary (brief text defining the topic):",
    height=100,
    value="IT service providers offer technology services that help businesses manage IT infrastructure, cybersecurity, cloud solutions, and consulting."
)

content_input = st.text_area(
    "Paste full content text here (without headings):",
    height=300,
    value="Paste your content paragraphs here separated by blank lines..."
)

if st.button("Analyze Semantic Flow and Context Alignment"):

    if not headings_input.strip() or not content_input.strip() or not context_input.strip():
        st.error("Please provide Headings, Content, and Context text.")
    else:
        headings = parse_headings(headings_input)
        assigned_content = assign_content_to_headings(headings, content_input)

        # Show semantic flow results
        flow_report = analyze_assigned_content(assigned_content)
        st.subheader("Semantic Flow Analysis")
        st.text(flow_report)

        # Context-aware suggestions
        st.subheader("Context-Aware Paragraph Suggestions")
        paragraphs = split_into_paragraphs(content_input)
        suggestions = generate_specific_suggestions(paragraphs, context_input)
        for idx, para, suggestion in suggestions:
            preview = para if len(para) < 200 else para[:197] + "..."
            st.markdown(f"**Paragraph {idx+1}:** {preview}")
            st.markdown(f"- Suggestion: {suggestion}")
            st.markdown("---")
