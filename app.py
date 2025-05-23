import streamlit as st
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Semantic Flow Analyzer", layout="wide")

# Load models
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, model

nlp, model = load_models()

THRESHOLDS = {
    'sentence': {'low': 0.60, 'ideal': (0.70, 0.90), 'high': 0.95},
    'paragraph': {'low': 0.55, 'ideal': (0.65, 0.85), 'high': 0.90},
    'section': {'low': 0.45, 'ideal': (0.50, 0.80), 'high': 0.85}
}
def check_similarity(text_list, level):
    results = []
    embeddings = model.encode(text_list)
    pairs = list(zip(text_list, text_list[1:], embeddings, embeddings[1:]))
    for a, b, vec_a, vec_b in pairs:
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
    text = re.sub(r'\r\n|\r', '\n', text)
    raw_paragraphs = re.split(r'\n{2,}|\.\s*\n', text)
    return [p.strip() for p in raw_paragraphs if p.strip()]

def parse_headings(text):
    headings = []
    tag_level_map = {
        'h1': 1,
        'h2': 2,
        'h3': 3,
        'h4': 3,
        'h5': 3,
        'h6': 3,
        'li': 3,
        'p': 3,
        'section': 2,
        'article': 2
    }

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        tag = None
        title = None

        # Match <tag> Title
        m1 = re.match(r'<(\w+)[^>]*>\s*(.*)', line, re.IGNORECASE)
        if m1:
            tag = m1.group(1).lower()
            title = m1.group(2).strip()
            if not title and i + 1 < len(lines):
                i += 1
                title = lines[i].strip()
        else:
            # Match tag: title or tag title or tag - title
            m2 = re.match(r'(?i)(h\d|li|section|article|p)\s*[:\-]?\s*(.*)', line)
            if m2:
                tag = m2.group(1).lower()
                title = m2.group(2).strip()
                if not title and i + 1 < len(lines):
                    i += 1
                    title = lines[i].strip()

        if tag in tag_level_map and title:
            level = tag_level_map[tag]
            headings.append({'level': level, 'tag': tag, 'title': title})

        i += 1

    return headings
    
def extract_paragraphs_under_headings(content_text, headings):
    paragraphs = []
    current = None
    buffer = []
    heading_titles = [h['title'] for h in headings]
    lines = content_text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_heading = any(line.lower() == h.lower() for h in heading_titles)
        if is_heading:
            if current and buffer:
                paragraphs.append({'heading': current, 'paragraph': ' '.join(buffer).strip()})
                buffer = []
            current = line
        else:
            buffer.append(line)

    if current and buffer:
        paragraphs.append({'heading': current, 'paragraph': ' '.join(buffer).strip()})
    return paragraphs

def assign_content_to_headings_multiple_levels(headings, content, levels=[1,2,3]):
    paragraphs = [p['paragraph'] for p in extract_paragraphs_under_headings(content, headings)]
    assigned_all = []

    for level in levels:
        top_headings = [h for h in headings if h['level'] == level]
        n = len(paragraphs)
        m = len(top_headings)
        if m == 0:
            continue
        approx_chunk_size = max(1, n // m)

        idx = 0
        for i, heading in enumerate(top_headings):
            if i < m - 1:
                chunk = paragraphs[idx: idx + approx_chunk_size]
                idx += approx_chunk_size
            else:
                chunk = paragraphs[idx:]
            assigned_all.append({'heading': heading, 'paragraphs': chunk, 'level': level})
    return assigned_all

def analyze_assigned_content_by_level(assigned):
    output_lines = []
    for level in sorted(set(item['level'] for item in assigned)):
        items = [item for item in assigned if item['level'] == level]
        titles = [item['heading']['title'] for item in items]
        if len(titles) > 1:
            sims = check_similarity(titles, 'section')
            flow_problem = any(score < THRESHOLDS['section']['low'] for _, _, score, _ in sims)
            if flow_problem:
                output_lines.append(f"⚠ Section flow is distorted/disconnected at heading level {level}. Suggestions:")
                for a, b, score, comment in sims:
                    if score < THRESHOLDS['section']['low']:
                        output_lines.append(f"- Between '{a}' and '{b}': Add bridging sentence or reorganize sections.")
                output_lines.append("")
            output_lines.append(f"=== Section-to-Section Flow (H{level}) ===")
            for a, b, score, comment in sims:
                output_lines.append(f"Between '{a}' and '{b}': {comment} (score: {score:.3f})")
            output_lines.append("")
        else:
            output_lines.append(f"No enough sections for flow analysis at heading level {level}.\n")

        output_lines.append(f"=== Paragraph-to-Paragraph Flow (H{level}) ===")
        for item in items:
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

def extract_key_concepts(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text.strip()) > 2:
            keywords.add(chunk.text.lower())
    return list(keywords)

def paragraph_concept_coverage(paragraph, context_keywords):
    para_doc = nlp(paragraph.lower())
    para_text = para_doc.text
    missing_concepts = []
    for concept in context_keywords:
        if concept not in para_text:
            missing_concepts.append(concept)
    return missing_concepts

def generate_specific_suggestions(paragraphs, context_text):
    context_keywords = extract_key_concepts(context_text)
    suggestions = []
    for idx, para in enumerate(paragraphs):
        missing = paragraph_concept_coverage(para, context_keywords)
        if missing:
            missing_preview = ', '.join(missing[:5])
            suggestion = (
                f"Paragraph {idx+1} misses key concepts: {missing_preview}. "
                "Consider adding explanations, examples, or clarifications about these topics."
            )
        else:
            suggestion = f"Paragraph {idx+1} covers main concepts well."
        suggestions.append((idx, para, suggestion))
    return suggestions

def find_best_aligned_outline(titles):
    if len(titles) <= 1:
        return titles
    embeddings = model.encode(titles)
    remaining = set(range(len(titles)))
    order = [remaining.pop()]
    while remaining:
        last = order[-1]
        next_idx = max(remaining, key=lambda i: cosine_similarity([embeddings[last]], [embeddings[i]])[0][0])
        order.append(next_idx)
        remaining.remove(next_idx)
    return [titles[i] for i in order]

# Streamlit UI
st.title("📚 Semantic Flow Analyzer")

headings_input = st.text_area("Headings (with tags like <H2>, h3: etc)", height=200)
context_input = st.text_area("Context Summary", height=100)
content_input = st.text_area("Main Content", height=300)

if st.button("Analyze Semantic Flow"):
    if not headings_input or not context_input or not content_input:
        st.error("Please fill in all three inputs.")
    else:
        headings = parse_headings(headings_input)
        assigned = assign_content_to_headings_multiple_levels(headings, content_input, levels=[1, 2, 3])
        
        st.subheader("🔍 Semantic Flow Analysis")
        st.text(analyze_assigned_content_by_level(assigned))

        st.subheader("🧠 Context-Aware Paragraph Suggestions")
        paragraphs = split_into_paragraphs(content_input)
        suggestions = generate_specific_suggestions(paragraphs, context_input)
        for idx, para, suggestion in suggestions:
            st.markdown(f"**Paragraph {idx+1}**: {para[:200]}{'...' if len(para) > 200 else ''}")
            st.info(suggestion)

        st.subheader("🧾 Best Aligned Outline")
        all_titles = [item['heading']['title'] for item in assigned]
        for i, t in enumerate(find_best_aligned_outline(all_titles), 1):
            st.markdown(f"{i}. {t}")
