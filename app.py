import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

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

def assign_content_to_headings_multiple_levels(headings, content, levels=[1,2]):
    paragraphs = split_into_paragraphs(content)
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

def sims_to_df(sims, level, flow_type):
    rows = []
    for i, (a, b, score, comment) in enumerate(sims):
        # Trim paragraph text to 150 chars for readability
        a_display = a if len(a) <= 150 else a[:147] + '...'
        b_display = b if len(b) <= 150 else b[:147] + '...'
        rows.append({
            'Item 1': a_display,
            'Item 2': b_display,
            'Score': round(score, 3),
            'Remark': comment,
            'Level': f"H{level}",
            'Flow Type': flow_type
        })
    return pd.DataFrame(rows)

def analyze_assigned_content_by_level(assigned):
    all_dfs = []
    for level in sorted(set(item['level'] for item in assigned)):
        items = [item for item in assigned if item['level'] == level]
        titles = [item['heading']['title'] for item in items]

        # Section-to-section flow
        if len(titles) > 1:
            sims = check_similarity(titles, 'section')
            df_section = sims_to_df(sims, level, 'Section-to-Section')
            all_dfs.append(df_section)

        # Paragraph-to-paragraph flow
        for item in items:
            paras = item['paragraphs']
            if len(paras) < 2:
                continue
            sims = check_similarity(paras, 'paragraph')
            df_para = sims_to_df(sims, level, f"Paragraph-to-Paragraph under '{item['heading']['title']}'")
            all_dfs.append(df_para)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Item 1', 'Item 2', 'Score', 'Remark', 'Level', 'Flow Type'])

def define_context(paragraphs):
    embeddings = model.encode(paragraphs)
    similarities = cosine_similarity(embeddings)

    context = []
    for i, sim_row in enumerate(similarities):
        avg_similarity = sum(sim_row) / len(sim_row)
        if avg_similarity < 0.5:
            context.append((paragraphs[i], "Out of context. Consider adding more details."))
        else:
            context.append((paragraphs[i], "Well-aligned with main context."))
    return context

def generate_improvement_suggestions(paragraphs, context):
    suggestions = []
    for i, (para, stat) in enumerate(context):
        if stat == "Out of context. Consider adding more details.":
            suggestions.append(f"Paragraph {i+1} is out of context. Suggest adding a bridging sentence or more background information.")
        else:
            suggestions.append(f"Paragraph {i+1} is aligned with the main context.")
    return suggestions

st.title("Semantic Flow Analysis with Context and Suggestions")

headings_text = st.text_area(
    "Headings (Use <H1>, <H2>, <H3> tags)",
    '''<H1> What is an IT Service Provider? Types, Importance, and Requirements
<H2> What Service Does IT Service Provider Offer?
<H2> What Does the IT Service Provider Do?
<H2> Types of IT Service Provider
<H3> 1. Consultation and Strategy
<H3> 2. Internet Service Provider (ISP)
<H3> 3. Cloud Service Provider
<H3> 4. Network and Cloud Security Service Provider
<H3> 5. Digital Adoption Service Provider
<H3> 6. SaaS Service Provider
<H3> 7. Hosting Service Provider
<H2> Why Do Companies Require IT Service Providers?
<H2> How to Choose the Right IT Service Provider?''',
    height=200
)

content_text = st.text_area(
    "Content (without headings)",
    "Paste full content here...",
    height=300
)

if st.button("Analyze Semantic Flow"):
    if not headings_text.strip() or not content_text.strip():
        st.warning("Please enter both headings and content.")
    else:
        headings = parse_headings(headings_text)
        assigned_content = assign_content_to_headings_multiple_levels(headings, content_text, levels=[1,2])
        flow_df = analyze_assigned_content_by_level(assigned_content)

        paragraphs = split_into_paragraphs(content_text)
        context = define_context(paragraphs)
        suggestions = generate_improvement_suggestions(paragraphs, context)

        st.subheader("Semantic Flow Analysis")
        if flow_df.empty:
            st.write("No flow data to display.")
        else:
            st.dataframe(flow_df[['Level', 'Flow Type', 'Item 1', 'Item 2', 'Score', 'Remark']])

        st.subheader("Improvement Suggestions")
        for s in suggestions:
            st.write(f"- {s}")
