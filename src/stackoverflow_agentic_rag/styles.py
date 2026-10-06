"""Presentation styles for the Streamlit workspace."""

STYLE = """
<style>
:root {
    --canvas: #f8faf9;
    --paper: #ffffff;
    --ink: #26332d;
    --muted: #62736a;
    --jade: #176b53;
    --line: #dce5df;
    --violet: #7562a8;
}
html, body, [data-testid="stApp"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
    letter-spacing: 0;
}
[data-testid="stAppViewContainer"] { background: var(--canvas); }
[data-testid="stHeader"] { background: var(--canvas); }
[data-testid="stMainBlockContainer"] {
    max-width: 1200px;
    padding: 2.4rem 3rem 4rem;
}
h1, h2, h3, p, label, input, textarea, button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 0 !important;
}
h1 {
    font-size: 2rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
}
h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { line-height: 1.65; }
[data-testid="stVerticalBlock"] { min-width: 0; }
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] code {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .86rem;
}
[data-testid="stMarkdownContainer"] pre {
    background: #eef3f0;
    border: 1px solid var(--line);
    border-radius: 6px;
}
[data-testid="stSidebar"] {
    background: #edf3ef;
    border-right: 1px solid var(--line);
    min-width: 250px;
    max-width: 250px;
}
[data-testid="stSidebarUserContent"] { padding: 2rem 1.3rem; }
.brand { margin: 0 0 2.4rem; }
.brand strong { display: block; font-size: 1.3rem; font-weight: 600; }
.brand span { display: block; color: var(--muted); margin-top: .2rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: .5rem .2rem;
    font-size: .95rem;
}
[data-testid="stSidebar"] hr { border-color: var(--line); margin: 1.7rem 0; }
[data-testid="stTextArea"] textarea {
    background: var(--paper);
    color: var(--ink);
    font-size: 1rem;
    line-height: 1.6;
}
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-radius: 6px;
    border: 1px solid #c7d5cc;
}
[data-testid="stForm"] { padding: 0; }
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: var(--jade);
    border: 1px solid var(--jade);
    color: white;
    border-radius: 6px;
    min-height: 42px;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
    background: #12543f;
    border-color: #12543f;
    color: white;
}
button[kind="secondary"], button[kind="secondaryFormSubmit"] {
    background: var(--paper);
    border-color: #c7d5cc;
    color: var(--ink);
    border-radius: 6px;
    min-height: 40px;
}
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 3px solid #8cbbab !important;
    outline-offset: 3px;
}
[data-testid="stMetric"] { padding: .75rem 0; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .8rem; }
[data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 500; }
[data-testid="stCaptionContainer"] { color: var(--muted); }
[data-testid="stExpander"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 6px;
}
[data-testid="stAlert"] { border-radius: 6px; }
hr { border-color: var(--line); }
.result-question { color: var(--muted); font-size: .95rem; margin: .2rem 0 1rem; }
.st-key-answer_content { max-width: 68ch; }
.verdict {
    display: inline-block;
    font-size: .85rem;
    font-weight: 600;
    border-radius: 4px;
    padding: .3rem .65rem;
}
.verdict-relevant { background: #e1efe7; color: #16543b; }
.verdict-partly { background: #fff0d7; color: #765012; }
.verdict-none { background: #f7e6e7; color: #873743; }
.empty-answer { border-top: 1px solid var(--line); padding-top: 1.5rem; color: var(--muted); }
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] { padding: 1.8rem 1.15rem 3rem; }
    h1 { font-size: 1.6rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""
