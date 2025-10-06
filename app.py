# CSV Dashboard Generator - Streamlit App
# Single-file app: app.py
# Features:
# - Upload a CSV
# - Preview data
# - Ask plain-English queries (e.g., "Show sales by region as a bar chart", "Top 10 products by revenue")
# - Heuristic NL->action parser (no paid LLM required). Optionally use OPENAI_API_KEY to translate complex queries.
# - Interactive charts (plotly)
# - Export chart as PNG and data as CSV

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import re
import textwrap
import os

# Optional OpenAI usage: if you set OPENAI_API_KEY in env, the app will try to use OpenAI to parse complex queries.
USE_OPENAI = bool(os.getenv('OPENAI_API_KEY'))
if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        USE_OPENAI = False

st.set_page_config(page_title="CSV Dashboard Generator", layout="wide")

st.title("CSV Dashboard Generator")
st.write("Upload a CSV, ask a plain-English question, and get an interactive dashboard.")

# Sidebar: upload and options
with st.sidebar:
    st.header("Upload / Settings")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    sample_data = st.selectbox("Or try sample dataset:", ["(none)", "Sample sales data"], index=0)
    if sample_data != "(none)":
        uploaded_file = None

    st.markdown("---")
    st.write("Optional: Set default number of rows to show for top-N queries")
    TOP_N_DEFAULT = st.number_input("Top N default", min_value=1, max_value=100, value=10)

# Load data
@st.cache_data
def load_sample_sales():
    # Simulated sample sales dataset
    import numpy as np
    n = 500
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n)
    df = pd.DataFrame({
        'order_id': range(1, n+1),
        'order_date': rng,
        'region': pd.Categorical(pd.np.random.choice(['North','South','East','West'], size=n)),
        'product': pd.np.random.choice(['Phone','Charger','Headset','Cover','Tablet'], size=n),
        'units_sold': pd.np.random.randint(1, 10, size=n),
        'unit_price': pd.np.random.uniform(10, 1000, size=n).round(2)
    })
    df['revenue'] = (df['units_sold'] * df['unit_price']).round(2)
    return df

if sample_data == "Sample sales data":
    df = load_sample_sales()
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    df = None

if df is None:
    st.info("Upload a CSV or pick the sample dataset from the sidebar to get started.")
    st.stop()

st.subheader("Data preview")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(50))

# Utility: try to coerce datetime-like columns
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass

# Natural-language parsing heuristics
CHART_KEYWORDS = {
    'bar': ['bar', 'bars', 'histogram', 'count by', 'count'],
    'line': ['trend', 'over time', 'line', 'timeseries', 'by date', 'by month', 'per month'],
    'pie': ['pie', 'share', 'percentage', 'percent'],
    'scatter': ['scatter', 'correlation', 'vs', 'versus'],
    'table': ['table', 'show rows', 'list', 'display rows']
}

AGG_KEYWORDS = {
    'sum': ['sum', 'total', 'revenue', 'sales', 'amount', 'income'],
    'mean': ['average', 'mean', 'avg'],
    'count': ['count', 'number of', 'frequency'],
    'max': ['max', 'maximum', 'highest'],
    'min': ['min', 'minimum', 'lowest']
}


def detect_chart_type(query: str):
    q = query.lower()
    for ctype, kws in CHART_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                return ctype
    # fallback heuristics
    if 'vs' in q or ' vs ' in q:
        return 'scatter'
    return 'table'


def detect_agg(query: str):
    q = query.lower()
    for agg, kws in AGG_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                return agg
    return 'sum'


def find_columns(query: str, df: pd.DataFrame):
    # naive matching: look for column names in the query (case-insensitive)
    q = query.lower()
    cols = list(df.columns)
    matched = []
    for c in cols:
        if c.lower() in q:
            matched.append(c)
    # also try partial matches for common words
    if not matched:
        for c in cols:
            root = re.sub(r'[^a-zA-Z0-9]', ' ', c).lower()
            for token in root.split():
                if token and token in q:
                    matched.append(c)
                    break
    return matched


def extract_top_n(query: str, default=TOP_N_DEFAULT):
    m = re.search(r'top\s*(\d+)', query.lower())
    if m:
        return int(m.group(1))
    m = re.search(r'\b(\d+)\s*(largest|biggest|top)\b', query.lower())
    if m:
        return int(m.group(1))
    return default


# Function to handle query and produce figure or table

def handle_query(query: str, df: pd.DataFrame):
    # If OpenAI available and query is complex, call LLM to produce an instruction (optional)
    if USE_OPENAI:
        try:
            prompt = textwrap.dedent(f"""
            You are a helpful assistant that converts a plain-English data request into a short structured JSON describing the action.
            Data columns: {list(df.columns)}
            User query: "{query}"

            Output JSON format:
            {{
              "action": "plot/table",
              "chart": "bar|line|pie|scatter|table",
              "x": "column_name or null",
              "y": "column_name or null",
              "agg": "sum|mean|count|max|min|none",
              "filters": "optional SQL-like filter or null",
              "top_n": number or null
            }}
            """)
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            text = resp.output_text.strip()
            # attempt to parse JSON from text
            import json
            import re
            m = re.search(r'\{.*\}', text, re.S)
            if m:
                spec = json.loads(m.group(0))
                # apply spec
                return execute_spec(spec, df)
        except Exception:
            pass

    # Heuristic path
    chart = detect_chart_type(query)
    agg = detect_agg(query)
    cols = find_columns(query, df)
    top_n = extract_top_n(query)

    # Simple heuristics mapping
    if chart in ['bar', 'pie']:
        # require a categorical x and numeric y
        if len(cols) >= 1:
            x = cols[0]
            # find numeric column to aggregate
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            y = None
            for n in numeric:
                if n in cols[1:]:
                    y = n
                    break
            if not y and numeric:
                y = numeric[0]
            # build grouped frame
            if agg == 'count' or y is None:
                out = df.groupby(x).size().reset_index(name='count').sort_values(by='count', ascending=False)
                fig = px.bar(out, x=x, y='count', title=f'{query}') if chart == 'bar' else px.pie(out, names=x, values='count', title=f'{query}')
                return {'type':'chart', 'figure':fig, 'data':out}
            else:
                func = {'sum':'sum','mean':'mean','max':'max','min':'min'}.get(agg,'sum')
                out = df.groupby(x)[y].agg(func).reset_index().sort_values(by=y, ascending=False)
                fig = px.bar(out, x=x, y=y, title=f'{query}') if chart == 'bar' else px.pie(out, names=x, values=y, title=f'{query}')
                return {'type':'chart', 'figure':fig, 'data':out}

    if chart == 'line':
        # look for date/time column
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if date_cols:
            x = date_cols[0]
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            y = numeric[0] if numeric else None
            if y is None:
                out = df.groupby(x).size().reset_index(name='count')
                fig = px.line(out, x=x, y='count', title=query)
                return {'type':'chart','figure':fig,'data':out}
            else:
                out = df.set_index(x).resample('D')[y].agg(agg if agg!='count' else 'sum').reset_index()
                fig = px.line(out, x=x, y=y, title=query)
                return {'type':'chart','figure':fig,'data':out}

    if chart == 'scatter':
        # try to find two numeric cols
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric) >= 2:
            x, y = numeric[0], numeric[1]
            fig = px.scatter(df, x=x, y=y, hover_data=df.columns, title=query)
            return {'type':'chart','figure':fig,'data':df[[x,y]]}

    # Fallback: show table or filtered result
    # Try find filter like "where region = North"
    filters = None
    m = re.search(r'where\s+(.+)', query.lower())
    if m:
        filters = m.group(1)
    # If top N
    if 'top' in query.lower() or 'largest' in query.lower():
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        y = numeric[0] if numeric else None
        if y:
            out = df.sort_values(by=y, ascending=False).head(top_n)
            return {'type':'table','figure':None,'data':out}

    return {'type':'table','figure':None,'data':df.head(200)}


def execute_spec(spec: dict, df: pd.DataFrame):
    # Minimal executor for JSON spec produced by an LLM
    action = spec.get('action')
    chart = spec.get('chart')
    x = spec.get('x')
    y = spec.get('y')
    agg = spec.get('agg')
    top_n = spec.get('top_n') or TOP_N_DEFAULT
    filters = spec.get('filters')
    tmp = df.copy()
    # apply filters - naive
    if filters:
        try:
            tmp = tmp.query(filters)
        except Exception:
            pass
    if chart in ['bar','pie'] and x:
        if not y:
            out = tmp.groupby(x).size().reset_index(name='count')
            if chart=='bar':
                fig = px.bar(out, x=x, y='count', title=str(spec))
            else:
                fig = px.pie(out, names=x, values='count', title=str(spec))
            return {'type':'chart','figure':fig,'data':out}
    return {'type':'table','figure':None,'data':tmp.head(200)}

# UI: query input and results
st.subheader("Ask a question about the data (plain English)")
query = st.text_input("Example: 'Show revenue by region as a bar chart' or 'Top 5 products by revenue'", value='Show revenue by region as a bar chart')
if st.button('Run Query'):
    if not query.strip():
        st.warning('Please type a question or instruction.')
    else:
        with st.spinner('Parsing query and generating dashboard...'):
            result = handle_query(query, df)
            if result['type'] == 'chart' and result['figure'] is not None:
                st.plotly_chart(result['figure'], use_container_width=True)
                st.markdown('**Data used for chart**')
                st.dataframe(result['data'])
                # Provide CSV download
                csv_buffer = io.StringIO()
                result['data'].to_csv(csv_buffer, index=False)
                st.download_button('Download data as CSV', csv_buffer.getvalue(), file_name='chart_data.csv')
            else:
                st.markdown('**Result table**')
                st.dataframe(result['data'])
                csv_buffer = io.StringIO()
                result['data'].to_csv(csv_buffer, index=False)
                st.download_button('Download table as CSV', csv_buffer.getvalue(), file_name='result_table.csv')

# Tips and next steps
st.sidebar.markdown('---')
st.sidebar.header('Tips & Extensions')
st.sidebar.markdown('''
- Use explicit column names in queries (e.g., "Show sum of revenue by region").
- For time-series, ensure your date column is recognized as date.
- To enable better NL parsing, set an OPENAI_API_KEY in environment; the app will use an LLM to convert complex queries to structured specs.

Extensions you can implement for your project submission:
1. Add embeddings + RAG to answer factual questions about the dataset (e.g., "Which months had sales > 10000?")
2. Add a small vector DB (FAISS) to store dataset schema and common queries for retrieval.
3. Build user auth and save dashboards per user.
4. Add export to PDF or dashboard layout builder.
''')

st.markdown('---')
st.caption('CSV Dashboard Generator — lightweight starter for your college project. Customize the NL parsing and add RAG/LLM capabilities to meet prompt-engineering requirements.')
