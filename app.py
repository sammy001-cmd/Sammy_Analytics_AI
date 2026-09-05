import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import io
import contextlib

# 1. Page Configuration
st.set_page_config(
    page_title="Sammy Analytics | Intelligent Data Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Responsive Custom CSS
st.markdown("""
<style>
    /* --- 1. CLEAN UP THE HEADER --- */
    /* Hide specifically the Deploy button, main menu, and footer. 
       Do NOT hide the toolbar or header so the sidebar toggle stays safe! */
    .stDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .viewerBadge_container__1QSob { display: none !important; }

    /* --- 2. DESKTOP TYPOGRAPHY & LAYOUT --- */
    .block-container { 
        padding-top: 4.5rem !important; /* FIXED: Pushed content down so it doesn't cover the toggle button */
        padding-bottom: 3rem !important; 
        max-width: 1200px;
    }
    .brand-title { font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
    .brand-subtitle { font-size: 0.95rem; color: #64748b; margin-bottom: 1.5rem; }
    
    .history-item { 
        padding: 8px 12px; border-radius: 6px; background-color: #f8fafc; 
        border: 1px solid #e2e8f0; font-size: 0.85rem; color: #334155; 
        margin-bottom: 6px; word-break: break-word; 
    }
    .custom-card {
        background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1.25rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;
    }

    /* --- 3. MOBILE-FIRST UI OVERRIDES --- */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 4rem !important; /* FIXED: Ensure content clears the mobile toggle button */
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .brand-title { font-size: 1.6rem !important; }
        .brand-subtitle { font-size: 0.85rem !important; margin-bottom: 1rem !important; }
        
        button[data-baseweb="tab"] {
            font-size: 0.8rem !important;
            padding: 0.5rem !important;
            margin-right: 0 !important;
            flex-grow: 1; 
        }
        [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# 3. API Key Setup
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in `.streamlit/secrets.toml`. Please add it to proceed.")
    st.stop()

client = Groq(api_key=groq_api_key)

# 4. State Management
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

# 5. Sidebar Navigation
with st.sidebar:
    st.markdown("### ⚡ **Sammy Analytics**")
    st.caption("AI-Powered Dataset Intelligence")
    st.divider()

    st.markdown("#### 📂 **Dataset**")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.markdown("#### 🕒 **Query History**")
    
    if st.session_state.history:
        for query in reversed(st.session_state.history):
            st.markdown(f'<div class="history-item">💬 {query}</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()
    else:
        st.caption("No queries executed yet.")

# 6. Main Dashboard Layout
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown('<div class="brand-title">Data Analysis Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Ask questions in natural language, generate interactive Plotly visuals, and inspect data.</div>', unsafe_allow_html=True)

    # Top Metric Tiles
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Records", f"{df.shape[0]:,}")
    with col2: st.metric("Total Features", f"{df.shape[1]}")
    with col3: st.metric("Missing Values", f"{df.isna().sum().sum():,}")
    with col4: st.metric("Numeric Columns", f"{len(df.select_dtypes(include='number').columns)}")
    st.write("")

    # Workspace Tabs
    tab_chat, tab_data, tab_summary = st.tabs(["💬 Query & Visualizations", "📋 Data Explorer", "📈 Automated Summary"])

    # TAB 1: Query & Execution
    with tab_chat:
        st.markdown("##### **Ask a Question or Request an Interactive Chart**")
        user_query = st.text_input("Query Input", placeholder="e.g. 'Plot an interactive scatter plot of sales vs profit'", label_visibility="collapsed")
        
        if st.button("Generate Analysis", type="primary") and user_query:
            if user_query not in st.session_state.history:
                st.session_state.history.append(user_query)
            with st.spinner("Executing analysis..."):
                prompt = f"""
                You are a Python data analyst working with a DataFrame named 'df'.
                Columns and types: {dict(df.dtypes)}
                User request: "{user_query}"
                Instructions:
                1. Write ONLY valid Python code. No markdown.
                2. Assign text/number answers to 'final_answer'.
                3. For charts, use Plotly (px or go) and assign to 'fig'.
                """
                try:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="openai/gpt-oss-20b", temperature=0
                    )
                    code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
                    local_env = {"df": df, "pd": pd, "plt": plt, "sns": sns, "px": px, "go": go}
                    
                    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                        exec(code, globals(), local_env)
                        
                    st.session_state.current_result = {
                        "query": user_query, "answer": local_env.get("final_answer"),
                        "fig": local_env.get("fig"), "code": code
                    }
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.current_result:
            res = st.session_state.current_result
            st.divider()
            st.markdown(f"#### Results for: *\"{res['query']}\"*")
            if res["answer"]: st.info(res["answer"])
            if res["fig"]:
                if hasattr(res["fig"], "to_plotly_json"): st.plotly_chart(res["fig"], use_container_width=True)
                else: st.pyplot(res["fig"], use_container_width=True)
            with st.expander("Inspect Code"): st.code(res["code"], language="python")

    # TAB 2: Interactive Data Explorer
    with tab_data:
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button("📥 Export CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")

    # TAB 3: Automated AI Summary
    with tab_summary:
        st.markdown("##### **Executive AI Summary**")
        
        if st.button("✨ Generate AI Dataset Report", type="primary"):
            with st.spinner("Analyzing dataset structure and generating report..."):
                meta_info = f"""
                Row Count: {df.shape[0]}
                Column Count: {df.shape[1]}
                Columns and Types: {dict(df.dtypes)}
                Missing Values per Column: {df.isna().sum().to_dict()}
                Sample Data (First 2 rows): {df.head(2).to_dict()}
                """
                
                summary_prompt = f"""
                You are an expert Data Scientist. Review the following metadata for a newly uploaded dataset:
                {meta_info}
                
                Write a concise, professional executive summary of this dataset. 
                Include:
                1. What this dataset appears to be about based on the column names.
                2. Data quality (mentioning missing values or data types).
                3. Two or three suggested questions the user should ask the AI Analyst to get the most value out of this data.
                Output the response in clean Markdown format.
                """
                try:
                    summary_resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": summary_prompt}],
                        model="openai/gpt-oss-20b",
                        temperature=0.3
                    )
                    st.session_state.ai_summary = summary_resp.choices[0].message.content
                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")

        if st.session_state.ai_summary:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown(st.session_state.ai_summary)
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("##### **Column Metadata Overview**")
        meta_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Null Count": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(meta_df, use_container_width=True)

else:
    # Redesigned Empty State / Landing Page
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem 1rem 1rem;">
        <h1 style="font-size: 2.2rem; color: #1e293b; margin-bottom: 0.5rem;">⚡ Sammy Analytics AI</h1>
        <p style="font-size: 1rem; color: #64748b; margin-bottom: 2rem;">
            Your intelligent data companion. Upload a dataset to unlock automated insights, interactive visualizations, and natural language queries.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="custom-card" style="text-align:center; padding: 1rem;">
            <h2 style="margin-bottom: 0;">💬</h2>
            <p style="margin-top: 10px; line-height: 1.3;"><b>Chat with Data</b><br><span style="font-size:0.85rem; color:#64748b;">Ask questions in plain English</span></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="custom-card" style="text-align:center; padding: 1rem;">
            <h2 style="margin-bottom: 0;">📈</h2>
            <p style="margin-top: 10px; line-height: 1.3;"><b>Auto Charts</b><br><span style="font-size:0.85rem; color:#64748b;">Instantly generate Plotly visuals</span></p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="custom-card" style="text-align:center; padding: 1rem;">
            <h2 style="margin-bottom: 0;">✨</h2>
            <p style="margin-top: 10px; line-height: 1.3;"><b>AI Summary</b><br><span style="font-size:0.85rem; color:#64748b;">Get executive reports in one click</span></p>
        </div>
        """, unsafe_allow_html=True)

    # Mobile-friendly upload instructions
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background-color: #f8fafc; border-radius: 10px; border: 1px dashed #cbd5e1;">
        <h4 style="color: #334155; margin-bottom: 0.5rem;">Ready to start?</h4>
        <p style="color: #64748b; margin-bottom: 0; font-size: 0.95rem;">
            Tap the <b>&gt;</b> icon in the top left corner to open the menu and upload your CSV file.
        </p>
    </div>
    """, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from groq import Groq
# import io
# import contextlib

# # 1. Page Configuration
# st.set_page_config(
#     page_title="Sammy Analytics| Intelligent Data Analyst",
#     page_icon="⚡",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
#     /* Hide Streamlit's top-right toolbar (Share/Fork/GitHub/Star/Edit icons) */
#     [data-testid="stToolbar"] {
#         visibility: hidden;
#         height: 0%;
#         position: fixed;
#     }
#     /* Hide "Made with Streamlit" footer */
#     footer {visibility: hidden;}
#     /* Optional: hide the hamburger menu (top-right ⋮) too */
#     #MainMenu {visibility: hidden;}
#     /* Optional: hide the little "Fork"/GitHub badge specifically if it persists */
#     .viewerBadge_container__1QSob {display: none;}
#     .stDeployButton {display: none;}
# </style>
# """, unsafe_allow_html=True)

# # 2. Custom CSS
# st.markdown("""
# <style>
#     .block-container { padding-top: 2rem; padding-bottom: 3rem; }
#     .brand-title { font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
#     .brand-subtitle { font-size: 0.95rem; color: #64748b; margin-bottom: 1.5rem; }
#     .history-item { 
#         padding: 8px 12px; 
#         border-radius: 6px; 
#         background-color: #f8fafc; 
#         border: 1px solid #e2e8f0; 
#         font-size: 0.85rem; 
#         color: #334155; 
#         margin-bottom: 6px; 
#         word-break: break-word; 
#     }
#     .custom-card {
#         background-color: #ffffff;
#         border: 1px solid #e2e8f0;
#         border-radius: 10px;
#         padding: 1.25rem;
#         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
#         margin-bottom: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # 3. API Key Setup
# try:
#     groq_api_key = st.secrets["GROQ_API_KEY"]
# except KeyError:
#     st.error("GROQ_API_KEY not found in `.streamlit/secrets.toml`. Please add it to proceed.")
#     st.stop()

# client = Groq(api_key=groq_api_key)

# # 4. State Management
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "current_result" not in st.session_state:
#     st.session_state.current_result = None
# if "ai_summary" not in st.session_state:
#     st.session_state.ai_summary = None

# # 5. Sidebar Navigation
# with st.sidebar:
#     st.markdown("### ⚡ **Sammy Analytics**")
#     st.caption("AI-Powered Dataset Intelligence")
#     st.divider()

#     st.markdown("#### 📂 **Dataset**")
#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

#     st.divider()
#     st.markdown("#### 🕒 **Query History**")
    
#     if st.session_state.history:
#         for query in reversed(st.session_state.history):
#             st.markdown(f'<div class="history-item">💬 {query}</div>', unsafe_allow_html=True)
#         st.write("")
#         if st.button("Clear History", use_container_width=True):
#             st.session_state.history = []
#             st.session_state.current_result = None
#             st.rerun()
#     else:
#         st.caption("No queries executed yet.")

# # 6. Main Dashboard Layout
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     st.markdown('<div class="brand-title">Data Analysis Workspace</div>', unsafe_allow_html=True)
#     st.markdown('<div class="brand-subtitle">Ask questions in natural language, generate interactive Plotly visuals, and inspect data.</div>', unsafe_allow_html=True)

#     # Top Metric Tiles
#     col1, col2, col3, col4 = st.columns(4)
#     with col1: st.metric("Total Records", f"{df.shape[0]:,}")
#     with col2: st.metric("Total Features", f"{df.shape[1]}")
#     with col3: st.metric("Missing Values", f"{df.isna().sum().sum():,}")
#     with col4: st.metric("Numeric Columns", f"{len(df.select_dtypes(include='number').columns)}")
#     st.write("")

#     # Workspace Tabs
#     tab_chat, tab_data, tab_summary = st.tabs(["💬 Query & Visualizations", "📋 Data Explorer", "📈 Automated Summary"])

#     # TAB 1: Query & Execution
#     with tab_chat:
#         st.markdown("##### **Ask a Question or Request an Interactive Chart**")
#         user_query = st.text_input("Query Input", placeholder="e.g. 'Plot an interactive scatter plot of sales vs profit'", label_visibility="collapsed")
        
#         if st.button("Generate Analysis", type="primary") and user_query:
#             if user_query not in st.session_state.history:
#                 st.session_state.history.append(user_query)
#             with st.spinner("Executing analysis..."):
#                 prompt = f"""
#                 You are a Python data analyst working with a DataFrame named 'df'.
#                 Columns and types: {dict(df.dtypes)}
#                 User request: "{user_query}"
#                 Instructions:
#                 1. Write ONLY valid Python code. No markdown.
#                 2. Assign text/number answers to 'final_answer'.
#                 3. For charts, use Plotly (px or go) and assign to 'fig'.
#                 """
#                 try:
#                     response = client.chat.completions.create(
#                         messages=[{"role": "user", "content": prompt}],
#                         model="openai/gpt-oss-20b", temperature=0
#                     )
#                     code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
#                     local_env = {"df": df, "pd": pd, "plt": plt, "sns": sns, "px": px, "go": go}
                    
#                     with io.StringIO() as buf, contextlib.redirect_stdout(buf):
#                         exec(code, globals(), local_env)
                        
#                     st.session_state.current_result = {
#                         "query": user_query, "answer": local_env.get("final_answer"),
#                         "fig": local_env.get("fig"), "code": code
#                     }
#                 except Exception as e:
#                     st.error(f"Error: {e}")

#         if st.session_state.current_result:
#             res = st.session_state.current_result
#             st.divider()
#             st.markdown(f"#### Results for: *\"{res['query']}\"*")
#             if res["answer"]: st.info(res["answer"])
#             if res["fig"]:
#                 if hasattr(res["fig"], "to_plotly_json"): st.plotly_chart(res["fig"], use_container_width=True)
#                 else: st.pyplot(res["fig"], use_container_width=True)
#             with st.expander("Inspect Code"): st.code(res["code"], language="python")

#     # TAB 2: Interactive Data Explorer
#     with tab_data:
#         st.dataframe(df, use_container_width=True, height=400)
#         st.download_button("📥 Export CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")

#     # TAB 3: Automated AI Summary
#     with tab_summary:
#         st.markdown("##### **Executive AI Summary**")
        
#         if st.button("✨ Generate AI Dataset Report", type="primary"):
#             with st.spinner("Analyzing dataset structure and generating report..."):
#                 meta_info = f"""
#                 Row Count: {df.shape[0]}
#                 Column Count: {df.shape[1]}
#                 Columns and Types: {dict(df.dtypes)}
#                 Missing Values per Column: {df.isna().sum().to_dict()}
#                 Sample Data (First 2 rows): {df.head(2).to_dict()}
#                 """
                
#                 summary_prompt = f"""
#                 You are an expert Data Scientist. Review the following metadata for a newly uploaded dataset:
#                 {meta_info}
                
#                 Write a concise, professional executive summary of this dataset. 
#                 Include:
#                 1. What this dataset appears to be about based on the column names.
#                 2. Data quality (mentioning missing values or data types).
#                 3. Two or three suggested questions the user should ask the AI Analyst to get the most value out of this data.
#                 Output the response in clean Markdown format.
#                 """
#                 try:
#                     summary_resp = client.chat.completions.create(
#                         messages=[{"role": "user", "content": summary_prompt}],
#                         model="openai/gpt-oss-20b",
#                         temperature=0.3
#                     )
#                     st.session_state.ai_summary = summary_resp.choices[0].message.content
#                 except Exception as e:
#                     st.error(f"Failed to generate summary: {e}")

#         if st.session_state.ai_summary:
#             st.markdown('<div class="custom-card">', unsafe_allow_html=True)
#             st.markdown(st.session_state.ai_summary)
#             st.markdown('</div>', unsafe_allow_html=True)

#         st.divider()
#         st.markdown("##### **Column Metadata Overview**")
#         meta_df = pd.DataFrame({
#             "Data Type": df.dtypes.astype(str),
#             "Non-Null Count": df.notnull().sum(),
#             "Null Count": df.isnull().sum(),
#             "Unique Values": df.nunique()
#         })
#         st.dataframe(meta_df, use_container_width=True)

# else:
#     st.markdown('<div class="brand-title">Sammy Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="brand-subtitle">Upload a CSV dataset in the sidebar navigation to get started.</div>', unsafe_allow_html=True)
#     st.info("👈 Please use the navigation menu on the left to upload your data.")