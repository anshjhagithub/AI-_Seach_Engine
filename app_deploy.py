import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.callbacks import StreamlitCallbackHandler
import os
import time
import json
from datetime import datetime
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="🚀 AI Search Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI - FIXED VERSION
st.markdown("""
<style>
    /* Fix white boxes - Remove all default white backgrounds */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Remove white boxes from containers */
    div[data-testid="stVerticalBlock"] > div,
    div[data-testid="stHorizontalBlock"] > div,
    .element-container,
    .stContainer,
    .css-1kyxreq {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Fix specific tool card containers */
    .css-12oz5g7,
    .css-1d391kg,
    .css-16huue1 {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Make all form containers transparent */
    .stForm {
        background: transparent !important;
        border: none !important;
    }
    
    /* Fix selectbox and input backgrounds for dark theme */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background-color: #1e1e1e !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    /* Fix button styling */
    .stButton > button {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    /* Fix checkbox styling */
    .stCheckbox {
        background: transparent !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .search-stats {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .tool-card {
        background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: white;
        border: 1px solid #333;
    }
    
    .response-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar-header {
        background: linear-gradient(45deg, #f093fb, #f5576c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: #2b2b2b;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin: 0.5rem 0;
        border: 1px solid #333;
        color: white;
    }
    
    /* Fix metric styling */
    .css-1xarl3l {
        background: transparent !important;
    }
    
    /* Fix expander styling */
    .streamlit-expanderHeader {
        background: transparent !important;
        color: white !important;
    }
    
    /* Fix sidebar background */
    .css-1d391kg {
        background-color: #0e1117 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_count" not in st.session_state:
    st.session_state.search_count = 0
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "favorite_searches" not in st.session_state:
    st.session_state.favorite_searches = []
if "response_times" not in st.session_state:
    st.session_state.response_times = []

# Session cleanup to prevent memory issues
def cleanup_session():
    """Clean up session state to prevent memory accumulation"""
    if len(st.session_state.messages) > 50:  # Keep only last 50 messages
        st.session_state.messages = st.session_state.messages[-50:]
    
    if len(st.session_state.search_history) > 100:  # Keep only last 100 searches
        st.session_state.search_history = st.session_state.search_history[-100:]
    
    if len(st.session_state.response_times) > 100:  # Keep only last 100 response times
        st.session_state.response_times = st.session_state.response_times[-100:]

# Get API key from secrets or user input
def get_default_api_key():
    """Get API key from Streamlit secrets if available"""
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except:
        return ""

# Tools setup with error handling
@st.cache_resource
def setup_tools():
    """Setup tools with proper error handling"""
    try:
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=500)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
        wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        
        search = DuckDuckGoSearchRun(name="WebSearch")
        
        return [search, arxiv, wiki]
    except Exception as e:
        st.error(f"Error initializing search tools: {str(e)}")
        return []

tools = setup_tools()

# Create LLM with retry logic
def create_llm_with_retry(api_key, model_name, max_retries=3):
    """Create LLM with retry logic and timeout"""
    for attempt in range(max_retries):
        try:
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                streaming=True,
                temperature=0.1,
                timeout=30
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)

# Memory usage check
def check_memory_usage():
    """Simple memory usage check"""
    total_messages = len(st.session_state.get('messages', []))
    total_history = len(st.session_state.get('search_history', []))
    
    if total_messages > 100 or total_history > 200:
        st.warning("⚠️ Session data is getting large. Consider clearing history.")
        return True
    return False

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Search by Ansh Jha</h1>
    <p>Advanced Multi-Source AI Search Engine with Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🎛️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    default_key = get_default_api_key()
    api_key = st.text_input(
        "Enter your Groq API Key:", 
        type="password", 
        help="Get your API key from Groq Console",
        value=default_key
    )
    
    # Model Selection
    model_options = {
        "Llama3-8B (Fast)": "llama3-8b-8192",
        "Llama3-70B (Powerful)": "llama3-70b-8192",
        "Mixtral-8x7B (Balanced)": "mixtral-8x7b-32768"
    }
    selected_model = st.selectbox("🤖 Select AI Model:", list(model_options.keys()))
    
    # Search Settings
    st.subheader("⚙️ Search Settings")
    search_depth = st.select_slider("Search Depth:", options=["Quick", "Standard", "Deep"], value="Standard")
    max_results = st.slider("Max Results per Tool:", 1, 5, 3)
    enable_sources = st.checkbox("Show Sources", value=True)
    
    # Search Statistics
    st.subheader("📊 Search Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Searches", st.session_state.search_count)
    with col2:
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times) if st.session_state.response_times else 0
        st.metric("Avg Response Time", f"{avg_time:.1f}s")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.search_history = []
        st.session_state.response_times = []
        st.session_state.search_count = 0
        st.rerun()
    
    if st.button("📥 Export Chat"):
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "search_count": st.session_state.search_count
        }
        st.download_button(
            "Download Chat JSON",
            json.dumps(chat_data, indent=2),
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Memory usage warning
    if check_memory_usage():
        if st.button("🧹 Clean Session Data"):
            cleanup_session()
            st.success("Session data cleaned!")
            st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Search Interface
    st.subheader("🔍 Search Interface")
    
    # Quick search suggestions
    st.markdown("**💡 Quick Suggestions:**")
    suggestion_cols = st.columns(4)
    suggestions = [
        "Latest AI research",
        "Climate change solutions",
        "Quantum computing basics",
        "Space exploration news"
    ]
    
    for i, suggestion in enumerate(suggestions):
        if suggestion_cols[i].button(f"🔸 {suggestion}", key=f"suggest_{i}"):
            st.session_state.selected_suggestion = suggestion
    
    # Main search input
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="Ask anything... I'll search across web, academic papers, and Wikipedia!",
        value=st.session_state.get('selected_suggestion', ''),
        key="main_search"
    )
    
    # Advanced search options
    with st.expander("🔧 Advanced Search Options"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            search_type = st.radio("Search Type:", ["General", "Academic", "News", "Technical"])
        with col_b:
            language = st.selectbox("Language:", ["English", "Spanish", "French", "German"])
        with col_c:
            time_filter = st.selectbox("Time Filter:", ["Any time", "Past day", "Past week", "Past month"])

with col2:
    st.markdown("""
    <div style="background: transparent; padding: 0;">
        <h3 style="color: white; margin-bottom: 1rem;">🛠️ Active Tools</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic tool status based on actual availability
    tool_status = {}
    if tools:
        tool_status = {
            "🌐 Web Search": "✅ Ready",
            "📚 ArXiv Papers": "✅ Ready",
            "📖 Wikipedia": "✅ Ready"
        }
    else:
        tool_status = {
            "🌐 Web Search": "❌ Error",
            "📚 ArXiv Papers": "❌ Error",
            "📖 Wikipedia": "❌ Error"
        }
    
    for tool, status in tool_status.items():
        st.markdown(f"""
        <div class="tool-card">
            <strong>{tool}</strong><br>
            <small>{status}</small>
        </div>
        """, unsafe_allow_html=True)

# Chat Interface
st.subheader("💬 AI Assistant")

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Add action buttons for assistant messages
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("👍 Helpful", key=f"helpful_{i}"):
                    st.success("Thanks for the feedback!")
            with col_btn2:
                if st.button("💾 Save Response", key=f"save_{i}"):
                    st.session_state.favorite_searches.append({
                        "query": message.get("original_query", "Unknown"),
                        "response": message["content"],
                        "timestamp": datetime.now().isoformat()
                    })
                    st.success("Response saved!")
            with col_btn3:
                if st.button("🔄 Regenerate", key=f"regen_{i}"):
                    st.info("Regenerating response...")

# Process search query
if search_query and api_key:
    if search_query not in [msg.get("query", "") for msg in st.session_state.messages if msg["role"] == "user"]:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": search_query,
            "query": search_query,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.chat_message("user"):
            st.write(search_query)
        
        # Process with AI
        with st.chat_message("assistant"):
            start_time = time.time()
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Check if tools are available
                if not tools:
                    st.error("❌ Search tools are not properly initialized. Please refresh the page.")
                    st.stop()
                
                # Initialize LLM
                status_text.text("🤖 Initializing AI model...")
                progress_bar.progress(20)
                
                llm = create_llm_with_retry(
                    api_key, 
                    model_options[selected_model]
                )
                
                # Setup agent
                status_text.text("🔧 Setting up search agent...")
                progress_bar.progress(40)
                
                react_prompt = hub.pull("hwchase17/react")
                agent = create_react_agent(llm, tools, react_prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=50
                )
                
                # Execute search
                status_text.text("🔍 Searching across multiple sources...")
                progress_bar.progress(60)
                
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                
                # Enhanced prompt based on search settings
                enhanced_query = f"""
                Search Query: {search_query}
                Search Type: {search_type}
                Depth: {search_depth}
                
                Please provide a comprehensive answer with:
                1. Key findings and main points
                2. Multiple perspectives if applicable
                3. Recent developments or updates
                4. Reliable sources and citations
                """
                
                progress_bar.progress(80)
                
                response = agent_executor.invoke(
                    {"input": enhanced_query},
                    {"callbacks": [st_cb]}
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Search completed!")
                
                # Calculate response time
                end_time = time.time()
                response_time = end_time - start_time
                st.session_state.response_times.append(response_time)
                
                # Display response
                final_response = response['output']
                
                st.markdown(f"""
                <div class="response-card">
                    <h4>🎯 Search Results</h4>
                    <p>{final_response}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add metadata
                st.markdown(f"""
                <div class="search-stats">
                    ⏱️ Response Time: {response_time:.2f}s | 
                    🤖 Model: {selected_model} | 
                    🔍 Sources: Web + ArXiv + Wikipedia
                </div>
                """, unsafe_allow_html=True)
                
                # Update session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "response_time": response_time,
                    "model": selected_model,
                    "original_query": search_query,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.session_state.search_count += 1
                st.session_state.search_history.append({
                    "query": search_query,
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time
                })
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Clean up session if needed
                cleanup_session()
                
            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "original_query": search_query
                })
                
                # Clear progress indicators on error
                progress_bar.empty()
                status_text.empty()

elif search_query and not api_key:
    st.warning("🔑 Please enter your Groq API key in the sidebar to start searching!")
elif not search_query and api_key:
    st.info("💡 Enter a search query above to get started!")

# Analytics Dashboard
if st.session_state.search_history:
    with st.expander("📈 Analytics Dashboard"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time chart
            if len(st.session_state.response_times) > 1:
                try:
                    df_times = pd.DataFrame({
                        'Search': range(1, len(st.session_state.response_times) + 1),
                        'Response Time (s)': st.session_state.response_times
                    })
                    fig = px.line(df_times, x='Search', y='Response Time (s)', 
                                title='Response Time Trend')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        with col2:
            # Search frequency by hour
            try:
                hours = [datetime.fromisoformat(h['timestamp']).hour for h in st.session_state.search_history]
                df_hours = pd.DataFrame({'Hour': hours})
                fig2 = px.histogram(df_hours, x='Hour', title='Search Activity by Hour')
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 AI Search Pro - Powered by LangChain & Groq | Built by Ansh Jha using Streamlit</p>
    <p><small>Ready for Streamlit Cloud Deployment</small></p>
</div>
""", unsafe_allow_html=True)