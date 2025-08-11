import streamlit as st
import requests
import time
import concurrent.futures
import pandas as pd
import numpy as np
import io
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import hashlib
import os

# --- Authentication Config ---
VALID_CREDENTIALS = {
    "admin": "password123",
    "tester": "test2024", 
    "user": "mypassword"
}

# --- Authentication Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password):
    return username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password

def login_form():
    st.title("🔐 Rainfall Model Dashboard - Login")
    st.markdown("Please enter your credentials to access the testing dashboard.")
    
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)
            if submitted:
                if check_credentials(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success("✅ Login successful! Redirecting to dashboard...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()

# --- Config ---
MODEL_HOST = os.getenv("MODEL_HOST", "localhost")
MODEL_PORT = os.getenv("MODEL_PORT", "8083")

MODEL_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model:predict"
PROFILING_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model/metadata"

# --- Helper Functions ---
def run_inference_with_detailed_timing(payload):
    timings = {}
    prep_start = time.time()
    json_payload = json.dumps(payload)
    timings['request_preparation'] = time.time() - prep_start
    
    network_start = time.time()
    try:
        response = requests.post(MODEL_URL, json=payload)
        timings['total_request'] = time.time() - network_start
        process_start = time.time()
        if response.status_code == 200:
            result = response.json()
            timings['response_processing'] = time.time() - process_start
            return {"success": True, "result": result, "timings": timings, "status_code": response.status_code}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "timings": timings, "status_code": response.status_code}
    except Exception as e:
        return {"success": False, "error": str(e), "timings": timings, "status_code": "Error"}

def run_inference(payload):
    start = time.time()
    try:
        response = requests.post(MODEL_URL, json=payload)
        latency = time.time() - start
        return {"latency": latency, "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else {}}
    except Exception as e:
        return {"latency": None, "status_code": "Error", "response": str(e)}

def batch_payload(batch_size, feature_count=5):
    return {"instances": [[round(np.random.random(), 3) for _ in range(feature_count)] for _ in range(batch_size)]}

def throughput_test(batch_size, num_requests, max_workers=10):
    payload = batch_payload(batch_size)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_inference, payload) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    latencies = [r["latency"] for r in results if r["latency"] is not None]
    throughput = len(latencies) / sum(latencies) if latencies else 0
    stats = {
        "batch_size": batch_size,
        "num_requests": num_requests,
        "avg_latency": np.mean(latencies) if latencies else None,
        "p95_latency": np.percentile(latencies, 95) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
        "throughput": throughput
    }
    return stats, results

def profile_inference_breakdown(num_samples=10):
    results = []
    payload = batch_payload(1)
    for _ in range(num_samples):
        result = run_inference_with_detailed_timing(payload)
        if result["success"]:
            results.append(result["timings"])
    if not results:
        return None
    df = pd.DataFrame(results)
    stats = {
        'component': list(df.columns),
        'avg_time_ms': [df[col].mean() * 1000 for col in df.columns],
        'std_time_ms': [df[col].std() * 1000 for col in df.columns],
        'min_time_ms': [df[col].min() * 1000 for col in df.columns],
        'max_time_ms': [df[col].max() * 1000 for col in df.columns]
    }
    return pd.DataFrame(stats), df

# --- Tabs (unchanged except they use MODEL_URL dynamically) ---
# [Keep all your existing tab functions here without IP changes because MODEL_URL is now dynamic]

# --- Main Dashboard ---
def main_dashboard():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Rainfall Model Testing Dashboard")
        st.markdown(f"Test and benchmark the deployed TensorFlow model at `{MODEL_HOST}:{MODEL_PORT}`")
    with col2:
        st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}!**")
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧪 Smoke Test", 
        "⚡ Performance", 
        "📊 Batch Analysis", 
        "🔍 Advanced Profiling", 
        "📥 Reports"
    ])
    
    with tab1:
        smoke_test_tab()
    with tab2:
        performance_test_tab()
    with tab3:
        batch_comparison_tab()
    with tab4:
        profiling_tab()
    with tab5:
        reports_tab()

    st.markdown("---")
    st.caption("Built for TensorFlow Serving Inference Analysis 🚀")

# --- Main ---
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if not st.session_state["authenticated"]:
        login_form()
    else:
        main_dashboard()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Rainfall Model Testing Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown("""
    <style>
    .main > div { padding-top: 1rem; }
    .stButton > button { width: 100%; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 16px; }
    .stTabs [data-baseweb="tab"] { height: 50px; }
    </style>
    """, unsafe_allow_html=True)
    main()
