# # import streamlit as st
# # import requests
# # import time
# # import concurrent.futures
# # import pandas as pd
# # import numpy as np
# # import io
# # import json
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from typing import Dict, List, Any
# # import hashlib
# # import os

# # # --- Authentication Config ---
# # VALID_CREDENTIALS = {
# #     "admin": "password123",
# #     "tester": "test2024", 
# #     "user": "mypassword"
# # }

# # # --- Authentication Functions ---
# # def hash_password(password):
# #     return hashlib.sha256(password.encode()).hexdigest()

# # def check_credentials(username, password):
# #     return username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password

# # def login_form():
# #     st.title("🔐 Rainfall Model Dashboard - Login")
# #     st.markdown("Please enter your credentials to access the testing dashboard.")
    
# #     with st.form("login_form"):
# #         col1, col2, col3 = st.columns([1, 2, 1])
# #         with col2:
# #             st.markdown("### Login")
# #             username = st.text_input("Username", placeholder="Enter your username")
# #             password = st.text_input("Password", type="password", placeholder="Enter your password")
# #             submitted = st.form_submit_button("🚀 Login", use_container_width=True)
# #             if submitted:
# #                 if check_credentials(username, password):
# #                     st.session_state["authenticated"] = True
# #                     st.session_state["username"] = username
# #                     st.success("✅ Login successful! Redirecting to dashboard...")
# #                     time.sleep(1)
# #                     st.rerun()
# #                 else:
# #                     st.error("❌ Invalid username or password. Please try again.")

# # def logout():
# #     st.session_state["authenticated"] = False
# #     st.session_state["username"] = None
# #     st.rerun()

# # # --- Config ---
# # MODEL_HOST = os.getenv("MODEL_HOST", "localhost")
# # MODEL_PORT = os.getenv("MODEL_PORT", "8520")

# # MODEL_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model:predict"
# # PROFILING_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model/metadata"

# # # --- Helper Functions ---
# # def run_inference_with_detailed_timing(payload):
# #     timings = {}
# #     prep_start = time.time()
# #     json_payload = json.dumps(payload)
# #     timings['request_preparation'] = time.time() - prep_start
    
# #     network_start = time.time()
# #     try:
# #         response = requests.post(MODEL_URL, json=payload)
# #         timings['total_request'] = time.time() - network_start
# #         process_start = time.time()
# #         if response.status_code == 200:
# #             result = response.json()
# #             timings['response_processing'] = time.time() - process_start
# #             return {"success": True, "result": result, "timings": timings, "status_code": response.status_code}
# #         else:
# #             return {"success": False, "error": f"HTTP {response.status_code}", "timings": timings, "status_code": response.status_code}
# #     except Exception as e:
# #         return {"success": False, "error": str(e), "timings": timings, "status_code": "Error"}

# # def run_inference(payload):
# #     start = time.time()
# #     try:
# #         response = requests.post(MODEL_URL, json=payload)
# #         latency = time.time() - start
# #         return {"latency": latency, "status_code": response.status_code,
# #                 "response": response.json() if response.status_code == 200 else {}}
# #     except Exception as e:
# #         return {"latency": None, "status_code": "Error", "response": str(e)}

# # def batch_payload(batch_size, feature_count=5):
# #     return {"instances": [[round(np.random.random(), 3) for _ in range(feature_count)] for _ in range(batch_size)]}

# # def throughput_test(batch_size, num_requests, max_workers=10):
# #     payload = batch_payload(batch_size)
# #     results = []
# #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         futures = [executor.submit(run_inference, payload) for _ in range(num_requests)]
# #         for f in concurrent.futures.as_completed(futures):
# #             results.append(f.result())
# #     latencies = [r["latency"] for r in results if r["latency"] is not None]
# #     throughput = len(latencies) / sum(latencies) if latencies else 0
# #     stats = {
# #         "batch_size": batch_size,
# #         "num_requests": num_requests,
# #         "avg_latency": np.mean(latencies) if latencies else None,
# #         "p95_latency": np.percentile(latencies, 95) if latencies else None,
# #         "max_latency": max(latencies) if latencies else None,
# #         "throughput": throughput
# #     }
# #     return stats, results

# # def profile_inference_breakdown(num_samples=10):
# #     results = []
# #     payload = batch_payload(1)
# #     for _ in range(num_samples):
# #         result = run_inference_with_detailed_timing(payload)
# #         if result["success"]:
# #             results.append(result["timings"])
# #     if not results:
# #         return None
# #     df = pd.DataFrame(results)
# #     stats = {
# #         'component': list(df.columns),
# #         'avg_time_ms': [df[col].mean() * 1000 for col in df.columns],
# #         'std_time_ms': [df[col].std() * 1000 for col in df.columns],
# #         'min_time_ms': [df[col].min() * 1000 for col in df.columns],
# #         'max_time_ms': [df[col].max() * 1000 for col in df.columns]
# #     }
# #     return pd.DataFrame(stats), df

# # # --- Tabs (unchanged except they use MODEL_URL dynamically) ---
# # # [Keep all your existing tab functions here without IP changes because MODEL_URL is now dynamic]

# # # --- Main Dashboard ---
# # def main_dashboard():
# #     col1, col2 = st.columns([3, 1])
# #     with col1:
# #         st.title("🔍 Rainfall Model Testing Dashboard")
# #         st.markdown(f"Test and benchmark the deployed TensorFlow model at `{MODEL_HOST}:{MODEL_PORT}`")
# #     with col2:
# #         st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}!**")
# #         if st.button("🚪 Logout", use_container_width=True):
# #             logout()

# #     tab1, tab2, tab3, tab4, tab5 = st.tabs([
# #         "🧪 Smoke Test", 
# #         "⚡ Performance", 
# #         "📊 Batch Analysis", 
# #         "🔍 Advanced Profiling", 
# #         "📥 Reports"
# #     ])
    
# #     with tab1:
# #         smoke_test_tab()
# #     with tab2:
# #         performance_test_tab()
# #     with tab3:
# #         batch_comparison_tab()
# #     with tab4:
# #         profiling_tab()
# #     with tab5:
# #         reports_tab()

# #     st.markdown("---")
# #     st.caption("Built for TensorFlow Serving Inference Analysis 🚀")

# # # --- Main ---
# # def main():
# #     if "authenticated" not in st.session_state:
# #         st.session_state["authenticated"] = False
# #     if "username" not in st.session_state:
# #         st.session_state["username"] = None
# #     if not st.session_state["authenticated"]:
# #         login_form()
# #     else:
# #         main_dashboard()

# # if __name__ == "__main__":
# #     st.set_page_config(
# #         page_title="Rainfall Model Testing Dashboard",
# #         page_icon="🔍",
# #         layout="wide",
# #         initial_sidebar_state="collapsed"
# #     )
# #     st.markdown("""
# #     <style>
# #     .main > div { padding-top: 1rem; }
# #     .stButton > button { width: 100%; }
# #     .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 16px; }
# #     .stTabs [data-baseweb="tab"] { height: 50px; }
# #     </style>
# #     """, unsafe_allow_html=True)
# #     main()





# import streamlit as st
# import requests
# import time
# import concurrent.futures
# import pandas as pd
# import numpy as np
# import io
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List, Any
# import hashlib
# import os

# # --- Authentication Config ---
# VALID_CREDENTIALS = {
#     "admin": "password123",
#     "tester": "test2024", 
#     "user": "mypassword"
# }

# # --- Authentication Functions ---
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def check_credentials(username, password):
#     return username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password

# def login_form():
#     st.title("🔐 Rainfall Model Dashboard - Login")
#     st.markdown("Please enter your credentials to access the testing dashboard.")
    
#     with st.form("login_form"):
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             st.markdown("### Login")
#             username = st.text_input("Username", placeholder="Enter your username")
#             password = st.text_input("Password", type="password", placeholder="Enter your password")
#             submitted = st.form_submit_button("🚀 Login", use_container_width=True)
#             if submitted:
#                 if check_credentials(username, password):
#                     st.session_state["authenticated"] = True
#                     st.session_state["username"] = username
#                     st.success("✅ Login successful! Redirecting to dashboard...")
#                     time.sleep(1)
#                     st.rerun()
#                 else:
#                     st.error("❌ Invalid username or password. Please try again.")

# def logout():
#     st.session_state["authenticated"] = False
#     st.session_state["username"] = None
#     st.rerun()

# # --- Config ---
# MODEL_HOST = os.getenv("MODEL_HOST", "localhost")
# MODEL_PORT = os.getenv("MODEL_PORT", "8520")

# MODEL_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model:predict"
# PROFILING_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model/metadata"

# # --- Helper Functions ---
# def run_inference_with_detailed_timing(payload):
#     timings = {}
#     prep_start = time.time()
#     json_payload = json.dumps(payload)
#     timings['request_preparation'] = time.time() - prep_start
    
#     network_start = time.time()
#     try:
#         response = requests.post(MODEL_URL, json=payload)
#         timings['total_request'] = time.time() - network_start
#         process_start = time.time()
#         if response.status_code == 200:
#             result = response.json()
#             timings['response_processing'] = time.time() - process_start
#             return {"success": True, "result": result, "timings": timings, "status_code": response.status_code}
#         else:
#             return {"success": False, "error": f"HTTP {response.status_code}", "timings": timings, "status_code": response.status_code}
#     except Exception as e:
#         return {"success": False, "error": str(e), "timings": timings, "status_code": "Error"}

# def run_inference(payload):
#     start = time.time()
#     try:
#         response = requests.post(MODEL_URL, json=payload)
#         latency = time.time() - start
#         return {"latency": latency, "status_code": response.status_code,
#                 "response": response.json() if response.status_code == 200 else {}}
#     except Exception as e:
#         return {"latency": None, "status_code": "Error", "response": str(e)}

# def batch_payload(batch_size, feature_count=5):
#     return {"instances": [[round(np.random.random(), 3) for _ in range(feature_count)] for _ in range(batch_size)]}

# def throughput_test(batch_size, num_requests, max_workers=10):
#     payload = batch_payload(batch_size)
#     results = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(run_inference, payload) for _ in range(num_requests)]
#         for f in concurrent.futures.as_completed(futures):
#             results.append(f.result())
#     latencies = [r["latency"] for r in results if r["latency"] is not None]
#     throughput = len(latencies) / sum(latencies) if latencies else 0
#     stats = {
#         "batch_size": batch_size,
#         "num_requests": num_requests,
#         "avg_latency": np.mean(latencies) if latencies else None,
#         "p95_latency": np.percentile(latencies, 95) if latencies else None,
#         "max_latency": max(latencies) if latencies else None,
#         "throughput": throughput
#     }
#     return stats, results

# def profile_inference_breakdown(num_samples=10):
#     results = []
#     payload = batch_payload(1)
#     for _ in range(num_samples):
#         result = run_inference_with_detailed_timing(payload)
#         if result["success"]:
#             results.append(result["timings"])
#     if not results:
#         return None
#     df = pd.DataFrame(results)
#     stats = {
#         'component': list(df.columns),
#         'avg_time_ms': [df[col].mean() * 1000 for col in df.columns],
#         'std_time_ms': [df[col].std() * 1000 for col in df.columns],
#         'min_time_ms': [df[col].min() * 1000 for col in df.columns],
#         'max_time_ms': [df[col].max() * 1000 for col in df.columns]
#     }
#     return pd.DataFrame(stats), df

# # --- Placeholder Tab Functions ---
# def smoke_test_tab():
#     st.subheader("🧪 Smoke Test")
#     st.info("Smoke test functionality coming soon.")

# def performance_test_tab():
#     st.subheader("⚡ Performance Test")
#     st.info("Performance testing functionality coming soon.")

# def batch_comparison_tab():
#     st.subheader("📊 Batch Analysis")
#     st.info("Batch analysis functionality coming soon.")

# def profiling_tab():
#     st.subheader("🔍 Advanced Profiling")
#     st.info("Advanced profiling functionality coming soon.")

# def reports_tab():
#     st.subheader("📥 Reports")
#     st.info("Reports functionality coming soon.")

# # --- Main Dashboard ---
# def main_dashboard():
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.title("🔍 Rainfall Model Testing Dashboard")
#         st.markdown(f"Test and benchmark the deployed TensorFlow model at `{MODEL_HOST}:{MODEL_PORT}`")
#     with col2:
#         st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}!**")
#         if st.button("🚪 Logout", use_container_width=True):
#             logout()

#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "🧪 Smoke Test", 
#         "⚡ Performance", 
#         "📊 Batch Analysis", 
#         "🔍 Advanced Profiling", 
#         "📥 Reports"
#     ])
    
#     with tab1:
#         smoke_test_tab()
#     with tab2:
#         performance_test_tab()
#     with tab3:
#         batch_comparison_tab()
#     with tab4:
#         profiling_tab()
#     with tab5:
#         reports_tab()

#     st.markdown("---")
#     st.caption("Built for TensorFlow Serving Inference Analysis 🚀")

# # --- Main ---
# def main():
#     if "authenticated" not in st.session_state:
#         st.session_state["authenticated"] = False
#     if "username" not in st.session_state:
#         st.session_state["username"] = None
#     if not st.session_state["authenticated"]:
#         login_form()
#     else:
#         main_dashboard()

# if __name__ == "__main__":
#     st.set_page_config(
#         page_title="Rainfall Model Testing Dashboard",
#         page_icon="🔍",
#         layout="wide",
#         initial_sidebar_state="collapsed"
#     )
#     st.markdown("""
#     <style>
#     .main > div { padding-top: 1rem; }
#     .stButton > button { width: 100%; }
#     .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 16px; }
#     .stTabs [data-baseweb="tab"] { height: 50px; }
#     </style>
#     """, unsafe_allow_html=True)
#     main() 



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
from typing import Dict, List, Any, Optional, Tuple, Union
import hashlib
import os
 
# --- Authentication Config ---
VALID_CREDENTIALS = {
    "admin": "password123",
    "tester": "test2024",
    "user": "mypassword"
}
 
# --- Authentication Functions ---
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()
 
def check_credentials(username: str, password: str) -> bool:
    """Verify username and password against valid credentials"""
    return username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password
 
def login_form():
    """Display login form"""
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
    """Clear authentication session and redirect to login"""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()
 
# --- Config ---
MODEL_HOST = os.getenv("MODEL_HOST", "localhost")
MODEL_PORT = os.getenv("MODEL_PORT", "8520")
 
MODEL_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model:predict"
PROFILING_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/models/rainfall_model/metadata"
 
# --- Helper Functions ---
def run_inference_with_detailed_timing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference with detailed timing breakdown"""
    timings = {}
    prep_start = time.time()
    # Removed unused json_payload variable
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
 
def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run single inference request"""
    start = time.time()
    try:
        response = requests.post(MODEL_URL, json=payload)
        latency = time.time() - start
        return {"latency": latency, "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else {}}
    except Exception as e:
        return {"latency": None, "status_code": "Error", "response": str(e)}
 
def batch_payload(batch_size: int, feature_count: int = 5) -> Dict[str, List[List[float]]]:
    """Generate batch payload for inference"""
    return {"instances": [[round(np.random.random(), 3) for _ in range(feature_count)] for _ in range(batch_size)]}
 
def throughput_test(batch_size: int, num_requests: int, max_workers: int = 10) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run throughput test with multiple concurrent requests"""
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
 
def profile_inference_breakdown(num_samples: int = 10) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Profile inference timing breakdown - FIXED: Consistent return types"""
    results = []
    payload = batch_payload(1)
   
    for _ in range(num_samples):
        result = run_inference_with_detailed_timing(payload)
        if result["success"]:
            results.append(result["timings"])
   
    # Always return a tuple for consistency
    if not results:
        return None, None
   
    df = pd.DataFrame(results)
    stats = {
        'component': list(df.columns),
        'avg_time_ms': [df[col].mean() * 1000 for col in df.columns],
        'std_time_ms': [df[col].std() * 1000 for col in df.columns],
        'min_time_ms': [df[col].min() * 1000 for col in df.columns],
        'max_time_ms': [df[col].max() * 1000 for col in df.columns]
    }
    return pd.DataFrame(stats), df
 
# --- Placeholder Tab Functions ---
def smoke_test_tab():
    """Smoke test tab implementation"""
    st.subheader("🧪 Smoke Test")
   
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Test basic model connectivity and response validation")
    with col2:
        if st.button("🚀 Run Smoke Test", use_container_width=True):
            with st.spinner("Running smoke test..."):
                test_payload = batch_payload(1)
                result = run_inference_with_detailed_timing(test_payload)
               
                if result["success"]:
                    st.success("✅ Smoke test passed!")
                    st.json(result["result"])
                else:
                    st.error(f"❌ Smoke test failed: {result.get('error', 'Unknown error')}")
 
def performance_test_tab():
    """Performance test tab implementation"""
    st.subheader("⚡ Performance Test")
   
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
    with col2:
        num_requests = st.number_input("Number of Requests", min_value=1, max_value=1000, value=50)
    with col3:
        max_workers = st.number_input("Concurrent Workers", min_value=1, max_value=50, value=10)
   
    if st.button("🚀 Run Performance Test", use_container_width=True):
        with st.spinner("Running performance test..."):
            stats, results = throughput_test(batch_size, num_requests, max_workers)
           
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Latency", f"{stats['avg_latency']:.3f}s" if stats['avg_latency'] else "N/A")
            with col2:
                st.metric("P95 Latency", f"{stats['p95_latency']:.3f}s" if stats['p95_latency'] else "N/A")
            with col3:
                st.metric("Max Latency", f"{stats['max_latency']:.3f}s" if stats['max_latency'] else "N/A")
            with col4:
                st.metric("Throughput", f"{stats['throughput']:.2f} req/s")
 
def batch_comparison_tab():
    """Batch analysis tab implementation"""
    st.subheader("📊 Batch Analysis")
   
    st.info("Compare performance across different batch sizes")
   
    batch_sizes = st.multiselect(
        "Select batch sizes to compare",
        options=[1, 5, 10, 20, 50, 100],
        default=[1, 10, 50]
    )
   
    num_requests = st.slider("Requests per batch size", 1, 100, 20)
   
    if st.button("🚀 Run Batch Comparison", use_container_width=True) and batch_sizes:
        comparison_results = []
        progress_bar = st.progress(0)
       
        for i, batch_size in enumerate(batch_sizes):
            with st.spinner(f"Testing batch size {batch_size}..."):
                stats, _ = throughput_test(batch_size, num_requests, 5)
                comparison_results.append(stats)
                progress_bar.progress((i + 1) / len(batch_sizes))
       
        # Display results
        df_results = pd.DataFrame(comparison_results)
        st.dataframe(df_results, use_container_width=True)
       
        # Create visualization
        if len(comparison_results) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
           
            # Latency comparison
            ax1.plot(df_results['batch_size'], df_results['avg_latency'], marker='o', label='Avg Latency')
            ax1.plot(df_results['batch_size'], df_results['p95_latency'], marker='s', label='P95 Latency')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Latency (s)')
            ax1.set_title('Latency vs Batch Size')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
           
            # Throughput comparison
            ax2.plot(df_results['batch_size'], df_results['throughput'], marker='o', color='green')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (req/s)')
            ax2.set_title('Throughput vs Batch Size')
            ax2.grid(True, alpha=0.3)
           
            st.pyplot(fig)
 
def profiling_tab():
    """Advanced profiling tab implementation"""
    st.subheader("🔍 Advanced Profiling")
   
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Analyze detailed timing breakdown of inference pipeline")
        num_samples = st.slider("Number of samples", 5, 50, 20)
   
    with col2:
        if st.button("🚀 Run Profiling", use_container_width=True):
            with st.spinner("Profiling inference pipeline..."):
                stats_df, raw_df = profile_inference_breakdown(num_samples)
               
                if stats_df is not None and raw_df is not None:
                    st.success("✅ Profiling completed!")
                   
                    # Display timing statistics
                    st.subheader("📊 Timing Statistics")
                    st.dataframe(stats_df, use_container_width=True)
                   
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x_pos = np.arange(len(stats_df))
                    bars = ax.bar(x_pos, stats_df['avg_time_ms'], yerr=stats_df['std_time_ms'], capsize=5)
                    ax.set_xlabel('Pipeline Component')
                    ax.set_ylabel('Time (ms)')
                    ax.set_title('Inference Pipeline Timing Breakdown')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(stats_df['component'], rotation=45)
                    ax.grid(True, alpha=0.3)
                   
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + stats_df['std_time_ms'].iloc[i],
                               f'{height:.2f}ms', ha='center', va='bottom')
                   
                    plt.tight_layout()
                    st.pyplot(fig)
                   
                    # Show raw data
                    with st.expander("📋 Raw Timing Data"):
                        st.dataframe(raw_df, use_container_width=True)
                else:
                    st.error("❌ Profiling failed - no successful requests")
 
def reports_tab():
    """Reports tab implementation"""
    st.subheader("📥 Reports")
   
    col1, col2 = st.columns(2)
    with col1:
        st.info("Export test results and generate reports")
        report_type = st.selectbox(
            "Select report type",
            ["Performance Summary", "Batch Analysis", "Profiling Report", "Full Test Suite"]
        )
   
    with col2:
        st.markdown("### Quick Actions")
        if st.button("📊 Generate Sample Report", use_container_width=True):
            # Generate a sample report
            sample_data = {
                'Metric': ['Avg Latency', 'P95 Latency', 'Throughput', 'Success Rate'],
                'Value': ['0.125s', '0.234s', '45.2 req/s', '99.8%'],
                'Status': ['✅ Good', '⚠️ Fair', '✅ Good', '✅ Excellent']
            }
           
            st.subheader(f"📋 {report_type}")
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
           
            # Create download link for CSV
            csv = pd.DataFrame(sample_data).to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"rainfall_model_{report_type.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
 
# --- Main Dashboard ---
def main_dashboard():
    """Main dashboard with tabs"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Rainfall Model Testing Dashboard")
        st.markdown(f"Test and benchmark the deployed TensorFlow model at `{MODEL_HOST}:{MODEL_PORT}`")
    with col2:
        st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}!**")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
 
    # Model status check
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                test_result = run_inference(batch_payload(1))
                if test_result["status_code"] == 200:
                    st.success("🟢 Model Online")
                elif test_result["status_code"] == "Error":
                    st.error("🔴 TensorFlow model URL is unreachable/unavailable")
                else:
                    st.error(f"🔴 Model Error (HTTP {test_result['status_code']})")
            except Exception as e:
                st.error("🔴 TensorFlow model URL is unreachable/unavailable")
       
        with col2:
            st.info(f"🌐 Host: {MODEL_HOST}:{MODEL_PORT}")
        with col3:
            st.info(f"🕒 Last Check: {time.strftime('%H:%M:%S')}")
 
    # Main tabs
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
    """Main application entry point"""
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
   
    # Show login or dashboard based on authentication
    if not st.session_state["authenticated"]:
        login_form()
    else:
        main_dashboard()
 
if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Rainfall Model Testing Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
   
    # Custom CSS styling
    st.markdown("""
    <style>
    .main > div { padding-top: 1rem; }
    .stButton > button { width: 100%; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 16px; }
    .stTabs [data-baseweb="tab"] { height: 50px; }
    .stMetric > label { font-size: 14px; }
    .stSuccess, .stError, .stWarning, .stInfo { margin-top: 1rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)
   
    main()
