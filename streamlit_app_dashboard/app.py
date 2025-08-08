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

# # --- Authentication Config ---
# # You can modify these credentials or store them securely
# VALID_CREDENTIALS = {
#     "admin": "password123",
#     "tester": "test2024", 
#     "user": "mypassword"
# }

# # --- Authentication Functions ---
# def hash_password(password):
#     """Simple password hashing for basic security"""
#     return hashlib.sha256(password.encode()).hexdigest()

# def check_credentials(username, password):
#     """Verify username and password"""
#     return username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password

# def login_form():
#     """Display login form"""
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
    
#     # # Display demo credentials (remove in production!)
#     # with st.expander("🔍 Demo Credentials (for testing)"):
#     #     st.markdown("**Available accounts:**")
#     #     for user, pwd in VALID_CREDENTIALS.items():
#     #         st.code(f"Username: {user} | Password: {pwd}")
#     #     st.warning("⚠️ Remove this section in production!")

# def logout():
#     """Handle user logout"""
#     st.session_state["authenticated"] = False
#     st.session_state["username"] = None
#     st.rerun()

# # --- Config ---
# MODEL_URL = "http://141.147.118.28:8520/v1/models/rainfall_model:predict"
# PROFILING_URL = "http://141.147.118.28:8520/v1/models/rainfall_model/metadata"  # If available

# # --- Helper Functions ---
# def run_inference_with_detailed_timing(payload):
#     """Run inference with detailed timing breakdown"""
#     timings = {}
    
#     # 1. Request preparation time
#     prep_start = time.time()
#     json_payload = json.dumps(payload)
#     timings['request_preparation'] = time.time() - prep_start
    
#     # 2. Network + Inference time
#     network_start = time.time()
#     try:
#         response = requests.post(MODEL_URL, json=payload)
#         timings['total_request'] = time.time() - network_start
        
#         # 3. Response processing time
#         process_start = time.time()
#         if response.status_code == 200:
#             result = response.json()
#             timings['response_processing'] = time.time() - process_start
            
#             return {
#                 "success": True,
#                 "result": result,
#                 "timings": timings,
#                 "status_code": response.status_code
#             }
#         else:
#             return {
#                 "success": False,
#                 "error": f"HTTP {response.status_code}",
#                 "timings": timings,
#                 "status_code": response.status_code
#             }
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "timings": timings,
#             "status_code": "Error"
#         }

# def run_inference(payload):
#     start = time.time()
#     try:
#         response = requests.post(MODEL_URL, json=payload)
#         latency = time.time() - start
#         return {
#             "latency": latency,
#             "status_code": response.status_code,
#             "response": response.json() if response.status_code == 200 else {},
#         }
#     except Exception as e:
#         return {"latency": None, "status_code": "Error", "response": str(e)}

# def batch_payload(batch_size, feature_count=5):
#     return {
#         "instances": [[round(np.random.random(), 3) for _ in range(feature_count)] for _ in range(batch_size)]
#     }

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

# def try_get_model_metadata():
#     """Attempt to get model metadata for profiling insights"""
#     try:
#         response = requests.get(PROFILING_URL, timeout=5)
#         if response.status_code == 200:
#             return response.json()
#     except:
#         pass
#     return None

# def profile_inference_breakdown(num_samples=10):
#     """Profile inference with detailed timing breakdown"""
#     results = []
#     payload = batch_payload(1)  # Single instance for detailed profiling
    
#     for i in range(num_samples):
#         result = run_inference_with_detailed_timing(payload)
#         if result["success"]:
#             results.append(result["timings"])
    
#     if not results:
#         return None
    
#     # Calculate statistics for each timing component
#     df = pd.DataFrame(results)
#     stats = {
#         'component': list(df.columns),
#         'avg_time_ms': [df[col].mean() * 1000 for col in df.columns],
#         'std_time_ms': [df[col].std() * 1000 for col in df.columns],
#         'min_time_ms': [df[col].min() * 1000 for col in df.columns],
#         'max_time_ms': [df[col].max() * 1000 for col in df.columns]
#     }
    
#     return pd.DataFrame(stats), df

# # --- Main Application ---
# def main_dashboard():
#     """Main dashboard content (only shown after authentication)"""
    
#     # Header with user info and logout
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.title("🔍 Rainfall Model Testing Dashboard")
#         st.markdown("Test and benchmark the deployed TensorFlow model at `141.147.118.28:8520`")
    
#     with col2:
#         st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}!**")
#         if st.button("🚪 Logout", use_container_width=True):
#             logout()

#     # Section 1: Smoke Test
#     st.header("1. 🔹 Functionality Check (Smoke Test)")

#     month = st.slider("Month", 1, 12, 5)
#     wind_speed = st.number_input("Wind Speed", 50, 80, 60)
#     wind_dir = st.slider("Wind Direction", 1, 4, 2)
#     sun = st.number_input("Sun", 45, 69, 55)
#     cloud = st.slider("Cloud Cover", 31, 79, 50)

#     payload = {"instances": [[month, wind_speed, wind_dir, sun, cloud]]}

#     if st.button("Run Smoke Test"):
#         result = run_inference(payload)
#         st.success(f"✅ Status Code: {result['status_code']}")
#         st.write("🔁 Prediction:", result["response"])
#         st.write(f"⚡ Latency: {round(result['latency'] * 1000, 2)} ms")

#     # Section 2: Latency & Throughput Test
#     st.header("2. 🔹 Latency & Throughput Testing")

#     col1, col2 = st.columns(2)
#     with col1:
#         batch_size = st.selectbox("Batch size", [1, 4, 8, 16, 32], index=0)
#     with col2:
#         num_requests = st.slider("Number of Requests", 5, 100, step=5, value=20)

#     if st.button("Run Performance Test"):
#         stats, results = throughput_test(batch_size, num_requests)
#         df_perf = pd.DataFrame([stats])
#         st.write("📊 Performance Stats:")
#         st.dataframe(df_perf)

#         st.bar_chart(pd.DataFrame({
#             "avg_latency_ms": [stats["avg_latency"] * 1000],
#             "p95_latency_ms": [stats["p95_latency"] * 1000],
#             "max_latency_ms": [stats["max_latency"] * 1000],
#             "throughput_rps": [stats["throughput"]],
#         }).T.rename(columns={0: "Metric"}))

#         st.session_state["perf_results"] = results
#         st.session_state["perf_stats"] = df_perf

#     # Section 3: Batch Inference Comparison
#     st.header("3. 🔹 Batch Inference Efficiency")

#     batch_sizes = [1, 4, 8, 16, 32]
#     comparison_stats = []

#     if st.button("Run Batch Comparison"):
#         with st.spinner("Testing..."):
#             for bs in batch_sizes:
#                 stats, _ = throughput_test(bs, 10)
#                 comparison_stats.append(stats)

#         df_compare = pd.DataFrame(comparison_stats)
#         st.line_chart(df_compare.set_index("batch_size")[["avg_latency", "p95_latency", "throughput"]])
#         st.session_state["compare_stats"] = df_compare

#     # Section 4: Profile TensorFlow Inference Time (Advanced)
#     st.header("4. 🔹 Profile TensorFlow Inference Time (Advanced)")

#     st.markdown("""
#     **Comprehensive inference profiling with multiple analysis modes:**
#     - **Timing Breakdown**: Request prep, network, processing components
#     - **Load Analysis**: Performance under different batch sizes
#     - **Variability Study**: Timing consistency and outlier detection
#     - **Payload Impact**: How input size affects performance
#     """)

#     # Profiling Configuration
#     st.subheader("⚙️ Profiling Configuration")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         profile_samples = st.slider("Samples per Test", 10, 100, 25)
        
#     with col2:
#         profile_modes = st.multiselect(
#             "Profiling Modes",
#             ["Timing Breakdown", "Batch Load Analysis", "Variability Study", "Payload Size Impact"],
#             default=["Timing Breakdown", "Variability Study"]
#         )

#     with col3:
#         stress_test = st.checkbox("Include Stress Test", value=False)
#         concurrent_requests = st.slider("Concurrent Requests (if stress)", 1, 20, 5) if stress_test else 1

#     # Run Profiling Button
#     if st.button("🔍 Run Comprehensive Profiling", key="run_profiling"):
#         with st.spinner("Running comprehensive profiling analysis..."):
            
#             profiling_results = {}
            
#             # 1. TIMING BREAKDOWN ANALYSIS
#             if "Timing Breakdown" in profile_modes:
#                 st.subheader("⏱️ Timing Breakdown Analysis")
                
#                 stats_df, raw_df = profile_inference_breakdown(profile_samples)
                
#                 if stats_df is not None:
#                     # Display stats table
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.write("**Component Statistics (ms)**")
#                         display_df = stats_df.round(2)
#                         st.dataframe(display_df, use_container_width=True)
                    
#                     with col2:
#                         # Create timing pie chart
#                         fig, ax = plt.subplots(figsize=(6, 6))
#                         ax.pie(stats_df['avg_time_ms'], labels=stats_df['component'], autopct='%1.1f%%')
#                         ax.set_title('Time Distribution by Component')
#                         st.pyplot(fig)
                    
#                     # Timeline visualization
#                     st.write("**Timing Variability Over Samples**")
#                     fig, ax = plt.subplots(figsize=(10, 4))
#                     for col in raw_df.columns:
#                         ax.plot(raw_df[col] * 1000, label=col, marker='o', markersize=3)
#                     ax.set_xlabel('Sample Number')
#                     ax.set_ylabel('Time (ms)')
#                     ax.set_title('Timing Components Across Samples')
#                     ax.legend()
#                     ax.grid(True, alpha=0.3)
#                     st.pyplot(fig)
                    
#                     profiling_results['timing_breakdown'] = stats_df
            
#             # 2. BATCH LOAD ANALYSIS
#             if "Batch Load Analysis" in profile_modes:
#                 st.subheader("📊 Batch Load Analysis")
                
#                 batch_sizes = [1, 2, 4, 8, 16]
#                 batch_results = []
                
#                 progress_bar = st.progress(0)
#                 for i, batch_size in enumerate(batch_sizes):
#                     payload = batch_payload(batch_size)
#                     batch_timings = []
                    
#                     for _ in range(10):  # 10 samples per batch size
#                         result = run_inference_with_detailed_timing(payload)
#                         if result["success"]:
#                             batch_timings.append(result["timings"]["total_request"] * 1000)
                    
#                     if batch_timings:
#                         batch_results.append({
#                             'batch_size': batch_size,
#                             'avg_latency_ms': np.mean(batch_timings),
#                             'throughput_samples_per_sec': batch_size / (np.mean(batch_timings) / 1000),
#                             'latency_per_sample_ms': np.mean(batch_timings) / batch_size
#                         })
                    
#                     progress_bar.progress((i + 1) / len(batch_sizes))
                
#                 if batch_results:
#                     batch_df = pd.DataFrame(batch_results)
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.write("**Batch Performance Metrics**")
#                         st.dataframe(batch_df.round(2))
                    
#                     with col2:
#                         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                        
#                         # Latency vs batch size
#                         ax1.plot(batch_df['batch_size'], batch_df['avg_latency_ms'], 'bo-')
#                         ax1.set_xlabel('Batch Size')
#                         ax1.set_ylabel('Average Latency (ms)')
#                         ax1.set_title('Latency vs Batch Size')
#                         ax1.grid(True, alpha=0.3)
                        
#                         # Efficiency (latency per sample)
#                         ax2.plot(batch_df['batch_size'], batch_df['latency_per_sample_ms'], 'ro-')
#                         ax2.set_xlabel('Batch Size')
#                         ax2.set_ylabel('Latency per Sample (ms)')
#                         ax2.set_title('Efficiency vs Batch Size')
#                         ax2.grid(True, alpha=0.3)
                        
#                         plt.tight_layout()
#                         st.pyplot(fig)
                    
#                     profiling_results['batch_analysis'] = batch_df
            
#             # 3. VARIABILITY STUDY
#             if "Variability Study" in profile_modes:
#                 st.subheader("📈 Variability Study")
                
#                 # Run more samples for variability analysis
#                 variability_samples = []
#                 payload = batch_payload(1)
                
#                 progress_bar = st.progress(0)
#                 for i in range(profile_samples):
#                     result = run_inference_with_detailed_timing(payload)
#                     if result["success"]:
#                         variability_samples.append(result["timings"]["total_request"] * 1000)
#                     progress_bar.progress((i + 1) / profile_samples)
                
#                 if variability_samples:
#                     # Statistical analysis
#                     mean_time = np.mean(variability_samples)
#                     std_time = np.std(variability_samples)
#                     cv = std_time / mean_time * 100  # Coefficient of variation
                    
#                     # Outlier detection (using IQR method)
#                     q1, q3 = np.percentile(variability_samples, [25, 75])
#                     iqr = q3 - q1
#                     outlier_threshold = q3 + 1.5 * iqr
#                     outliers = [x for x in variability_samples if x > outlier_threshold]
                    
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         st.write("**Variability Metrics**")
#                         variability_metrics = pd.DataFrame({
#                             'Metric': ['Mean (ms)', 'Std Dev (ms)', 'Coefficient of Variation (%)', 
#                                       'Min (ms)', 'Max (ms)', 'Outliers Count', 'Outlier Rate (%)'],
#                             'Value': [f"{mean_time:.2f}", f"{std_time:.2f}", f"{cv:.1f}", 
#                                     f"{min(variability_samples):.2f}", f"{max(variability_samples):.2f}",
#                                     f"{len(outliers)}", f"{len(outliers)/len(variability_samples)*100:.1f}"]
#                         })
#                         st.dataframe(variability_metrics, use_container_width=True)
                    
#                     with col2:
#                         # Histogram with outlier highlighting
#                         fig, ax = plt.subplots(figsize=(6, 4))
#                         ax.hist(variability_samples, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
#                         if outliers:
#                             ax.axvline(outlier_threshold, color='red', linestyle='--', 
#                                       label=f'Outlier Threshold ({outlier_threshold:.1f} ms)')
#                             ax.legend()
#                         ax.set_xlabel('Response Time (ms)')
#                         ax.set_ylabel('Frequency')
#                         ax.set_title('Response Time Distribution')
#                         ax.grid(True, alpha=0.3)
#                         st.pyplot(fig)
                    
#                     # Performance assessment
#                     if cv < 10:
#                         st.success(f"✅ **Excellent consistency** (CV: {cv:.1f}%)")
#                     elif cv < 20:
#                         st.info(f"ℹ️ **Good consistency** (CV: {cv:.1f}%)")
#                     else:
#                         st.warning(f"⚠️ **High variability detected** (CV: {cv:.1f}%)")
                    
#                     if len(outliers) > len(variability_samples) * 0.05:  # >5% outliers
#                         st.warning(f"⚠️ **High outlier rate**: {len(outliers)/len(variability_samples)*100:.1f}% of requests")
                    
#                     profiling_results['variability'] = {
#                         'samples': variability_samples,
#                         'metrics': variability_metrics
#                     }
            
#             # 4. PAYLOAD SIZE IMPACT
#             if "Payload Size Impact" in profile_modes:
#                 st.subheader("📦 Payload Size Impact Analysis")
                
#                 # Test different payload sizes (number of features)
#                 payload_configs = [
#                     {"name": "Small (5 features)", "features": 5, "instances": 1},
#                     {"name": "Medium (10 features)", "features": 10, "instances": 1},
#                     {"name": "Large (5 features, 10 instances)", "features": 5, "instances": 10},
#                     {"name": "XL (10 features, 10 instances)", "features": 10, "instances": 10}
#                 ]
                
#                 payload_results = []
#                 progress_bar = st.progress(0)
                
#                 for i, config in enumerate(payload_configs):
#                     payload = {
#                         "instances": [[round(np.random.random(), 3) for _ in range(config["features"])] 
#                                      for _ in range(config["instances"])]
#                     }
                    
#                     timings = []
#                     payload_sizes = []
                    
#                     for _ in range(10):
#                         payload_size = len(json.dumps(payload))
#                         result = run_inference_with_detailed_timing(payload)
#                         if result["success"]:
#                             timings.append(result["timings"]["total_request"] * 1000)
#                             payload_sizes.append(payload_size)
                    
#                     if timings:
#                         payload_results.append({
#                             'config': config["name"],
#                             'avg_payload_size_bytes': np.mean(payload_sizes),
#                             'avg_latency_ms': np.mean(timings),
#                             'throughput_mb_per_sec': (np.mean(payload_sizes) / 1024 / 1024) / (np.mean(timings) / 1000)
#                         })
                    
#                     progress_bar.progress((i + 1) / len(payload_configs))
                
#                 if payload_results:
#                     payload_df = pd.DataFrame(payload_results)
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.write("**Payload Impact Results**")
#                         st.dataframe(payload_df.round(3))
                    
#                     with col2:
#                         fig, ax = plt.subplots(figsize=(6, 4))
#                         ax.scatter(payload_df['avg_payload_size_bytes'], payload_df['avg_latency_ms'], 
#                                   s=100, alpha=0.7)
#                         for i, row in payload_df.iterrows():
#                             ax.annotate(row['config'].split(' ')[0], 
#                                       (row['avg_payload_size_bytes'], row['avg_latency_ms']),
#                                       xytext=(5, 5), textcoords='offset points', fontsize=8)
#                         ax.set_xlabel('Payload Size (bytes)')
#                         ax.set_ylabel('Average Latency (ms)')
#                         ax.set_title('Latency vs Payload Size')
#                         ax.grid(True, alpha=0.3)
#                         st.pyplot(fig)
                    
#                     profiling_results['payload_impact'] = payload_df
            
#             # 5. STRESS TEST (if enabled)
#             if stress_test:
#                 st.subheader("🔥 Stress Test Analysis")
                
#                 with st.spinner(f"Running stress test with {concurrent_requests} concurrent requests..."):
#                     # Run concurrent requests
#                     payload = batch_payload(1)
                    
#                     def stress_run():
#                         return run_inference_with_detailed_timing(payload)
                    
#                     start_time = time.time()
#                     with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
#                         futures = [executor.submit(stress_run) for _ in range(profile_samples)]
#                         stress_results = [f.result() for f in concurrent.futures.as_completed(futures)]
#                     total_time = time.time() - start_time
                    
#                     # Analyze stress test results
#                     successful_requests = [r for r in stress_results if r["success"]]
#                     success_rate = len(successful_requests) / len(stress_results) * 100
                    
#                     if successful_requests:
#                         stress_latencies = [r["timings"]["total_request"] * 1000 for r in successful_requests]
#                         avg_latency = np.mean(stress_latencies)
#                         actual_throughput = len(successful_requests) / total_time
                        
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.write("**Stress Test Results**")
#                             stress_metrics = pd.DataFrame({
#                                 'Metric': ['Success Rate (%)', 'Average Latency (ms)', 
#                                          'Throughput (req/sec)', 'Total Test Time (s)'],
#                                 'Value': [f"{success_rate:.1f}", f"{avg_latency:.2f}",
#                                         f"{actual_throughput:.2f}", f"{total_time:.2f}"]
#                             })
#                             st.dataframe(stress_metrics)
                            
#                             if success_rate < 95:
#                                 st.error(f"❌ **Poor reliability under load**: {success_rate:.1f}% success rate")
#                             elif success_rate < 99:
#                                 st.warning(f"⚠️ **Some failures under load**: {success_rate:.1f}% success rate")
#                             else:
#                                 st.success(f"✅ **Excellent reliability**: {success_rate:.1f}% success rate")
                        
#                         with col2:
#                             fig, ax = plt.subplots(figsize=(6, 4))
#                             ax.hist(stress_latencies, bins=15, alpha=0.7, color='orange', edgecolor='black')
#                             ax.axvline(np.mean(stress_latencies), color='red', linestyle='--', 
#                                       label=f'Mean: {np.mean(stress_latencies):.1f} ms')
#                             ax.set_xlabel('Response Time (ms)')
#                             ax.set_ylabel('Frequency')
#                             ax.set_title(f'Latency Distribution Under Load ({concurrent_requests} concurrent)')
#                             ax.legend()
#                             ax.grid(True, alpha=0.3)
#                             st.pyplot(fig)
                        
#                         profiling_results['stress_test'] = stress_metrics
            
#             # Store all profiling results
#             st.session_state["profiling_results"] = profiling_results
            
#             # Summary insights
#             st.subheader("💡 Profiling Summary & Recommendations")
            
#             recommendations = []
            
#             if "timing_breakdown" in profiling_results:
#                 timing_df = profiling_results["timing_breakdown"]
#                 network_time = timing_df[timing_df['component'] == 'total_request']['avg_time_ms'].iloc[0]
#                 total_time = timing_df['avg_time_ms'].sum()
#                 network_pct = (network_time / total_time) * 100
                
#                 if network_pct > 85:
#                     recommendations.append("🌐 **Network/Server Optimization**: Network+inference time dominates. Consider model optimization, caching, or edge deployment.")
#                 elif network_pct < 30:
#                     recommendations.append("⚡ **Client Optimization**: High client-side overhead. Consider request batching or payload optimization.")
            
#             if "variability" in profiling_results and len(profiling_results["variability"]["samples"]) > 0:
#                 cv = np.std(profiling_results["variability"]["samples"]) / np.mean(profiling_results["variability"]["samples"]) * 100
#                 if cv > 25:
#                     recommendations.append("📊 **Consistency Issues**: High response time variability. Investigate server load balancing or resource scaling.")
            
#             if "batch_analysis" in profiling_results:
#                 batch_df = profiling_results["batch_analysis"]
#                 efficiency_improvement = batch_df.iloc[-1]['latency_per_sample_ms'] / batch_df.iloc[0]['latency_per_sample_ms']
#                 if efficiency_improvement < 0.7:
#                     recommendations.append("📦 **Batching Benefits**: Significant efficiency gains with larger batches. Consider request batching for production.")
            
#             if recommendations:
#                 for rec in recommendations:
#                     st.write(rec)
#             else:
#                 st.success("✅ **Performance looks good!** No major issues detected in profiling analysis.")
            
#             st.success("🎉 **Profiling Complete!** All selected analyses have been run successfully.")

#     # Section 5: Download Report
#     st.header("5. 📥 Download Test Reports")

#     if "perf_stats" in st.session_state:
#         csv = st.session_state["perf_stats"].to_csv(index=False)
#         st.download_button("Download Performance Report", data=csv, file_name="performance_report.csv", mime="text/csv")

#     if "compare_stats" in st.session_state:
#         csv = st.session_state["compare_stats"].to_csv(index=False)
#         st.download_button("Download Batch Comparison Report", data=csv, file_name="batch_comparison_report.csv", mime="text/csv")

#     if "profiling_results" in st.session_state:
#         # Create comprehensive profiling report
#         profiling_data = st.session_state["profiling_results"]
        
#         # Combine all profiling results into one report
#         report_sections = []
        
#         if "timing_breakdown" in profiling_data:
#             report_sections.append("=== TIMING BREAKDOWN ===")
#             report_sections.append(profiling_data["timing_breakdown"].to_csv(index=False))
        
#         if "batch_analysis" in profiling_data:
#             report_sections.append("\n=== BATCH ANALYSIS ===")
#             report_sections.append(profiling_data["batch_analysis"].to_csv(index=False))
        
#         if "variability" in profiling_data:
#             report_sections.append("\n=== VARIABILITY METRICS ===")
#             report_sections.append(profiling_data["variability"]["metrics"].to_csv(index=False))
        
#         if "payload_impact" in profiling_data:
#             report_sections.append("\n=== PAYLOAD IMPACT ===")
#             report_sections.append(profiling_data["payload_impact"].to_csv(index=False))
        
#         if "stress_test" in profiling_data:
#             report_sections.append("\n=== STRESS TEST ===")
#             report_sections.append(profiling_data["stress_test"].to_csv(index=False))
        
#         comprehensive_report = "\n".join(report_sections)
        
#         st.download_button(
#             "📊 Download Comprehensive Profiling Report", 
#             data=comprehensive_report, 
#             file_name="comprehensive_profiling_report.csv", 
#             mime="text/csv"
#         )

#     st.markdown("---")
#     st.caption("Built for TensorFlow Serving Inference Analysis 🚀")


# # --- Main Application Entry Point ---
# def main():
#     """Main application entry point with authentication check"""
    
#     # Initialize session state
#     if "authenticated" not in st.session_state:
#         st.session_state["authenticated"] = False
#     if "username" not in st.session_state:
#         st.session_state["username"] = None
    
#     # Check authentication status
#     if not st.session_state["authenticated"]:
#         login_form()
#     else:
#         main_dashboard()


# # Run the application
# if __name__ == "__main__":
#     # Configure Streamlit page
#     st.set_page_config(
#         page_title="Rainfall Model Testing Dashboard",
#         page_icon="🔍",
#         layout="wide",
#         initial_sidebar_state="collapsed"
#     )
    
#     # Add custom CSS for better styling
#     st.markdown("""
#     <style>
#     .main > div {
#         padding-top: 2rem;
#     }
#     .stButton > button {
#         width: 100%;
#     }
#     .login-container {
#         max-width: 400px;
#         margin: 0 auto;
#         padding: 2rem;
#         border: 1px solid #ddd;
#         border-radius: 10px;
#         background-color: #f8f9fa;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Run main application
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
from typing import Dict, List, Any
import hashlib

# --- Authentication Config ---
# You can modify these credentials or store them securely
VALID_CREDENTIALS = {
    "admin": "password123",
    "tester": "test2024", 
    "user": "mypassword"
}

# --- Authentication Functions ---
def hash_password(password):
    """Simple password hashing for basic security"""
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password):
    """Verify username and password"""
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
    """Handle user logout"""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()

# --- Config ---
MODEL_URL = "http://172.166.149.252:8502/v1/models/rainfall_model:predict"
PROFILING_URL = "http://172.166.149.252:8502/v1/models/rainfall_model/metadata"  # If available

# --- Helper Functions ---
def run_inference_with_detailed_timing(payload):
    """Run inference with detailed timing breakdown"""
    timings = {}
    
    # 1. Request preparation time
    prep_start = time.time()
    json_payload = json.dumps(payload)
    timings['request_preparation'] = time.time() - prep_start
    
    # 2. Network + Inference time
    network_start = time.time()
    try:
        response = requests.post(MODEL_URL, json=payload)
        timings['total_request'] = time.time() - network_start
        
        # 3. Response processing time
        process_start = time.time()
        if response.status_code == 200:
            result = response.json()
            timings['response_processing'] = time.time() - process_start
            
            return {
                "success": True,
                "result": result,
                "timings": timings,
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "timings": timings,
                "status_code": response.status_code
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timings": timings,
            "status_code": "Error"
        }

def run_inference(payload):
    start = time.time()
    try:
        response = requests.post(MODEL_URL, json=payload)
        latency = time.time() - start
        return {
            "latency": latency,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else {},
        }
    except Exception as e:
        return {"latency": None, "status_code": "Error", "response": str(e)}

def batch_payload(batch_size, feature_count=5):
    return {
        "instances": [[round(np.random.random(), 3) for _ in range(feature_count)] for _ in range(batch_size)]
    }

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
    """Profile inference with detailed timing breakdown"""
    results = []
    payload = batch_payload(1)  # Single instance for detailed profiling
    
    for i in range(num_samples):
        result = run_inference_with_detailed_timing(payload)
        if result["success"]:
            results.append(result["timings"])
    
    if not results:
        return None
    
    # Calculate statistics for each timing component
    df = pd.DataFrame(results)
    stats = {
        'component': list(df.columns),
        'avg_time_ms': [df[col].mean() * 1000 for col in df.columns],
        'std_time_ms': [df[col].std() * 1000 for col in df.columns],
        'min_time_ms': [df[col].min() * 1000 for col in df.columns],
        'max_time_ms': [df[col].max() * 1000 for col in df.columns]
    }
    
    return pd.DataFrame(stats), df

# --- Tab Functions ---
def smoke_test_tab():
    """Smoke Test Tab Content"""
    st.header("🔹 Functionality Check (Smoke Test)")
    st.markdown("Test basic model functionality with custom inputs")

    col1, col2 = st.columns(2)
    
    with col1:
        month = st.slider("Month", 1, 12, 5)
        wind_speed = st.number_input("Wind Speed", 50, 80, 60)
        wind_dir = st.slider("Wind Direction", 1, 4, 2)
    
    with col2:
        sun = st.number_input("Sun", 45, 69, 55)
        cloud = st.slider("Cloud Cover", 31, 79, 50)

    payload = {"instances": [[month, wind_speed, wind_dir, sun, cloud]]}

    if st.button("🚀 Run Smoke Test", use_container_width=True):
        with st.spinner("Running test..."):
            result = run_inference(payload)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status Code", result['status_code'])
        with col2:
            st.metric("Latency (ms)", f"{round(result['latency'] * 1000, 2)}")
        with col3:
            if result['status_code'] == 200:
                st.success("✅ Test Passed")
            else:
                st.error("❌ Test Failed")
        
        st.subheader("Response Details")
        st.json(result["response"])

def performance_test_tab():
    """Performance Test Tab Content"""
    st.header("🔹 Latency & Throughput Testing")
    st.markdown("Test model performance under different load conditions")

    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.selectbox("Batch Size", [1, 4, 8, 16, 32], index=0)
    with col2:
        num_requests = st.slider("Number of Requests", 5, 100, step=5, value=20)
    with col3:
        max_workers = st.slider("Concurrent Workers", 1, 20, value=10)

    if st.button("🚀 Run Performance Test", use_container_width=True):
        with st.spinner("Running performance test..."):
            stats, results = throughput_test(batch_size, num_requests, max_workers)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Latency (ms)", f"{round(stats['avg_latency'] * 1000, 2)}")
        with col2:
            st.metric("P95 Latency (ms)", f"{round(stats['p95_latency'] * 1000, 2)}")
        with col3:
            st.metric("Max Latency (ms)", f"{round(stats['max_latency'] * 1000, 2)}")
        with col4:
            st.metric("Throughput (req/s)", f"{round(stats['throughput'], 2)}")
        
        # Performance chart
        chart_data = pd.DataFrame({
            "Metric": ["Avg Latency", "P95 Latency", "Max Latency"],
            "Value (ms)": [stats["avg_latency"] * 1000, stats["p95_latency"] * 1000, stats["max_latency"] * 1000]
        })
        st.bar_chart(chart_data.set_index("Metric"))
        
        # Store results in session state
        st.session_state["perf_results"] = results
        st.session_state["perf_stats"] = pd.DataFrame([stats])

def batch_comparison_tab():
    """Batch Comparison Tab Content"""
    st.header("🔹 Batch Inference Efficiency")
    st.markdown("Compare performance across different batch sizes")

    test_requests = st.slider("Requests per batch size", 5, 50, value=10)
    batch_sizes = st.multiselect("Batch Sizes to Test", [1, 2, 4, 8, 16, 32], default=[1, 4, 8, 16])

    if st.button("🚀 Run Batch Comparison", use_container_width=True):
        comparison_stats = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, bs in enumerate(batch_sizes):
            status_text.text(f"Testing batch size {bs}...")
            stats, _ = throughput_test(bs, test_requests)
            comparison_stats.append(stats)
            progress_bar.progress((i + 1) / len(batch_sizes))
        
        status_text.text("✅ Batch comparison complete!")
        
        df_compare = pd.DataFrame(comparison_stats)
        
        # Display results table
        st.subheader("📊 Comparison Results")
        display_cols = ['batch_size', 'avg_latency', 'p95_latency', 'throughput']
        st.dataframe(df_compare[display_cols].round(4))
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Latency by Batch Size")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_compare['batch_size'], df_compare['avg_latency'] * 1000, 'bo-', label='Avg Latency')
            ax.plot(df_compare['batch_size'], df_compare['p95_latency'] * 1000, 'ro-', label='P95 Latency')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Throughput by Batch Size")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_compare['batch_size'], df_compare['throughput'], 'go-')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (req/s)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.session_state["compare_stats"] = df_compare

def profiling_tab():
    """Advanced Profiling Tab Content"""
    st.header("🔹 Advanced Profiling & Analysis")
    st.markdown("Comprehensive performance analysis with multiple profiling modes")

    # Profiling Configuration
    st.subheader("⚙️ Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        profile_samples = st.slider("Samples per Test", 10, 100, 25)
        
    with col2:
        profile_modes = st.multiselect(
            "Profiling Modes",
            ["Timing Breakdown", "Batch Load Analysis", "Variability Study", "Payload Size Impact"],
            default=["Timing Breakdown"]
        )

    with col3:
        stress_test = st.checkbox("Include Stress Test", value=False)
        if stress_test:
            concurrent_requests = st.slider("Concurrent Requests", 1, 20, 5)

    if not profile_modes:
        st.warning("⚠️ Please select at least one profiling mode")
        return

    if st.button("🔍 Run Comprehensive Profiling", use_container_width=True):
        with st.spinner("Running comprehensive profiling analysis..."):
            
            profiling_results = {}
            
            # 1. TIMING BREAKDOWN ANALYSIS
            if "Timing Breakdown" in profile_modes:
                st.subheader("⏱️ Timing Breakdown Analysis")
                
                stats_df, raw_df = profile_inference_breakdown(profile_samples)
                
                if stats_df is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Component Statistics (ms)**")
                        display_df = stats_df.round(2)
                        st.dataframe(display_df, use_container_width=True)
                    
                    with col2:
                        # Create timing pie chart
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(stats_df['avg_time_ms'], labels=stats_df['component'], autopct='%1.1f%%')
                        ax.set_title('Time Distribution by Component')
                        st.pyplot(fig)
                    
                    profiling_results['timing_breakdown'] = stats_df
            
            # 2. VARIABILITY STUDY
            if "Variability Study" in profile_modes:
                st.subheader("📈 Variability Study")
                
                variability_samples = []
                payload = batch_payload(1)
                
                for i in range(profile_samples):
                    result = run_inference_with_detailed_timing(payload)
                    if result["success"]:
                        variability_samples.append(result["timings"]["total_request"] * 1000)
                
                if variability_samples:
                    mean_time = np.mean(variability_samples)
                    std_time = np.std(variability_samples)
                    cv = std_time / mean_time * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Variability Metrics**")
                        variability_metrics = pd.DataFrame({
                            'Metric': ['Mean (ms)', 'Std Dev (ms)', 'Coefficient of Variation (%)', 
                                      'Min (ms)', 'Max (ms)'],
                            'Value': [f"{mean_time:.2f}", f"{std_time:.2f}", f"{cv:.1f}", 
                                    f"{min(variability_samples):.2f}", f"{max(variability_samples):.2f}"]
                        })
                        st.dataframe(variability_metrics, use_container_width=True)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(variability_samples, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Response Time (ms)')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Response Time Distribution')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Performance assessment
                    if cv < 10:
                        st.success(f"✅ **Excellent consistency** (CV: {cv:.1f}%)")
                    elif cv < 20:
                        st.info(f"ℹ️ **Good consistency** (CV: {cv:.1f}%)")
                    else:
                        st.warning(f"⚠️ **High variability detected** (CV: {cv:.1f}%)")
                    
                    profiling_results['variability'] = {
                        'samples': variability_samples,
                        'metrics': variability_metrics
                    }
            
            # Store profiling results
            st.session_state["profiling_results"] = profiling_results
            
            st.success("🎉 **Profiling Complete!** All selected analyses have been run successfully.")

def reports_tab():
    """Reports and Downloads Tab Content"""
    st.header("📥 Test Reports & Downloads")
    st.markdown("Download comprehensive reports from your test runs")

    if "perf_stats" not in st.session_state and "compare_stats" not in st.session_state and "profiling_results" not in st.session_state:
        st.info("ℹ️ No test results available yet. Run some tests first to generate reports.")
        return

    st.subheader("📊 Available Reports")

    # Performance Report
    if "perf_stats" in st.session_state:
        st.write("**Performance Test Report**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Latest performance test results with latency and throughput metrics")
        with col2:
            csv = st.session_state["perf_stats"].to_csv(index=False)
            st.download_button("📥 Download", data=csv, file_name="performance_report.csv", mime="text/csv")

    # Batch Comparison Report
    if "compare_stats" in st.session_state:
        st.write("**Batch Comparison Report**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Batch size efficiency analysis with performance comparisons")
        with col2:
            csv = st.session_state["compare_stats"].to_csv(index=False)
            st.download_button("📥 Download", data=csv, file_name="batch_comparison_report.csv", mime="text/csv")

    # Profiling Report
    if "profiling_results" in st.session_state:
        st.write("**Comprehensive Profiling Report**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Advanced profiling analysis with timing breakdowns and variability studies")
        with col2:
            # Create comprehensive profiling report
            profiling_data = st.session_state["profiling_results"]
            
            report_sections = []
            
            if "timing_breakdown" in profiling_data:
                report_sections.append("=== TIMING BREAKDOWN ===")
                report_sections.append(profiling_data["timing_breakdown"].to_csv(index=False))
            
            if "variability" in profiling_data:
                report_sections.append("\n=== VARIABILITY METRICS ===")
                report_sections.append(profiling_data["variability"]["metrics"].to_csv(index=False))
            
            comprehensive_report = "\n".join(report_sections)
            
            st.download_button(
                "📥 Download", 
                data=comprehensive_report, 
                file_name="profiling_report.csv", 
                mime="text/csv"
            )

    st.subheader("📈 Quick Stats Overview")
    if "perf_stats" in st.session_state:
        latest_stats = st.session_state["perf_stats"].iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Avg Latency", f"{round(latest_stats['avg_latency'] * 1000, 1)} ms")
        with col2:
            st.metric("Last P95 Latency", f"{round(latest_stats['p95_latency'] * 1000, 1)} ms")
        with col3:
            st.metric("Last Throughput", f"{round(latest_stats['throughput'], 2)} req/s")
        with col4:
            st.metric("Last Batch Size", f"{int(latest_stats['batch_size'])}")

# --- Main Dashboard ---
def main_dashboard():
    """Main dashboard content with tabs"""
    
    # Header with user info and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Rainfall Model Testing Dashboard")
        st.markdown("Test and benchmark the deployed TensorFlow model at `141.147.118.28:8520`")
    
    with col2:
        st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}!**")
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    # Create tabs
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

# --- Main Application Entry Point ---
def main():
    """Main application entry point with authentication check"""
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    
    # Check authentication status
    if not st.session_state["authenticated"]:
        login_form()
    else:
        main_dashboard()

# Run the application
if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Rainfall Model Testing Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Run main application
    main()

