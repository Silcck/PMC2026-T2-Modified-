import streamlit as st
import pandas as pd
import os
from standardize_data import standardize_project_data
from project_network import ProjectNetwork
import matplotlib.pyplot as plt

st.set_page_config(page_title="Project Optimization Engine", layout="wide")

st.title("🏗️ Project construction Optimization Engine")
st.markdown("Standardize your project data, analyze the Critical Path, and optimize for Time or Budget.")

# Sidebar - File Upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Project Dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = "temp_uploaded_data" + os.path.splitext(uploaded_file.name)[1]
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 1. Standardization
    with st.spinner("Standardizing data..."):
        try:
            standardized_df = standardize_project_data(temp_path)
            if standardized_df is None or standardized_df.empty:
                st.error("Standardization failed. Please check the file format.")
                st.stop()
        except Exception as e:
            st.error(f"Error during standardization: {e}")
            st.stop()

    # Initialize Project Network
    standardized_csv = "standardized_project_dataset.csv"
    project = ProjectNetwork(standardized_csv)
    baseline_results = project.calculate_cpm()
    
    baseline_time = project.total_project_time
    baseline_cost = project.total_project_cost

    # Main Page - Baseline Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baseline Total Time", f"{baseline_time:.2f} units")
    with col2:
        st.metric("Baseline Total Cost", f"€{baseline_cost:,.2f}")

    st.subheader("Baseline AON Diagram")
    # Generate Graphviz AON
    dot_baseline = project.visualize_aon_graphviz(max_nodes=300)
    if dot_baseline:
        st.graphviz_chart(dot_baseline, use_container_width=True)
    else:
        st.warning(f"Graph too large ({len(project.G.nodes)} nodes) for Graphviz visualization. Displaying summary table instead.")
        st.dataframe(baseline_results[['Task_ID', 'Task_Name', 'ES', 'EF', 'LS', 'LF', 'Slack', 'is_critical']].head(100))

    # Controls - Optimization
    st.divider()
    st.subheader("Optimization Controls")
    
    opt_mode = st.radio("Select Optimization Goal", ["Optimize for Time (Crash)", "Optimize for Budget (Relax)"])
    
    if "Crash" in opt_mode:
        # Target duration slider
        target_val = st.slider("Insert target construction time", 
                               min_value=float(0), 
                               max_value=float(baseline_time), 
                               value=float(baseline_time * 0.9),
                               help="The project will attempt to reduce duration by crashing Labour tasks.")
    else:
        # Target budget slider
        target_val = st.slider("Insert target budget", 
                               min_value=float(0), 
                               max_value=float(baseline_cost), 
                               value=float(baseline_cost * 0.95),
                               help="The project will attempt to reduce budget by relaxing non-critical Labour tasks.")

    if st.button("🚀 Optimize Project"):
        with st.spinner("Running optimization algorithms..."):
            if "Crash" in opt_mode:
                modified_tasks, final_cost = project.crash_time(target_val)
                new_time = project.total_project_time
                new_cost = final_cost
                mod_label = "Crashed Tasks"
            else:
                modified_tasks, final_time = project.relax_cost(target_val)
                new_time = final_time
                new_cost = project.total_project_cost
                mod_label = "Delayed Tasks"

            # Output Page
            st.divider()
            st.header("✨ Optimization Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("NEW Total Time", f"{new_time:.2f} units", delta=f"{new_time - baseline_time:.2f}")
            with res_col2:
                st.metric("NEW Total Cost", f"€{new_cost:,.2f}", delta=f"€{new_cost - baseline_cost:,.2f}", delta_color="inverse")

            st.subheader(f"List of {mod_label}")
            if modified_tasks:
                mod_df = project.results[project.results['Task_ID'].isin(modified_tasks)]
                st.dataframe(mod_df[['Task_ID', 'Task_Name', 'Duration', 'Total_Cost', 'Slack', 'is_critical']])
            else:
                st.info("No tasks were modified. Target might already be met or no candidates available.")

            st.subheader("Optimized AON Diagram")
            dot_opt = project.visualize_aon_graphviz(max_nodes=300, modified_tasks=modified_tasks)
            if dot_opt:
                st.graphviz_chart(dot_opt, use_container_width=True)
            else:
                st.warning("Optimized graph too large for visualization.")

else:
    st.info("Please upload a CSV or Excel dataset in the sidebar to begin.")
    
    # Show example format
    st.subheader("Expected Schema Example")
    example_df = pd.DataFrame({
        "Task_ID": ["T1", "T2"],
        "Dependencies": ["", "T1"],
        "Task_Name": ["Excavation", "Foundation"],
        "Resource_Type": ["Cost of Labour", "Cost of Material"],
        "Unit_Value_EUR": [30.0, 150.0],
        "Quantity_Needed": [10.0, 5.0],
        "Total_Cost": [300.0, 750.0],
        "Crashability": ["Yes", "No"],
        "Crash_Price_Increase_Pct": [0.15, 0.0]
    })
    st.table(example_df)
