import os
import sys
import argparse
import json
from standardize_data import standardize_project_data
from project_network import ProjectNetwork

def run_pipeline(input_file, target_duration=None, target_budget=None):
    """
    Runs the end-to-end project data pipeline with optimizations.
    """
    print(f"--- Starting Pipeline for: {input_file} ---")
    
    # Step 1: Standardization
    print("\n[1/4] Standardizing data...")
    standardized_df = standardize_project_data(input_file)
    
    if standardized_df is None or standardized_df.empty:
        print("Error: Standardization failed or produced empty output.")
        return

    standardized_csv = "standardized_project_dataset.csv"
    
    # Step 2: CPM Analysis
    print("\n[2/4] Performing Initial CPM Analysis...")
    try:
        project = ProjectNetwork(standardized_csv)
        project.calculate_cpm()
        
        initial_duration = project.total_project_time
        initial_cost = project.total_project_cost
        
        print(f"Initial Duration: {initial_duration}")
        print(f"Initial Cost: €{initial_cost:,.2f}")
        
    except Exception as e:
        print(f"Error during Initial CPM Analysis: {e}")
        return

    optimization_results = {
        "initial_duration": initial_duration,
        "initial_cost": initial_cost,
        "crashes": [],
        "relaxations": []
    }

    # Step 3: Optimizations
    print("\n[3/4] Performing Optimizations...")
    
    # Crashing
    if target_duration:
        print(f"Crashing project to target duration: {target_duration}")
        crashed_tasks, final_cost = project.crash_time(target_duration)
        optimization_results["crashes"] = {
            "target": target_duration,
            "final_duration": project.total_project_time,
            "final_cost": final_cost,
            "tasks_affected": crashed_tasks
        }
        print(f"New Duration after crashing: {project.total_project_time}")
        print(f"New Cost after crashing: €{final_cost:,.2f}")

    # Relaxation (on top of current state)
    if target_budget:
        print(f"Relaxing project to target budget: {target_budget}")
        delayed_tasks, final_duration = project.relax_cost(target_budget)
        optimization_results["relaxations"] = {
            "target": target_budget,
            "final_cost": project.total_project_cost,
            "final_duration": final_duration,
            "tasks_affected": delayed_tasks
        }
        print(f"New Cost after relaxation: €{project.total_project_cost:,.2f}")
        print(f"New Duration after relaxation: {final_duration}")

    # Save results to JSON
    with open("optimization_results.json", "w") as f:
        json.dump(optimization_results, f, indent=4)
    print("\nOptimization results saved to optimization_results.json")

    # Step 4: Visualization
    print("\n[4/4] Generating AON Diagram...")
    try:
        project.visualize_aon()
        print(f"Success! Visualization saved as 'aon_diagram.png'")
    except Exception as e:
        print(f"Error during visualization: {e}")

    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize project data and perform optimized CPM analysis.")
    parser.add_argument("input_file", help="Path to the input CSV or Excel file.")
    parser.add_argument("--target_duration", type=float, help="Target project duration for crashing.")
    parser.add_argument("--target_budget", type=float, help="Target project budget for cost relaxation.")
    
    if len(sys.argv) == 1:
        default_test = "PCM - Dataset (.csv)/Exp.3 - randomised_crashing_dataset.csv"
        if os.path.exists(default_test):
            print(f"No input file provided. Using default test file: {default_test}")
            # For testing, let's target 95% of duration and 95% of cost
            run_pipeline(default_test, target_duration=380000, target_budget=90000000)
        else:
            parser.print_help()
    else:
        args = parser.parse_args()
        run_pipeline(args.input_file, args.target_duration, args.target_budget)
