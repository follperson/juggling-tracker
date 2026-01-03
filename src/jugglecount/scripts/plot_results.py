import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def parse_results(base_dir):
    data = []
    
    # Iterate over subdirectories like "af1.mov_n_320"
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Parse folder name parts
        # Expected format: {video}_{model_suffix}_{resolution}
        # e.g. af1.mov_n_320
        parts = folder.rsplit('_', 2)
        if len(parts) != 3:
            continue
            
        video_name = parts[0]
        model_size = parts[1] # n, s, m, l, x
        resolution = int(parts[2])
        
        json_path = os.path.join(folder_path, "analysis.json")
        if not os.path.exists(json_path):
            continue
            
        try:
            with open(json_path, "r") as f:
                analysis = json.load(f)
                
            runs = analysis.get("runs", [])
            run_lengths = sorted([r.get("throw_count", 0) for r in runs])
            total_throws = sum(run_lengths)
            num_runs = len(run_lengths)
            max_streak = run_lengths[-1] if run_lengths else 0
            
            ground_truth_total = 65
            ground_truth_max = 35
            
            error_total = abs(total_throws - ground_truth_total)
            error_max = abs(max_streak - ground_truth_max)
            
            data.append({
                "Video": video_name,
                "Model": model_size.upper(), # 'N', 'S', ...
                "Resolution": resolution,
                "Total Throws": total_throws,
                "Total Error": error_total,
                "Max Streak": max_streak,
                "Max Error": error_max,
                "Run Count": num_runs,
                "Runs": str(run_lengths)
            })
        except Exception as e:
            print(f"Error parsing {folder}: {e}")
            
    return pd.DataFrame(data)

def plot_benchmark(df, output_path):
    if df.empty:
        print("No data found.")
        return

    # Filter for af1.mov if mixed
    df = df[df["Video"].str.contains("af1")].copy()
    df = df.sort_values(by=["Resolution", "Model"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.set_style("whitegrid")
    
    # 1. Total Throws
    sns.lineplot(data=df, x="Resolution", y="Total Throws", hue="Model", style="Model", markers=True, markersize=8, linewidth=2, ax=axes[0])
    axes[0].axhline(65, color='green', linestyle='--', label='Truth (65)')
    axes[0].set_title("Total Throws (Target: 65)")
    
    # 2. Max Streak
    sns.lineplot(data=df, x="Resolution", y="Max Streak", hue="Model", style="Model", markers=True, markersize=8, linewidth=2, ax=axes[1])
    axes[1].axhline(35, color='green', linestyle='--', label='Truth (35)')
    axes[1].set_title("Max Streak (Target: 35)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Print best performers based on combined error
    df["Combined Error"] = df["Total Error"] + df["Max Error"]
    print("\n🏆 Top 5 Configs by Accuracy (Total + Max Streak):")
    print(df.sort_values("Combined Error")[["Model", "Resolution", "Total Throws", "Max Streak", "Run Count", "Runs"]].head(10))

if __name__ == "__main__":
    base_dir = "outputs/benchmark_results"
    df = parse_results(base_dir)
    print(df)
    plot_benchmark(df, "outputs/accuracy_plot.png")
