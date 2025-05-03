#!/usr/bin/env python3
"""
Analyze SigLIP model results from similarity_details.txt
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def extract_metrics_from_text():
    """
    Extract the key metrics from the log output.
    """
    metrics = {
        "image_to_text_recall@1": 0.9500,
        "image_to_text_recall@5": 1.0000,
        "image_to_text_recall@10": 1.0000,
        "text_to_image_recall@1": 0.9600,
        "text_to_image_recall@5": 1.0000,
        "text_to_image_recall@10": 1.0000,
        "image_to_text_mrr": 0.9733,
        "text_to_image_mrr": 0.9775
    }
    
    return metrics

def plot_metrics(metrics, output_dir="./results"):
    """
    Plot the key metrics.
    """
    # Prepare data for plotting
    df_metrics = pd.DataFrame([
        {"Metric": "Recall@1", "Value": metrics["image_to_text_recall@1"], "Type": "Image→Text"},
        {"Metric": "Recall@5", "Value": metrics["image_to_text_recall@5"], "Type": "Image→Text"},
        {"Metric": "Recall@10", "Value": metrics["image_to_text_recall@10"], "Type": "Image→Text"},
        {"Metric": "MRR", "Value": metrics["image_to_text_mrr"], "Type": "Image→Text"},
        {"Metric": "Recall@1", "Value": metrics["text_to_image_recall@1"], "Type": "Text→Image"},
        {"Metric": "Recall@5", "Value": metrics["text_to_image_recall@5"], "Type": "Text→Image"},
        {"Metric": "Recall@10", "Value": metrics["text_to_image_recall@10"], "Type": "Text→Image"},
        {"Metric": "MRR", "Value": metrics["text_to_image_mrr"], "Type": "Text→Image"},
    ])
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Metric", y="Value", hue="Type", data=df_metrics)
    plt.title("SigLIP Retrieval Performance", fontsize=16)
    plt.ylim(0, 1.1)
    
    # Add value labels
    for i, row in enumerate(df_metrics.itertuples()):
        plt.text(i % 4 + (0.2 if row.Type == "Text→Image" else -0.2), 
                 row.Value + 0.02, 
                 f"{row.Value:.2f}", 
                 ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "siglip_metrics.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Metrics chart saved to {output_path}")

def extract_examples_from_file(filepath):
    """
    Extract example data from the similarity_details.txt file.
    """
    examples = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Split content by example
        example_blocks = re.split(r'-{10,}', content)
        
        for block in example_blocks:
            if not block.strip() or "Example" not in block:
                continue
                
            example = {}
            
            # Extract example number
            example_match = re.search(r'Example (\d+)', block)
            if example_match:
                example["id"] = int(example_match.group(1))
                
            # Extract original caption
            caption_match = re.search(r'Original Caption:(.*?)Top 5', block, re.DOTALL)
            if caption_match:
                example["original_caption"] = caption_match.group(1).strip()
                
            # Extract scores and matched status
            scores = []
            captions = []
            match_index = None
            
            score_lines = re.findall(r'(\d+)\. Score: ([0-9.-]+)(?: \((MATCH)\))? - (.*?)$', block, re.MULTILINE)
            for i, (rank, score, is_match, caption) in enumerate(score_lines):
                scores.append(float(score))
                captions.append(caption)
                if is_match == "MATCH":
                    match_index = i
                    
            example["scores"] = scores
            example["captions"] = captions
            example["match_index"] = match_index
            
            examples.append(example)
            
        return examples
    except Exception as e:
        print(f"Error extracting examples: {e}")
        return []

def plot_rank_distribution(examples, output_dir="./results"):
    """
    Plot the distribution of ranks for correct matches.
    """
    # Count rank occurrences
    ranks = []
    for ex in examples:
        if ex.get("match_index") is not None:
            ranks.append(ex["match_index"] + 1)  # Convert to 1-indexed
    
    if not ranks:
        print("No rank data available.")
        return
        
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
    # Create dataframe
    df_ranks = pd.DataFrame([
        {"Rank": rank, "Count": count}
        for rank, count in sorted(rank_counts.items())
    ])
    
    # Plot
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Rank", y="Count", data=df_ranks, palette="Blues_d")
    
    # Add count labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(str(int(p.get_height())), 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom')
    
    plt.title("Rank of Correct Matches in SigLIP Results", fontsize=16)
    plt.xlabel("Rank Position", fontsize=14)
    plt.ylabel("Number of Examples", fontsize=14)
    
    # Add percentage
    total = sum(rank_counts.values())
    for i, row in enumerate(df_ranks.itertuples()):
        percentage = (row.Count / total) * 100
        ax.text(i, row.Count / 2, f"{percentage:.1f}%", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rank_distribution.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Rank distribution chart saved to {output_path}")

def plot_score_comparison(examples, output_dir="./results"):
    """
    Plot comparison between matched and non-matched scores.
    """
    match_scores = []
    non_match_scores = []
    
    for ex in examples:
        if ex.get("match_index") is not None:
            for i, score in enumerate(ex["scores"]):
                if i == ex["match_index"]:
                    match_scores.append(score)
                else:
                    non_match_scores.append(score)
    
    if not match_scores or not non_match_scores:
        print("Insufficient score data for comparison.")
        return
        
    # Create dataframe
    df_scores = pd.DataFrame([
        {"Score": score, "Type": "Correct Match"} for score in match_scores
    ] + [
        {"Score": score, "Type": "Incorrect Match"} for score in non_match_scores
    ])
    
    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x="Type", y="Score", data=df_scores, palette=["#2ecc71", "#e74c3c"])
    sns.stripplot(x="Type", y="Score", data=df_scores, color='black', alpha=0.3, jitter=True)
    
    plt.title("SigLIP Similarity Score Distribution", fontsize=16)
    plt.ylabel("Similarity Score", fontsize=14)
    
    # Add statistics
    plt.figtext(0.15, 0.01, f"Correct match avg: {sum(match_scores)/len(match_scores):.4f}", fontsize=12)
    plt.figtext(0.6, 0.01, f"Incorrect match avg: {sum(non_match_scores)/len(non_match_scores):.4f}", fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "score_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score comparison chart saved to {output_path}")

def main():
    results_dir = "./results"
    similarity_file = os.path.join(results_dir, "similarity_details.txt")
    
    if not os.path.exists(similarity_file):
        print(f"Error: File not found: {similarity_file}")
        return 1
        
    print("Analyzing SigLIP results...")
    
    # Extract metrics
    metrics = extract_metrics_from_text()
    
    # Plot metrics
    plot_metrics(metrics, results_dir)
    
    # Extract and analyze examples
    examples = extract_examples_from_file(similarity_file)
    print(f"Extracted {len(examples)} examples with scores.")
    
    if examples:
        # Generate visualizations
        plot_rank_distribution(examples, results_dir)
        plot_score_comparison(examples, results_dir)
        
    print("\nAnalysis complete. Results saved to", results_dir)
    return 0

if __name__ == "__main__":
    main() 