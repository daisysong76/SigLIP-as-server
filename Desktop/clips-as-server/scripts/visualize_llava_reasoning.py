#!/usr/bin/env python3
"""
Visualization tool for LLaVA reasoning results.
This script visualizes the results of the LLaVA reasoning pipeline,
showing the embeddings, reasoning outputs, and relationships between them.

Usage:
python scripts/visualize_llava_reasoning.py --input-file llava_reasoning.json --output-dir visualization_results
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from PIL import Image
import shutil
import base64
from io import BytesIO
import textwrap
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_reasoning_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load LLaVA reasoning data from a JSON or pickle file.
    
    Args:
        file_path: Path to JSON or pickle file with LLaVA reasoning data
        
    Returns:
        List of enriched metadata dictionaries
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(data)} entries from {file_path}")
    return data

def extract_embeddings(data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from LLaVA reasoning data.
    
    Args:
        data: Enriched metadata with embeddings
        
    Returns:
        Dictionary with extracted embeddings
    """
    embedding_types = {}
    first_entry = data[0]
    
    # Find all embedding types
    for key in first_entry:
        if key.endswith('_img_embedding') or key.endswith('_txt_embedding'):
            embedding_types[key] = []
    
    # Extract embeddings
    for entry in data:
        for embed_type in embedding_types:
            if embed_type in entry:
                if isinstance(entry[embed_type], list):
                    embedding_types[embed_type].append(np.array(entry[embed_type]))
                else:
                    embedding_types[embed_type].append(entry[embed_type])
    
    # Convert lists to numpy arrays
    for embed_type in embedding_types:
        embedding_types[embed_type] = np.array(embedding_types[embed_type])
    
    logger.info(f"Extracted {len(embedding_types)} embedding types")
    for embed_type, embeds in embedding_types.items():
        logger.info(f"  {embed_type}: {embeds.shape}")
    
    return embedding_types

def extract_reasoning_fields(data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract reasoning fields from LLaVA data.
    
    Args:
        data: Enriched metadata with reasoning outputs
        
    Returns:
        Dictionary with extracted reasoning fields
    """
    reasoning_fields = {}
    first_entry = data[0]
    
    # Find reasoning fields
    for key in first_entry:
        if (key.startswith('llava_') or key.startswith('qwen_')) and not key.endswith('_embedding'):
            reasoning_fields[key] = []
    
    # Extract reasoning outputs
    for entry in data:
        for field in reasoning_fields:
            if field in entry:
                reasoning_fields[field].append(entry[field])
            else:
                reasoning_fields[field].append("")
    
    logger.info(f"Extracted {len(reasoning_fields)} reasoning fields")
    return reasoning_fields

def visualize_embeddings_2d(embeddings: Dict[str, np.ndarray], 
                          output_dir: str,
                          method: str = 'tsne'):
    """
    Visualize embeddings in 2D using t-SNE or PCA.
    
    Args:
        embeddings: Dictionary with embeddings
        output_dir: Directory to save visualizations
        method: Dimensionality reduction method ('tsne' or 'pca')
    """
    logger.info(f"Visualizing embeddings in 2D using {method}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Colors for visualization
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Create combined visualization plot
    plt.figure(figsize=(15, 10))
    
    # Process each embedding type
    for i, (embed_type, embeds) in enumerate(embeddings.items()):
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        # Reduce dimensionality
        reduced_embeds = reducer.fit_transform(embeds)
        
        # Plot individual embedding type
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeds[:, 0], reduced_embeds[:, 1], 
                   c=colors[i % len(colors)], alpha=0.7, 
                   label=embed_type)
        
        # Add labels for each point
        for j, (x, y) in enumerate(reduced_embeds):
            plt.annotate(str(j), (x, y), fontsize=8, alpha=0.7)
        
        plt.title(f'2D {method.upper()} Projection - {embed_type}')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method}_{embed_type}.png'), dpi=300)
        plt.close()
        
        # Add to combined plot
        plt.figure(1)
        plt.scatter(reduced_embeds[:, 0], reduced_embeds[:, 1], 
                   c=colors[i % len(colors)], alpha=0.7, 
                   label=embed_type)
    
    # Save combined plot
    plt.figure(1)
    plt.title(f'Combined 2D {method.upper()} Projection of Embeddings')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method}_combined.png'), dpi=300)
    plt.close()
    
    logger.info(f"Saved 2D visualizations to {output_dir}")

def create_reasoning_heatmap(reasoning_fields: Dict[str, List[str]], 
                           output_dir: str):
    """
    Create heatmap of reasoning output lengths.
    
    Args:
        reasoning_fields: Dictionary with reasoning outputs
        output_dir: Directory to save visualizations
    """
    logger.info("Creating reasoning output length heatmap")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate lengths of reasoning outputs
    field_names = list(reasoning_fields.keys())
    num_samples = len(list(reasoning_fields.values())[0])
    
    # Create length matrix
    length_matrix = np.zeros((num_samples, len(field_names)))
    
    for i, field in enumerate(field_names):
        for j, output in enumerate(reasoning_fields[field]):
            length_matrix[j, i] = len(output) if not output.startswith("Error:") and output != "No image available for processing" else 0
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(length_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Output Length (chars)')
    plt.title('LLaVA Reasoning Output Lengths')
    plt.xlabel('Reasoning Type')
    plt.ylabel('Sample Index')
    plt.xticks(range(len(field_names)), field_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reasoning_length_heatmap.png'), dpi=300)
    plt.close()
    
    logger.info(f"Saved reasoning heatmap to {output_dir}")

def create_html_report(data: List[Dict[str, Any]], 
                     embeddings: Dict[str, np.ndarray],
                     reasoning_fields: Dict[str, List[str]],
                     output_dir: str):
    """
    Create interactive HTML report with visualizations.
    
    Args:
        data: Original LLaVA reasoning data
        embeddings: Dictionary with embeddings
        reasoning_fields: Dictionary with reasoning outputs
        output_dir: Directory to save HTML report
    """
    logger.info("Creating interactive HTML report")
    vis_dir = os.path.join(output_dir, 'vis_assets')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create plots directory for individual sample visualizations
    plots_dir = os.path.join(vis_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Copy images if available
    images_dir = os.path.join(vis_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # List to store image availability for each sample
    image_paths = []
    
    # Process images
    for i, entry in enumerate(data):
        if 'image_path' in entry and entry['image_path']:
            img_path = entry['image_path']
            if os.path.exists(img_path):
                # Copy image to images directory
                dest_path = os.path.join(images_dir, f"sample_{i}.jpg")
                shutil.copy(img_path, dest_path)
                image_paths.append(f"vis_assets/images/sample_{i}.jpg")
            else:
                image_paths.append("")
        else:
            image_paths.append("")
    
    # HTML template
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLaVA Reasoning Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3 {
                color: #333;
            }
            .sample-container {
                margin-bottom: 30px;
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 5px;
            }
            .sample-header {
                display: flex;
                justify-content: space-between;
                background-color: #f5f5f5;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 3px;
            }
            .sample-content {
                display: flex;
                flex-wrap: wrap;
            }
            .image-section {
                flex: 0 0 300px;
                margin-right: 20px;
            }
            .reasoning-section {
                flex: 1;
                min-width: 300px;
            }
            .embeddings-section {
                flex: 1;
                margin-top: 15px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            table, th, td {
                border: 1px solid #ddd;
            }
            th, td {
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .reasoning-output {
                white-space: pre-wrap;
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 3px;
                max-height: 200px;
                overflow-y: auto;
                margin-bottom: 10px;
            }
            .overview-section {
                margin-bottom: 30px;
            }
            .overview-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .overview-item {
                padding: 15px;
                background-color: #f5f5f5;
                border-radius: 3px;
            }
            .collapsible {
                background-color: #f1f1f1;
                color: #444;
                cursor: pointer;
                padding: 10px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 15px;
                border-radius: 3px;
            }
            .active, .collapsible:hover {
                background-color: #e1e1e1;
            }
            .content {
                display: none;
                padding: 0 18px;
                overflow: hidden;
                background-color: #fff;
            }
            .plot-section {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }
            .plot-item {
                flex: 0 0 calc(50% - 20px);
                max-width: calc(50% - 20px);
                margin-bottom: 20px;
            }
            .plot-item img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            .error {
                color: red;
                font-style: italic;
            }
            .nav-tabs {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                display: flex;
                margin-bottom: 20px;
            }
            .nav-tabs button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 16px;
                flex: 1;
            }
            .nav-tabs button:hover {
                background-color: #ddd;
            }
            .nav-tabs button.active {
                background-color: #ccc;
            }
            .tab-content {
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
            }
        </style>
    </head>
    <body>
        <h1>LLaVA Reasoning Analysis Report</h1>
        <p>Generated on: {timestamp}</p>
        
        <div class="nav-tabs">
            <button class="tablink active" onclick="openTab(event, 'overview')">Overview</button>
            <button class="tablink" onclick="openTab(event, 'embeddings')">Embeddings</button>
            <button class="tablink" onclick="openTab(event, 'samples')">Individual Samples</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content" style="display: block;">
            <h2>Overview</h2>
            <div class="overview-section">
                <h3>Dataset Summary</h3>
                <div class="overview-grid">
                    <div class="overview-item">
                        <h4>Dataset Size</h4>
                        <p>{num_samples} samples</p>
                    </div>
                    <div class="overview-item">
                        <h4>Embedding Types</h4>
                        <p>{num_embedding_types} types</p>
                        <ul>
                            {embedding_types_list}
                        </ul>
                    </div>
                    <div class="overview-item">
                        <h4>Reasoning Fields</h4>
                        <p>{num_reasoning_fields} fields</p>
                        <ul>
                            {reasoning_fields_list}
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="overview-section">
                <h3>Reasoning Output Length Analysis</h3>
                <div class="plot-item">
                    <img src="vis_assets/reasoning_length_heatmap.png" alt="Reasoning Length Heatmap">
                </div>
            </div>
        </div>
        
        <!-- Embeddings Tab -->
        <div id="embeddings" class="tab-content">
            <h2>Embedding Visualizations</h2>
            
            <div class="plot-section">
                <div class="plot-item">
                    <h3>t-SNE Projection (Combined)</h3>
                    <img src="vis_assets/tsne_combined.png" alt="t-SNE Combined">
                </div>
                <div class="plot-item">
                    <h3>PCA Projection (Combined)</h3>
                    <img src="vis_assets/pca_combined.png" alt="PCA Combined">
                </div>
            </div>
            
            <h3>Individual Embedding Type Projections</h3>
            <div class="plot-section">
                {individual_embedding_plots}
            </div>
        </div>
        
        <!-- Samples Tab -->
        <div id="samples" class="tab-content">
            <h2>Individual Sample Analysis</h2>
            
            {samples_content}
        </div>
        
        <script>
            // Collapsible sections
            var coll = document.getElementsByClassName("collapsible");
            for (var i = 0; i < coll.length; i++) {
                coll[i].addEventListener("click", function() {
                    this.classList.toggle("active");
                    var content = this.nextElementSibling;
                    if (content.style.display === "block") {
                        content.style.display = "none";
                    } else {
                        content.style.display = "block";
                    }
                });
            }
            
            // Tab navigation
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablink");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
        </script>
    </body>
    </html>
    """
    
    # Fill in the sample content
    samples_content = ""
    
    for i, entry in enumerate(data):
        sample_content = """
        <div class="sample-container">
            <div class="sample-header">
                <h3>Sample {sample_idx}</h3>
                <div>ID: {sample_id}</div>
            </div>
            <div class="sample-content">
                <div class="image-section">
                    <h4>Image</h4>
                    {image_content}
                    <p>Caption: {caption}</p>
                </div>
                <div class="reasoning-section">
                    <h4>LLaVA Reasoning Outputs</h4>
                    {reasoning_outputs}
                </div>
            </div>
            <button class="collapsible">Embedding Statistics</button>
            <div class="content embeddings-section">
                <h4>Embedding Values</h4>
                {embedding_stats}
            </div>
        </div>
        """
        
        # Handle image
        if image_paths[i]:
            image_content = f'<img src="{image_paths[i]}" alt="Sample {i}" style="max-width: 100%; max-height: 300px;">'
        else:
            image_content = '<p>No image available</p>'
        
        # Format reasoning outputs
        reasoning_outputs = ""
        for field in reasoning_fields:
            value = reasoning_fields[field][i]
            
            # Check for errors
            if value.startswith("Error:") or value == "No image available for processing":
                formatted_value = f'<div class="error">{value}</div>'
            else:
                # Wrap text for better readability
                wrapped_value = textwrap.fill(value, width=80)
                formatted_value = f'<div class="reasoning-output">{wrapped_value}</div>'
            
            reasoning_outputs += f"""
            <div>
                <h5>{field}</h5>
                {formatted_value}
            </div>
            """
        
        # Format embedding statistics
        embedding_stats = "<table>"
        embedding_stats += "<tr><th>Embedding Type</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr>"
        
        for embed_type in embeddings:
            if i < len(embeddings[embed_type]):
                embed = embeddings[embed_type][i]
                embedding_stats += f"""
                <tr>
                    <td>{embed_type}</td>
                    <td>{np.min(embed):.4f}</td>
                    <td>{np.max(embed):.4f}</td>
                    <td>{np.mean(embed):.4f}</td>
                    <td>{np.std(embed):.4f}</td>
                </tr>
                """
        
        embedding_stats += "</table>"
        
        # Fill in the template
        sample_id = entry.get("id", f"sample_{i}")
        caption = entry.get("caption", "No caption available")
        
        samples_content += sample_content.format(
            sample_idx=i,
            sample_id=sample_id,
            image_content=image_content,
            caption=caption,
            reasoning_outputs=reasoning_outputs,
            embedding_stats=embedding_stats
        )
    
    # Format embedding type list
    embedding_types_list = ""
    for embed_type in embeddings:
        shape = embeddings[embed_type].shape
        embedding_types_list += f"<li>{embed_type} (dimension: {shape[1]})</li>"
    
    # Format reasoning fields list
    reasoning_fields_list = ""
    for field in reasoning_fields:
        reasoning_fields_list += f"<li>{field}</li>"
    
    # Format individual embedding plots
    individual_embedding_plots = ""
    for embed_type in embeddings:
        individual_embedding_plots += f"""
        <div class="plot-item">
            <h4>{embed_type} (t-SNE)</h4>
            <img src="vis_assets/tsne_{embed_type}.png" alt="t-SNE {embed_type}">
        </div>
        <div class="plot-item">
            <h4>{embed_type} (PCA)</h4>
            <img src="vis_assets/pca_{embed_type}.png" alt="PCA {embed_type}">
        </div>
        """
    
    # Fill in the HTML template
    html_content = html_content.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_samples=len(data),
        num_embedding_types=len(embeddings),
        embedding_types_list=embedding_types_list,
        num_reasoning_fields=len(reasoning_fields),
        reasoning_fields_list=reasoning_fields_list,
        individual_embedding_plots=individual_embedding_plots,
        samples_content=samples_content
    )
    
    # Write the HTML file
    with open(os.path.join(output_dir, 'llava_reasoning_report.html'), 'w') as f:
        f.write(html_content)
    
    logger.info(f"Saved interactive HTML report to {output_dir}/llava_reasoning_report.html")

def main():
    parser = argparse.ArgumentParser(description="Visualize LLaVA reasoning results")
    parser.add_argument("--input-file", type=str, required=True,
                      help="Path to JSON or pickle file with LLaVA reasoning data")
    parser.add_argument("--output-dir", type=str, default="llava_visualization",
                      help="Directory to save visualizations")
    parser.add_argument("--skip-tsne", action="store_true",
                      help="Skip t-SNE visualization (faster processing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_reasoning_data(args.input_file)
    
    # Extract embeddings
    embeddings = extract_embeddings(data)
    
    # Extract reasoning fields
    reasoning_fields = extract_reasoning_fields(data)
    
    # Visualize embeddings
    if not args.skip_tsne:
        visualize_embeddings_2d(embeddings, os.path.join(args.output_dir, 'vis_assets'), method='tsne')
    visualize_embeddings_2d(embeddings, os.path.join(args.output_dir, 'vis_assets'), method='pca')
    
    # Create reasoning heatmap
    create_reasoning_heatmap(reasoning_fields, os.path.join(args.output_dir, 'vis_assets'))
    
    # Create HTML report
    create_html_report(data, embeddings, reasoning_fields, args.output_dir)
    
    logger.info(f"Visualization complete! Open {args.output_dir}/llava_reasoning_report.html in a browser to view the report.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 