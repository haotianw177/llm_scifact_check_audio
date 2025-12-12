import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import umap
import numpy as np

# --- STEP 1: LOAD DATA ---
print("Loading cluster_claims.csv...")
df = pd.read_csv('cluster_claims.csv')

# --- STEP 2: RE-GENERATE EMBEDDINGS ---
print("Loading AI Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings from claim text...")
embeddings = model.encode(df['claim'].tolist())

# --- STEP 3: OPTIMIZED NUMBER VISUALIZATION ---
def plot_numbered_clusters(coords, title, filename):
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Get the colormap
    cmap = plt.get_cmap('tab10', 10)
    
    # OPTIMIZATION: Instead of writing text for every point, we loop 10 times.
    # We create a standard scatter plot, but we set the SHAPE (marker) to be the number itself.
    # using f'${i}$' tells matplotlib to render the number like a math symbol.
    for topic_id in range(10):
        # Find all points that belong to this topic
        indices = df['topic_id'] == topic_id
        
        # Plot them all at once using the number as the marker
        plt.scatter(
            x[indices], 
            y[indices], 
            marker=f"${topic_id}$", # <--- This is the magic trick
            s=70,                   # Size needs to be slightly larger for text markers
            color=cmap(topic_id),   # Color matches the ID
            alpha=0.6               # Transparency
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Save tightly
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")

# --- STEP 4: RUN T-SNE ---
print("Running t-SNE analysis...")
tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
tsne_data = tsne.fit_transform(embeddings)

plot_numbered_clusters(tsne_data, "t-SNE Clusters (Numbered)", "cluster_viz_tsne_numbers.png")

# --- STEP 5: RUN UMAP ---
print("Running UMAP analysis...")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_data = umap_model.fit_transform(embeddings)

plot_numbered_clusters(umap_data, "UMAP Clusters (Numbered)", "cluster_viz_umap_numbers.png")

print("Done! Check your folder for the new images.")