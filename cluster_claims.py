#!/ariamis

# Import the json library so we can read the .jsonl data file
import json
# Import pandas to create a table (DataFrame) that is easy to manage and save
import pandas as pd
# Import the AI model that turns sentences into lists of numbers (vectors)
from sentence_transformers import SentenceTransformer
# Import the clustering tool that groups similar numbers together
from sklearn.cluster import KMeans




# --- STEP 1: LOAD THE DATA ---
# Load the claims_train file (usually a .jsonl file in the SciFact dataset) and remove any claim that doesn't have evidence attached.
# In the SciFact dataset structure, the evidence field is a dictionary. If it is empty {} or null, there is no evidence.

# Create an empty list to hold our claims
data = []

# Open the file 'claims_train.jsonl' in 'read' mode ('r')
# Note: Ensure this file is in the same folder as this script!
with open('claims_train.jsonl', 'r') as file:
    # Loop through every line in the file
    for line in file:
        # distinct JSON object on each line is converted to a Python dictionary and added to our list
        data.append(json.loads(line))

# --- STEP 2: FILTER THE DATA ---

# Create a new list called 'clean_data'
# We only keep an item if 'evidence' is NOT empty (meaning it has valid proof)
# Go through every item in our data list. Check if the evidence field exists and has stuff inside it. If it does, keep that item and put it into a new list called clean_data."
clean_data = [item for item in data if item.get('evidence')]

# Turn this list into a pandas DataFrame (like an Excel sheet in code)
df = pd.DataFrame(clean_data)

# --- STEP 3: TURN TEXT INTO NUMBERS (EMBEDDINGS) ---

# Load a small, fast AI model specifically made for comparing sentences
model = SentenceTransformer('all-MiniLM-L6-v2')

# Take the 'claim' column from our data, and encode it into numbers
# This 'embeddings' variable now holds the mathematical meaning of your sentences
embeddings = model.encode(df['claim'].tolist())

# --- STEP 4: GROUP THE TOPICS ---

# Set up the clustering machine to find exactly 10 different topics
kmeans = KMeans(n_clusters=10, random_state=42)

# Tell the machine to look at our 'embeddings' (numbers) and assign a group ID (0-9) to each one
# We save this group ID into a new column called 'topic_id'
df['topic_id'] = kmeans.fit_predict(embeddings)

# # --- STEP 5: SAVE THE RESULTS ---

# # Save our organized data to a CSV file so you can open it in Excel
# # 'index=False' means we don't save the row numbers (0, 1, 2...)
# df.to_csv('cluster_claims.csv', index=False)

# # Print a success message so you know it worked
# print("Success! Open 'cluster_claims.csv' to see your grouped topics.")

# # ==========================================
# # --- STEP 6: VISUALIZE WITH T-SNE ---
# # ==========================================

# print("Generating graph... this might take a moment...")

# # Import the visualization tools
# # TSNE is the tool that squashes 384-dimensional numbers down to 2 dimensions (X and Y)
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # Initialize t-SNE
# # n_components=2 means we want 2D (flat image)
# # perplexity=30 is standard for balancing local vs global clusters
# tsne_model = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')

# # Fit the model to your existing 'embeddings' variable
# # visual_data will be a list of [x, y] coordinates
# visual_data = tsne_model.fit_transform(embeddings)

# # Get the X and Y coordinates separately
# x_coords = visual_data[:, 0]
# y_coords = visual_data[:, 1]

# # Get the colors (based on the topic_id we calculated earlier)
# colors = df['topic_id']

# # Create the plot
# plt.figure(figsize=(12, 10)) # Make the image big (12x10 inches)

# # Scatter plot:
# # c=colors -> color the dots based on topic ID
# # cmap='tab10' -> use a color palette with 10 distinct colors
# # alpha=0.6 -> make dots slightly transparent so you can see overlaps
# scatter = plt.scatter(x_coords, y_coords, c=colors, cmap='tab10', alpha=0.6)

# # Add a legend/colorbar so you know which color is which ID
# plt.colorbar(scatter, label='Topic ID')

# # Add titles and labels
# plt.title("Visualization of Scientific Claim Clusters (t-SNE)", fontsize=16)
# plt.xlabel("t-SNE dimension 1")
# plt.ylabel("t-SNE dimension 2")

# # Save the picture to your folder
# plt.savefig("cluster_visualization.png")

# print("Graph saved as 'cluster_visualization.png'!")

# # ==========================================
# # --- STEP 7: VISUALIZE WITH UMAP ---
# # ==========================================

# print("Generating UMAP graph...")

# # Import UMAP
# import umap

# # Initialize UMAP
# # n_neighbors=15: Controls how UMAP balances local vs global structure
# # min_dist=0.1: Controls how tightly UMAP packs points together
# umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

# # Fit the model to your embeddings
# umap_data = umap_model.fit_transform(embeddings)

# # Get X and Y coordinates
# x_coords_umap = umap_data[:, 0]
# y_coords_umap = umap_data[:, 1]

# # Create the plot
# plt.figure(figsize=(12, 10))

# # Scatter plot
# scatter = plt.scatter(x_coords_umap, y_coords_umap, c=df['topic_id'], cmap='tab10', alpha=0.6)

# # Add legend and labels
# plt.colorbar(scatter, label='Topic ID')
# plt.title("Visualization of Scientific Claim Clusters (UMAP)", fontsize=16)
# plt.xlabel("UMAP dimension 1")
# plt.ylabel("UMAP dimension 2")

# # Save the picture
# plt.savefig("cluster_visualization_umap.png")

# print("Graph saved as 'cluster_visualization_umap.png'!")