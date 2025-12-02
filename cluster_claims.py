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

# --- STEP 5: SAVE THE RESULTS ---

# Save our organized data to a CSV file so you can open it in Excel
# 'index=False' means we don't save the row numbers (0, 1, 2...)
df.to_csv('cluster_claims.csv', index=False)

# Print a success message so you know it worked
print("Success! Open 'sorted_claims.csv' to see your grouped topics.")

