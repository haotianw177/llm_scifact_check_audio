import pandas as pd

# Load the file you just made
df = pd.read_csv('cluster_claims.csv')

# Loop through each topic number (0 to 9)
for topic_number in range(10):
    print(f"\n--- TOPIC {topic_number} ---")
    
    # Get all rows that belong to this topic
    subset = df[df['topic_id'] == topic_number]
    
    # Print the first 3 claims in this topic
    examples = subset['claim'].head(3).tolist()
    for claim in examples:
        print(f"- {claim}")