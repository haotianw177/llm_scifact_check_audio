import os
import random
import openai
import json


# The Abstract Dialogue Generation Framework
DIALOGUE_FRAMEWORK = {
    "dynamics": [
        "Originator vs. Interpreter",
        "Practitioner vs. Decision-Maker",
        "Advocate vs. Skeptic",
        "Beneficiary vs. Authority",
        "Peer vs. Peer",
    ],
    "arena": [
        "The Persuasive Arena (goal: to win over an audience)",
        "The Deliberative Arena (goal: to make a strategic choice)",
        "The Didactic Arena (goal: to teach or learn)",
        "The Exploratory Arena (goal: to brainstorm and understand)",
        "The Personal Arena (goal: to resolve a personal, high-stakes matter)",
    ],
    "trigger": [
        "An Opportunity (a potential gain)",
        "A Threat (a potential risk or cost)",
        "A Point of Confusion (a misunderstanding or contradiction)",
        "An External Mandate (a new law, policy, or trend)",
        "A Recent Discovery (new, urgent data)",
    ],
    "objective": [
        "To Reach a Decision (choose or reject an action)",
        "To Achieve Alignment (find a shared understanding)",
        "To Expose a Flaw or Truth (validate or weaken the claim)",
        "To Inform and Empower (enable an independent decision)",
        "To Define the Disagreement (map out points of conflict)",
    ]
}

def generate_dialogue_script(scientific_claim: str, model: str = "gpt-4o") -> str | None:
    """
    Generates a dialogue script based on a scientific claim using the abstract framework.

    Args:
        scientific_claim: The scientific claim to be discussed.
        model: The model to use for generation (e.g., "gpt-4o", "gpt-3.5-turbo").

    Returns:
        A string containing the generated dialogue script, or None if an error occurs.
    """
    if not openai.api_key:
        print("API key is not configured. Please set the OPENAI_API_KEY environment variable.")
        return None

    # 1. Randomly select parameters from the framework
    dynamics = random.choice(DIALOGUE_FRAMEWORK["dynamics"])
    arena = random.choice(DIALOGUE_FRAMEWORK["arena"])
    trigger = random.choice(DIALOGUE_FRAMEWORK["trigger"])
    objective = random.choice(DIALOGUE_FRAMEWORK["objective"])

    # 2. Construct the detailed prompt for the AI
    system_prompt = "You are a creative scriptwriter. Your task is to generate a short, realistic, multi-turn dialogue of approximately 300 words based on the provided parameters. The dialogue should feel authentic to the characters and situation."
    
    user_prompt = f"""
    Please generate a dialogue script with the following parameters:

    - **Scientific Claim:** "{scientific_claim}"

    - **Participant Dynamics:** {dynamics}
      (This defines the relationship and power balance between the speakers.)

    - **Contextual Arena:** {arena}
      (This defines the 'rules' and purpose of the conversation.)

    - **Interaction Trigger:** {trigger}
      (This is the spark that starts the conversation.)

    - **Dialogue Objective:** {objective}
      (This defines the desired end-state or goal of the conversation.)

    Based on these parameters, create a compelling and natural-sounding dialogue.
    """

    # 3. Call the API
    try:
        client = openai.OpenAI(api_key=openai.api_key) # Initialize client
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # A bit of creativity
            max_tokens=450 # Enough for a ~300 word dialogue + buffer
        )
        script = response.choices[0].message.content.strip()
        return script
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None

# --- Main execution block to demonstrate usage ---
if __name__ == "__main__":
    records = []
    filepath = '/Users/akritidhasmana/scifact-open/data/claims.jsonl'

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                # Strip whitespace, as it can interfere with JSON parsing
                stripped_line = line.strip()
                if not stripped_line:
                    continue # Skip empty lines

                try:
                    # Parse the JSON object from the line
                    record = json.loads(stripped_line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_number}: Failed to parse JSON.")
                    print(f"Error details: {e}")
                    # Optionally, print the problematic line: print(f"Problematic line: {stripped_line[:100]}...")
                    continue

    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        records =  []

    print(f"RECORDS: \n {records}")

    for record in records[0:5]:
        sample_claim = record['claim']


        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if openai.api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
        except ValueError as e:
            print(e)
            # You can also hardcode your key here for quick testing, but it's not recommended:
            openai.api_key = "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"

        # You can load your scientific claims one at a time here.
        # For this example, we'll use a sample claim.
        # sample_claim = "A newly developed gene-editing therapy can safely reverse hereditary blindness with a single treatment."
        
        print(f"Generating dialogue for claim: '{sample_claim}'\n")

        
        generated_script = generate_dialogue_script(sample_claim)
        
        if generated_script:
            print("--- Generated Dialogue Script ---")
            print(generated_script)
            print("-------------------------------")
        else:
            print("Failed to generate dialogue script.")