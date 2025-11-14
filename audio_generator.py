import os
import re
import openai
import json
import random
from openai import OpenAI
from pydub import AudioSegment
from pathlib import Path
# You can use python-dotenv to securely load your API key
# from dotenv import load_dotenv

# --- 1. CONFIGURATION ---

try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
except ValueError as e:
    print(e)
    # You can also hardcode your key here for quick testing, but it's not recommended:
    openai.api_key = "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"

client = OpenAI(api_key=openai.api_key)


CHARACTER_VOICES = ["alloy", "ash", "coral", "echo", "fable",
                    "nova","onyx", "sage", "shimmer"]
# Define the length of the pause between lines (in milliseconds)
PAUSE_DURATION_MS = 1000  # 1 second

# Dialogue script extracted and cleaned from uploaded:script_dialoge.txt
# Note: For simplicity and reliability in a script, it's best to format the raw text
# with clear speaker-line separators.

# --- 2. CORE FUNCTIONS ---
def assign_voices(characters):
    # Extract Characters 
    # Define the voice for each character
    char_voices_mapping = {}
    voices =  random.choices(CHARACTER_VOICES, k = len(characters))
    for char in characters:
        voice = voices.pop()
        char_voices_mapping[char] = voice
    return char_voices_mapping

def parse_dialogue(script: list[str]) -> list[tuple[str, str]]:
    """Parses the dialogue script into a list of (character, line) tuples."""
    parsed_lines = []
    # Regex to find "Character Name:" followed by the line
    pattern = re.compile(r"^(?P<character>[\*\w\s\.]+):\s*(?P<line>.*)$", re.MULTILINE)
    metadata = script[:7]
    characters = []
    for line in script[7:]:
        for match in pattern.finditer(line.strip()):
            character = match.group("character").strip()
            line = match.group("line").strip()
            if character and line:
                characters.append(character)
                parsed_lines.append((character, line))
    return characters, parsed_lines

def generate_audio_for_line(text: str, voice: str, output_path: Path):
    """Calls the OpenAI TTS API to generate audio and saves it to a file."""
    print(f"Generating audio for voice '{voice}': '{text[:50]}...'")
    try:
        response = client.audio.speech.create(
            model="tts-1", # or "tts-1-hd" for higher quality
            voice=voice,
            input=text
        )
        response.stream_to_file(output_path)
        print(f"Successfully saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during TTS generation: {e}")
        return None

def combine_audio_segments(audio_files: list[Path], pause_ms: int, output_file: str):
    """Combines a list of audio files with a pause in between each one."""
    
    # Create an AudioSegment object for the pause
    # Use the same sample rate (24kHz) as the TTS model for compatibility
    sample_rate = 24000
    pause_segment = AudioSegment.silent(duration=pause_ms)
    
    # Start with an empty AudioSegment
    combined_audio = AudioSegment.empty()
    
    # Concatenate all the line segments with the pause segment
    for file_path in audio_files:
        if file_path.exists():
            line_segment = AudioSegment.from_file(file_path, format="mp3")
            combined_audio += line_segment
            # Add a pause after every line, except for the very last one
            if file_path != audio_files[-1]:
                combined_audio += pause_segment
    
    # Export the final combined audio file
    print(f"\nExporting final audio to {output_file}...")
    combined_audio.export(output_file, format="mp3", bitrate="192k")
    print(f"Finished! Your final dialogue file is saved as '{output_file}'")

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    # Load the claims
    records = []
    filepath = './claims_train.jsonl'
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

    for record in records:
        if not record['evidence']:
            continue
        claim = record['claim']
        claim_id = record['id']
        with open(f"dialogue_scripts/mult/batch2/{claim_id}.txt", 'r', encoding='utf-8') as f:
                dialogue = f.readlines()
    
        # 1. Parse the dialogue
        characters, dialogue_lines = parse_dialogue(dialogue)
        character_voices = assign_voices(characters)
        output_filename = f"audio_files/batch2/{claim_id}.mp3"
        
        if not dialogue_lines:
            print("Error: Could not parse any dialogue lines from the script.")
        else:
            # 2. Generate audio for each line
            generated_files = []
            for i, (character, line) in enumerate(dialogue_lines):
                voice = character_voices.get(character)
                if voice:
                    output_path = Path(f"temp_line_{i}_{character.replace(' ', '_')}.mp3")
                    
                    # Generate audio and get the saved file path
                    file_path = generate_audio_for_line(line, voice, output_path)
                    
                    if file_path:
                        generated_files.append(file_path)
                else:
                    print(f"Warning: No voice defined for character '{character}'. Skipping line.")

            # 3. Combine all segments
            if generated_files:
                combine_audio_segments(generated_files, PAUSE_DURATION_MS, output_filename)
            
                # 4. Clean up temporary files
                print("\nCleaning up temporary files...")
                for f in generated_files:
                    f.unlink()
                print("Cleanup complete.")
            else:
                print("No audio files were generated to combine.")