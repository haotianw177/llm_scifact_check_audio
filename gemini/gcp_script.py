import os
import re
import json
import random
from pathlib import Path
from google.cloud import texttospeech

# --- 1. CONFIGURATION ---

# Initialize Google TTS Client
try:
    client = texttospeech.TextToSpeechClient()
except Exception as e:
    print(f"Error initializing Google Client: {e}")
    print("Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
    exit(1)

GOOGLE_VOICES = [
    "en-US-Neural2-A", "en-US-Neural2-C", "en-US-Neural2-D", 
    "en-US-Neural2-E", "en-US-Neural2-F", "en-US-Neural2-G", 
    "en-US-Neural2-H", "en-US-Neural2-I", "en-US-Neural2-J",
]

# Pause duration (SSML break)
PAUSE_DURATION_SEC = "1s" 

# Directories to search for scripts
SCRIPT_SEARCH_DIRS = [
    "../dialogue_scripts/batch1",
    "../dialogue_scripts/batch2"
]

# Single output directory for all audio
OUTPUT_DIR = "./gemini_audio"
CLAIMS_FILE = "./claims_train.jsonl"
TEMP_DIR = "./temp_audio_segments"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# --- 2. CORE FUNCTIONS ---

def assign_voices(characters):
    """Randomly assigns a Google voice to each character."""
    char_voices_mapping = {}
    available_voices = list(GOOGLE_VOICES) 
    random.shuffle(available_voices)
    
    for char in characters:
        if not available_voices:
            available_voices = list(GOOGLE_VOICES)
            random.shuffle(available_voices)
        char_voices_mapping[char] = available_voices.pop()
    
    return char_voices_mapping

def parse_dialogue(script_lines: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Parses the dialogue script into a list of (character, line) tuples."""
    parsed_lines = []
    characters = []
    pattern = re.compile(r"^(?P<character>[\w\s\.\-]+):\s*(?P<line>.*)$")

    for line in script_lines:
        line = line.strip()
        if line.startswith("```"): continue 
            
        match = pattern.match(line)
        if match:
            char_name = match.group("character").strip()
            dialogue_text = match.group("line").strip()
            
            if "Scientific claim" in char_name or "Contextual Arena" in char_name:
                continue

            if char_name and dialogue_text:
                if char_name not in characters:
                    characters.append(char_name)
                parsed_lines.append((char_name, dialogue_text))

    return characters, parsed_lines

def generate_audio_for_line(text: str, voice_name: str, output_path: Path):
    """Calls Google TTS API for a single line."""
    print(f"   -> Generating ({voice_name}): '{text[:30]}...'")
    
    input_text = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    try:
        response = client.synthesize_speech(
            input=input_text, voice=voice_params, audio_config=audio_config
        )
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        return output_path
    except Exception as e:
        print(f"   [!] Error generating TTS: {e}")
        return None

def generate_silence_file(output_path: Path):
    """Generates a silent MP3 using SSML."""
    if output_path.exists():
        return output_path

    print(f"   -> Generating silence clip...")
    ssml_text = f'<speak><break time="{PAUSE_DURATION_SEC}"/></speak>'
    input_text = texttospeech.SynthesisInput(ssml=ssml_text)
    voice_params = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Neural2-C")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=input_text, voice=voice_params, audio_config=audio_config
    )
    
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    return output_path

def combine_audio_files_binary(audio_files: list[Path], silence_file: Path, output_file: str):
    """Combines MP3 files by simple binary concatenation."""
    print(f"   -> Stitching files to: {output_file}")
    
    with open(output_file, 'wb') as outfile:
        silence_bytes = b""
        if silence_file and silence_file.exists():
            with open(silence_file, 'rb') as sf:
                silence_bytes = sf.read()

        for i, f_path in enumerate(audio_files):
            if f_path.exists():
                with open(f_path, 'rb') as infile:
                    outfile.write(infile.read())
                
                if i < len(audio_files) - 1:
                    outfile.write(silence_bytes)

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    records = []
    
    # 1. Load JSONL
    if os.path.exists(CLAIMS_FILE):
        with open(CLAIMS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
    else:
        print(f"Warning: {CLAIMS_FILE} not found.")

    # 2. Generate the Silence Clip Once
    silence_path = Path(TEMP_DIR) / "silence_gap.mp3"
    generate_silence_file(silence_path)

    # 3. Process Records
    for record in records:
        claim_id = record.get('id')
        if not claim_id: continue

        # --- PATH DISCOVERY LOGIC ---
        script_path = None
        for search_dir in SCRIPT_SEARCH_DIRS:
            potential_path = os.path.join(search_dir, f"{claim_id}.txt")
            if os.path.exists(potential_path):
                script_path = potential_path
                break
        
        # Skip if script not found in either batch folder
        if not script_path:
            # Optional: print(f"Skipping {claim_id} (not found in batch1 or batch2)")
            continue

        output_mp3 = os.path.join(OUTPUT_DIR, f"{claim_id}.mp3")
        print(f"\nProcessing ID: {claim_id} (Found in {os.path.dirname(script_path)})")

        with open(script_path, 'r', encoding='utf-8') as f:
            script_lines = f.readlines()

        characters, dialogue_lines = parse_dialogue(script_lines)
        if not dialogue_lines:
            continue

        char_voices = assign_voices(characters)
        generated_segments = []

        # Generate Audio Lines
        for i, (char, line) in enumerate(dialogue_lines):
            voice = char_voices.get(char)
            if voice:
                temp_path = Path(TEMP_DIR) / f"{claim_id}_{i}.mp3"
                if generate_audio_for_line(line, voice, temp_path):
                    generated_segments.append(temp_path)

        # Stitch Binary Files
        if generated_segments:
            combine_audio_files_binary(generated_segments, silence_path, output_mp3)
            
            # Cleanup Line Files
            for p in generated_segments:
                try: os.remove(p)
                except: pass

    # Final cleanup
    if silence_path.exists():
        os.remove(silence_path)
    try: os.rmdir(TEMP_DIR)
    except: pass
    
    print("\n✅ Batch Processing Complete.")