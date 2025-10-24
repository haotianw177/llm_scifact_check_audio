import os
import re
from openai import OpenAI
from pydub import AudioSegment
from pydub.generators import silence
from pathlib import Path
# You can use python-dotenv to securely load your API key
# from dotenv import load_dotenv

# --- 1. CONFIGURATION ---

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# OR set the key directly from environment variable
client = OpenAI()

# Define the voice for each character
CHARACTER_VOICES = {
    "Dr. Carter": "nova",  # Female voice
    "Dr. Patel": "onyx"   # Male voice
}

# Define the length of the pause between lines (in milliseconds)
PAUSE_DURATION_MS = 1000  # 1 second
OUTPUT_FILENAME = "combined_dialogue.mp3"

# Dialogue script extracted and cleaned from uploaded:script_dialoge.txt
# Note: For simplicity and reliability in a script, it's best to format the raw text
# with clear speaker-line separators.
DIALOGUE_SCRIPT = """
Dr. Carter: Raj, have you seen the latest report from the WHO? It states that 10-20% of people with severe mental disorders in low and middle-income countries receive no treatment at all.
Dr. Patel: Yes, I read it this morning. It's alarming, but I'm not entirely surprised by the numbers. The infrastructure in these regions is often lacking.
Dr. Carter: True, but what's shocking is the magnitude. We're talking about millions of people left without any form of mental health support.
Dr. Patel: Right, but let's dig deeper. The report says "receive no treatment." Does it account for traditional or community-based interventions that might not be classified as formal treatment?
Dr. Carter: That's a valid point. It seems like the focus was primarily on western-style psychiatric services. But even with community-based approaches, the coverage is still patchy.
Dr. Patel: Exactly. We need to be careful with the terminology. The absence of formal treatment doesn’t always mean the absence of care. However, it's clear that accessibility and quality are major concerns.
Dr. Carter: So, you’re suggesting that the claim isn’t entirely comprehensive?
Dr. Patel: Not quite. I'm suggesting that while the claim is likely accurate, it doesn't capture the full picture. We need more nuanced data that includes informal care patterns.
Dr. Carter: I agree. So, our next step should be advocating for more inclusive research methods, perhaps?
Dr. Patel: Definitely. And perhaps we can propose a mixed-methods study to explore both formal and informal care networks. This could expose more of the truth about mental health care gaps.
Dr. Carter: That's a solid plan. Let’s draft a proposal for that. We need to ensure the data reflects the reality on the ground.
Dr. Patel: Absolutely. It's time we bridge the gap between policy and practice.
"""

# --- 2. CORE FUNCTIONS ---

def parse_dialogue(script: str) -> list[tuple[str, str]]:
    """Parses the dialogue script into a list of (character, line) tuples."""
    parsed_lines = []
    # Regex to find "Character Name:" followed by the line
    pattern = re.compile(r"^(?P<character>[A-Za-z\s\.]+):\s*(?P<line>.*)$", re.MULTILINE)
    
    for match in pattern.finditer(script.strip()):
        character = match.group("character").strip()
        line = match.group("line").strip()
        if character and line:
            parsed_lines.append((character, line))
    return parsed_lines

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
    pause_segment = silence(duration=pause_ms, sample_rate=sample_rate)
    
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
    
    # 1. Parse the dialogue
    dialogue_lines = parse_dialogue(DIALOGUE_SCRIPT)
    
    if not dialogue_lines:
        print("Error: Could not parse any dialogue lines from the script.")
    else:
        # 2. Generate audio for each line
        generated_files = []
        for i, (character, line) in enumerate(dialogue_lines):
            voice = CHARACTER_VOICES.get(character)
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
            combine_audio_segments(generated_files, PAUSE_DURATION_MS, OUTPUT_FILENAME)
        
            # 4. Clean up temporary files
            print("\nCleaning up temporary files...")
            for f in generated_files:
                f.unlink()
            print("Cleanup complete.")
        else:
            print("No audio files were generated to combine.")