import os
import json
import asyncio
import pandas as pd
import numpy as np
import whisper
from openai import AsyncOpenAI
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Standard tqdm, not asyncio

# --- CONFIGURATION ---
API_KEY = "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"
INPUT_AUDIO_FOLDER = "./audios_gemini_new"
DATASET_FILE = "../claims_train.jsonl"
OUTPUT_CSV = "./gemini_fact_check_results1.csv"

if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
class VectorStore:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []
        self.embeddings = None

    def add_documents(self, docs: List[Dict]):
        texts = [d['text'] for d in docs]
        if not texts: return

        print(f"Embedding {len(texts)} database entries...")
        new_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.documents.extend(texts)
        self.metadata.extend(docs)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            res = self.metadata[idx].copy()
            res['score'] = similarities[idx]
            results.append(res)
        return results

class BatchProcessor:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=API_KEY)
        self.vector_store = VectorStore()
        print("Loading Whisper model...")
        # fp16=False prevents CPU NaN errors
        self.whisper_model = whisper.load_model("base")
        self.ground_truth_map = {} 

    def ingest_dataset(self, filepath):
        print(f"Reading {filepath}...")
        docs_to_embed = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    claim_id = str(data.get('id', ''))
                    text = data.get('claim', '')
                    
                    actual_label = "NOT_ENOUGH_INFO"
                    evidence = data.get('evidence', {})
                    for doc_id, ev_list in evidence.items():
                        for ev in ev_list:
                            if 'label' in ev:
                                actual_label = ev['label']
                                break
                        if actual_label != "NOT_ENOUGH_INFO": break
                    
                    self.ground_truth_map[claim_id] = {"text": text, "label": actual_label}
                    docs_to_embed.append({
                        "text": text,
                        "label": actual_label,
                        "id": claim_id,
                        "rag_content": f"Verified Claim: {text} | Verdict: {actual_label}"
                    })
            self.vector_store.add_documents(docs_to_embed)
        except FileNotFoundError:
            print(f"CRITICAL: Dataset file {filepath} not found.")
            exit(1)

    def transcribe_audio(self, file_path):
        """Step 1: Audio -> Raw Transcript (Sequential)"""
        try:
            # Force FP32 to avoid NaN errors on CPU
            result = self.whisper_model.transcribe(file_path, fp16=False)
            return result["text"].strip()
        except Exception as e:
            return f"TRANSCRIPTION_ERROR: {e}"

    async def extract_clean_claim(self, transcript):
        """Step 2: Raw Transcript -> Clean Scientific Claim"""
        # Fail fast if transcription failed
        if "TRANSCRIPTION_ERROR" in transcript:
            return transcript

        system_prompt = "You are an assistant that extracts the core scientific claim from a transcript."
        user_prompt = f"""
        Extract the SINGLE, specific scientific claim from this text. 
        Remove filler words, intro/outro, and phrases like 'The claim is...'
        Output ONLY the claim statement.

        Transcript: "{transcript}"
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM_ERROR: {e}"

    async def verify_claim(self, clean_claim):
        """Step 3: Clean Claim -> Verification"""
        print(clean_claim)
        if "ERROR" in clean_claim:
            return "ERROR", "No Data"

        results = self.vector_store.search(clean_claim, k=1)
        if not results:
            return "NO_MATCH_FOUND", "No Data"

        top_match = results[0]
        context = top_match['rag_content']
        
        system_prompt = "You are a scientific fact checker."
        user_prompt = f"""
        EXTRACTED CLAIM: "{clean_claim}"
        DATABASE ENTRY: {context}
        
        Task: Does the Database Entry SUPPORT or CONTRADICT the Extracted Claim?
        Output ONLY one word: "SUPPORT", "CONTRADICT", or "NEI".
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip().upper(), top_match['text']
        except Exception as e:
            return "LLM_ERROR", str(e)

    async def process_single_file(self, file_path):
        filename = os.path.basename(file_path)
        claim_id = os.path.splitext(filename)[0]
        
        gt_data = self.ground_truth_map.get(claim_id, {"text": "Unknown ID", "label": "UNKNOWN"})
        actual_label = gt_data['label']
        
        # 1. Transcribe (Runs directly in main thread now, strictly sequential)
        transcript = self.transcribe_audio(file_path)
        
        # 2. Extract Claim
        extracted_claim = await self.extract_clean_claim(transcript)
        
        # 3. Verify
        predicted_label, matched_db_text = await self.verify_claim(extracted_claim)
        
        return {
            "Claim_ID": claim_id,
            "Actual_Label": actual_label,
            "Predicted_Label": predicted_label,
            "Correct_Prediction": (actual_label == predicted_label),
            "Extracted_Claim": extracted_claim,
            "Matched_DB_Claim": matched_db_text,
            "Original_Transcript": transcript
        }

async def main():
    processor = BatchProcessor()
    processor.ingest_dataset(DATASET_FILE)
    
    if not os.path.exists(INPUT_AUDIO_FOLDER):
        print(f"Error: Folder '{INPUT_AUDIO_FOLDER}' not found.")
        return

    audio_files = [
        os.path.join(INPUT_AUDIO_FOLDER, f) 
        for f in os.listdir(INPUT_AUDIO_FOLDER) 
        if f.lower().endswith(('.mp3', '.wav'))
    ]
    
    if not audio_files:
        print("No audio files found.")
        return

    print(f"\nStarting SEQUENTIAL processing for {len(audio_files)} files...")
    
    results = []
    # Standard Loop - One file at a time
    for i, file_path in enumerate(tqdm(audio_files)):
        result = await processor.process_single_file(file_path)
        results.append(result)
        
        # Optional: Save partial progress every 10 files
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    # Final Save
    df = pd.DataFrame(results)
    cols = ["Claim_ID", "Actual_Label", "Predicted_Label", "Correct_Prediction", "Extracted_Claim", "Matched_DB_Claim", "Original_Transcript"]
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Results saved to '{OUTPUT_CSV}'")
    
    valid_results = df[df["Actual_Label"] != "UNKNOWN"]
    if not valid_results.empty:
        acc = valid_results["Correct_Prediction"].mean() * 100
        print(f"Accuracy (on known labels): {acc:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())