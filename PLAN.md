# Plan: Phase 3 – Task 2: Similarity Search

## Information Gathered:
- **core/embedding_engine.py**: Already exists and generates text embeddings using SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
- **app/database/models.py**: Has Patient model with face_embedding field
- **app/database/crud.py**: Has basic Patient CRUD operations
- **app/agents/**: Empty directory (needs learning_agent.py)
- **requirements.txt**: Missing FAISS dependency

## Plan:

### Step 1: Add FAISS to requirements.txt
- Add `faiss-cpu` or `faiss-gpu` to requirements.txt

### Step 2: Create core/faiss_store.py
- Create FAISSStore class to manage vector index
- Methods:
  - `add_embedding(embedding, case_id, metadata)` - Add embedding to index
  - `search(query_embedding, top_k)` - Search for similar cases
  - `save_index(path)` - Save index to disk
  - `load_index(path)` - Load index from disk
  - `get_similarity_percentage(distance)` - Convert distance to similarity %

### Step 3: Create app/agents/learning_agent.py
- Create LearningAgent class
- Methods:
  - `store_case(form_text, emotion_summary, sentiment_summary, diagnosis)` - Store case with embeddings
  - `find_similar_cases(query_embedding, top_k)` - Find similar past cases
  - `get_top_similar_cases(form_text, emotion_summary, sentiment_summary, top_k)` - Convenience method

### Step 4: Test the implementation
- Add test code to verify FAISS similarity search works

## Dependent Files to be Edited:
- requirements.txt (add faiss dependency)

## Files to be Created:
- core/faiss_store.py
- app/agents/learning_agent.py

## Followup Steps:
- Install dependencies: pip install faiss-cpu
- Test similarity search functionality
