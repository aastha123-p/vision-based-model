# TODO - Project Tasks

## Completed Tasks:

### Phase 3 – Task 2: Similarity Search (FAISS)
- [x] Add FAISS dependency to requirements.txt
- [x] Create core/faiss_store.py - FAISSStore class with:
  - add_embedding() - Add single embedding to index
  - add_embeddings() - Add multiple embeddings
  - search() - Search for similar cases
  - save_index()/load_index() - Persist index to disk
  - get_similarity_percentage() - Convert distance to similarity %
- [x] Create app/agents/learning_agent.py - LearningAgent class with:
  - store_case() - Store case with embeddings
  - find_similar_cases() - Find similar cases by embedding
  - get_top_similar_cases() - Convenience method for similarity search
  - get_stats() - Get agent statistics
  - save_index()/reload_index() - Index management

### Phase 3 – Task 1: Embedding Generation (Previously completed)
- [x] Fix 1: Add explicit numpy/tensor handling in generate_embedding method
- [x] Fix 2: Add singleton pattern or class-level caching for the EmbeddingEngine
- [x] Fix 3: Add batch processing method for generating multiple embeddings at once

## Pending Tasks:

### Phase 4 – Agentic AI Reasoning
- Phase 4 – Task 1: Condition Prediction (LLM Reasoning)
  - Create agents/condition_agent.py
  - Create core/llm_engine.py
- Phase 4 – Task 2: Medication Suggestion + Safety Layer
  - Create agents/medication_agent.py
  - Create agents/safety_agent.py
  - Create core/safety_rules.py

### Phase 5 – Reporting & Output
- Phase 5 – Task 1: Dashboard & Visualization
  - Update frontend/streamlit_app.py
- Phase 5 – Task 2: PDF & Email Report
  - Create reports/pdf_generator.py
  - Create reports/email_service.py
