# TODO - Fix three problems in core/embedding_engine.py

## Problems Identified:
1. **Line 43 - Missing explicit numpy conversion**: The `.encode()` method needs explicit handling for numpy conversion
2. **Line 52-53 - Inefficient model loading**: The convenience function creates new model instance every call
3. **Missing batch processing support**: Need to add batch processing for efficiency

## Plan:
- [x] Fix 1: Add explicit numpy/tensor handling in generate_embedding method
- [x] Fix 2: Add singleton pattern or class-level caching for the EmbeddingEngine
- [x] Fix 3: Add batch processing method for generating multiple embeddings at once

## Summary of Changes:
1. Added singleton pattern with `__new__` method and `_instances` class variable
2. Added check in `__init__` to skip re-initialization if already initialized
3. Modified `generate_embeddings` to use batch processing instead of calling `generate_embedding` for each text
4. Added `generate_batch_embeddings` method for efficient batch processing of multiple texts
5. Added explicit numpy/tensor conversion with `hasattr(emb, 'numpy')` check
