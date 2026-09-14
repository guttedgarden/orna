-- Cosine distance не определена для zero vector и возвращает NULL.
ALTER TABLE memories
ADD CONSTRAINT chk_memories_embedding_nonzero
CHECK (vector_norm(embedding) > 0);
