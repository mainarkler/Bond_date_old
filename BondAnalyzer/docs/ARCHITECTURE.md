# Architecture
`document_loader` preserves PDF page numbers and represents DOCX as logical page 1. Rule extraction always creates a text-backed source. The optional localhost-only llama.cpp result is validated before merge; unverified values are discarded.
