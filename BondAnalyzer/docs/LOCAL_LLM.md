# Local llama.cpp
Copy a Windows `llama-server.exe` to `runtime/llama/` and a GGUF model to `models/`. The application starts the server on `127.0.0.1:8765`, waits for `/health`, calls `/v1/chat/completions`, then terminates it. No non-loopback HTTP endpoint is used.
