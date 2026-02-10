# 99 Days of AI - Day 3: GPU-Powered LLM Chatbot

## 🚀 The Mission
Upgraded the basic chatbot to a modern, instruction-tuned LLM (Qwen-1.8B) capable of handling complex logic and math.

## 🛠️ Tech Stack
- **Model:** Qwen/Qwen1.5-1.8B-Chat
- **Framework:** PyTorch & Hugging Face Transformers
- **Hardware:** NVIDIA T4 GPU (via Google Colab)

## 💡 Key Learning
I learned how to use `device_map="auto"` to automatically offload model weights to the GPU, reducing latency from minutes to seconds.
