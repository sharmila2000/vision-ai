# Google Gemini Video Chatbot 🤖📹

An interactive, multi-turn video chatbot powered by **Google Gemini**. 

Unlike frame-by-frame heuristic pipelines, this chatbot leverages Gemini's native **multimodal temporal attention** and its massive **1-million-token context window**. You can upload any standard video file (such as `.mp4`, `.avi`, or `.mov`) and immediately chat with it in natural language—perfect for complex event reasoning, safety hazard reviews (e.g. car accidents, injuries), and open-ended video QA.

---

## 🚀 Key Features

* **Native Video Processing**: Uploads video files directly to Google's File API, letting Gemini handle spatial/temporal decoding. No manual frame slicing or downsampling is required.
* **Multi-Turn Conversation**: Remembers the video and the conversation context. You can ask follow-up questions, ask for clarifications, or probe specific elements.
* **Auto-Cleanup**: Automatically cleans up and deletes your uploaded files from Google's servers on exit (even on Ctrl+C interruption) so you don't leak files or exceed quotas.
* **Support for `/reset`**: Clear the chat history and start fresh with the same video without having to re-upload it.
* **Cost-Efficient**: Fully compatible with the **Google AI Studio Free Tier** (check [current free-tier rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), which vary by model and change over time).

---

## 🛠️ Setup Instructions

### 1. Obtain your Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Log in with your Google account.
3. Click **"Create API key"** and choose **"Create API key in new project"**.
4. Copy the generated key.

### 2. Install Dependencies
Make sure you have activated your Python virtual environment, then install the required packages:

```bash
source .venv/bin/activate
pip install -r gemini_chatbot/requirements.txt
```

### 3. Set the Environment Variable
Expose your API key to your environment:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

*Alternatively, you can create a `.env` file in the root folder of this project:*
```env
GEMINI_API_KEY="your_api_key_here"
```

---

## 💬 Usage Guide

To run the chatbot, simply point it to any local video file:

```bash
python gemini_chatbot/chatbot.py --video videos/video_01.mp4
```

### Optional CLI Arguments
* `--model` / `-m`: Select a Gemini model, e.g. `"gemini-flash-latest"` (default, ultra-fast) or `"gemini-pro-latest"` (for complex reasoning/math). Using the `-latest` aliases avoids breakage when Google retires a specific dated model version.
* `--system-instruction` / `-s`: Specify a custom persona or strict rules for the model.

*Example using the Pro model:*
```bash
python gemini_chatbot/chatbot.py --video videos/video_01.mp4 --model gemini-pro-latest
```

---

## 🕹️ Interactive Chat Commands

While inside the chat session, you can use these special commands:

* `/help` — Display the help message.
* `/reset` — Reset conversation history (retains the uploaded video so you don't have to wait for uploading again).
* `/exit` — Cleanly delete the remote file from Google's servers and quit.

---

## 💡 Example Queries to Try

* *"What happened in this video? Give me a full summary."*
* *"Is there any person wearing a green shirt? Do they get injured or fall down?"*
* *"At what timestamp did the accident/event occur?"*
* *"Identify the license plate or the color of the car that crashed."*
