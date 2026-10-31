#!/usr/bin/env python3
"""
gemini_chatbot/chatbot.py — Interactive CLI Video Chatbot powered by Google Gemini.

Usage:
    python gemini_chatbot/chatbot.py --video videos/video_01.mp4

Optional flags:
    --model gemini-1.5-pro      Use Pro model instead of Flash (Flash is default and faster)
    --system-instruction "..."  Define custom persona or system guidelines
"""
from __future__ import annotations

import argparse
import sys
import os
from typing import Optional

# Ensure the root of the repo is in the python path so we can run from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not installed, we can still proceed if GEMINI_API_KEY is in env,
    # but let's notify the user if there is a broader environment mismatch.
    pass

try:
    from gemini_chatbot.client import GeminiVideoClient
except ImportError as exc:
    print(f"\n[Import Error]: {exc}")
    print("Please make sure you have activated the virtual environment and installed dependencies.")
    print("Run:")
    print("   source .venv/bin/activate")
    print("   pip install -r gemini_chatbot/requirements.txt")
    print("And then try running this script again.\n")
    sys.exit(1)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a highly precise, factual, and helpful video analysis assistant. "
    "Your task is to analyze the uploaded video and answer user questions about it. "
    "Be descriptive, mention relevant actions, objects, colors, and timestamps if they "
    "are visible or clear. If you are unsure or if something is not in the video, state "
    "that clearly rather than making it up."
)

def run_chatbot(video_path: str, model_name: str, system_instruction: Optional[str]) -> None:
    # 1. Initialize client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[Error] GEMINI_API_KEY environment variable not set.")
        print("Please obtain an API key from Google AI Studio (https://aistudio.google.com/)")
        print("and set it in your terminal:")
        print("   export GEMINI_API_KEY=\"your_key_here\"")
        print("or create a '.env' file containing:")
        print("   GEMINI_API_KEY=\"your_key_here\"\n")
        sys.exit(1)

    print("=" * 65)
    print(f" Gemini Video Chatbot — Starting Session")
    print(f" Video file: {video_path}")
    print(f" Model:      {model_name}")
    print("=" * 65)

    client = GeminiVideoClient(api_key=api_key, model_name=model_name)
    video_file = None

    try:
        # 2. Upload video
        video_file = client.upload_video(video_path)

        # 3. Initialize chat session
        print("\nInitializing chat session with the video...")
        sys_inst = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        chat = client.start_chat_with_video(video_file, system_instruction=sys_inst)
        
        print("\n" + "=" * 65)
        print(" Chat session established! You can now query your video.")
        print(" Type your questions below. Here are some special commands:")
        print("   /help   - Show this help message")
        print("   /reset  - Clear conversation history and restart the chat")
        print("   /exit   - Clean up remote files and quit the chatbot")
        print("=" * 65 + "\n")

        # 4. Interactive loop
        while True:
            try:
                user_query = input("You: ").strip()
                if not user_query:
                    continue

                # Handle commands
                if user_query.lower() in ("/exit", "/quit", "/q"):
                    print("\nExiting chatbot...")
                    break
                
                elif user_query.lower() == "/help":
                    print("\n--- Available Commands ---")
                    print("  /help   : Show this help message")
                    print("  /reset  : Reset the conversation history (video is retained)")
                    print("  /exit   : Delete uploaded video from cloud and quit")
                    print("--------------------------\n")
                    continue
                
                elif user_query.lower() == "/reset":
                    print("\nResetting chat history...")
                    chat = client.start_chat_with_video(video_file, system_instruction=sys_inst)
                    print("Chat history reset. Ask a new question!\n")
                    continue

                # Send message to model and stream response
                print(f"\n{model_name}: ", end="", flush=True)
                
                # We use streaming for a better UI experience
                response = chat.send_message_stream(user_query)
                for chunk in response:
                    print(chunk.text, end="", flush=True)
                print("\n")  # Trailing newline

            except KeyboardInterrupt:
                # Ctrl+C inside prompt
                print("\n(Use /exit to quit cleanly, or Ctrl+C again to force close)")
                continue
            except Exception as e:
                print(f"\n[Error during query]: {e}\n")

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
    except Exception as e:
        print(f"\n[Initialization Error]: {e}")
    finally:
        # 5. Clean up remote files
        if video_file and client:
            print("\nCleaning up resources...")
            client.clean_up_all()
        print("Session ended. Goodbye!\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Video Chatbot utilizing Google Gemini 1.5 models.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to the local video file to analyze."
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-flash-latest",
        help="Gemini model to use, e.g. 'gemini-flash-latest' (default, fast/cheap), "
             "'gemini-pro-latest' (more powerful), or 'gemini-flash-lite-latest' (cheapest). "
             "Any valid Gemini model name is accepted; the '-latest' aliases automatically "
             "track Google's current recommended model instead of pinning to a version that "
             "may later be retired."
    )
    parser.add_argument(
        "--system-instruction", "-s",
        default=None,
        help="Define a custom system instruction/persona for the model."
    )

    args = parser.parse_args()
    run_chatbot(args.video, args.model, args.system_instruction)


if __name__ == "__main__":
    main()
