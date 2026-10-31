"""
gemini_chatbot/client.py — Gemini API interaction layer.
Handles API key configuration, video uploads, status tracking, multi-turn chat sessions, and cleanup.
"""
from __future__ import annotations

import os
import time
from google import genai
from google.genai import types
from typing import Any, List, Optional

class GeminiVideoClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        """
        Initializes the Gemini API client.
        
        Args:
            api_key: The Google Gemini API key. If None, it attempts to load from GEMINI_API_KEY env var.
            model_name: The name of the model to use (default: gemini-flash-latest for speed/cost).
                Note: "gemini-1.5-flash"/"gemini-1.5-pro" have been retired by Google and will
                return 404 errors. Use the rolling "-latest" aliases (e.g. "gemini-flash-latest",
                "gemini-pro-latest", "gemini-flash-lite-latest") or a current dated model name.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please provide it in the constructor or "
                "set the 'GEMINI_API_KEY' environment variable."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.uploaded_files: List[types.File] = []

    def upload_video(self, video_path: str, progress_callback: Optional[callable] = None) -> types.File:
        """
        Uploads a video to Google's File API and waits for processing to complete.
        
        Args:
            video_path: Local path to the video file (.mp4, .avi, etc.).
            progress_callback: A function called periodically with status updates.
            
        Returns:
            The Google File object once it is ACTIVE and ready.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        if progress_callback:
            progress_callback("Uploading video file to Gemini...")
        else:
            print(f"Uploading {video_path} to Gemini File API...")

        file_obj = self.client.files.upload(file=video_path)
        self.uploaded_files.append(file_obj)

        if progress_callback:
            progress_callback(f"Upload complete. URI: {file_obj.uri}. Waiting for processing...")
        else:
            print(f"Upload complete. URI: {file_obj.uri}\nWaiting for video processing to finish...")

        # Wait for file to become active
        start_time = time.time()
        while True:
            file_obj = self.client.files.get(name=file_obj.name)
            state = file_obj.state

            is_active = (state == types.FileState.ACTIVE or getattr(state, "name", None) == "ACTIVE" or str(state) == "ACTIVE")
            is_failed = (state == types.FileState.FAILED or getattr(state, "name", None) == "FAILED" or str(state) == "FAILED")

            if is_active:
                if progress_callback:
                    progress_callback("Video processing completed successfully!")
                else:
                    print("\nVideo is ready for analysis!")
                break
            elif is_failed:
                error_msg = getattr(file_obj, "error", None)
                err_text = error_msg.message if error_msg else "Unknown error"
                raise ValueError(f"Video processing failed on Gemini's server: {err_text}")
            
            # Print a dot or call progress callback
            if progress_callback:
                progress_callback(f"Processing... (elapsed: {int(time.time() - start_time)}s)")
            else:
                print(".", end="", flush=True)
            
            time.sleep(5)

        return file_obj

    def start_chat_with_video(self, video_file: types.File, system_instruction: Optional[str] = None) -> genai.chats.Chat:
        """
        Starts a multi-turn chat session with the uploaded video already in its history context.
        
        Args:
            video_file: The processed File object from upload_video.
            system_instruction: Optional instructions to guide the model's behavior.
            
        Returns:
            A generative ChatSession initialized with the video.
        """
        # Create a model config with system instructions if provided
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        ) if system_instruction else None
        
        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )
        
        # Inject the video as the first turn in the chat session
        # This registers the video inside the chat memory context.
        initial_setup_prompt = (
            "You are an expert video analysis chatbot. I have uploaded a video. "
            "Please analyze it and acknowledge that you are ready to answer my questions."
        )
        
        # Send the file and instructions as the first message
        chat.send_message([video_file, initial_setup_prompt])
        return chat

    def clean_up_file(self, file_obj: types.File) -> bool:
        """
        Deletes a specific uploaded file from Google's servers.
        """
        try:
            print(f"Deleting remote file {file_obj.name}...")
            self.client.files.delete(name=file_obj.name)
            if file_obj in self.uploaded_files:
                self.uploaded_files.remove(file_obj)
            return True
        except Exception as e:
            print(f"Failed to delete remote file {file_obj.name}: {e}")
            return False

    def clean_up_all(self) -> None:
        """
        Deletes all uploaded files tracked by this client instance.
        """
        # Copy list to avoid modification during iteration
        for f in list(self.uploaded_files):
            self.clean_up_file(f)
