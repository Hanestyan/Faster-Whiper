from faster_whisper import WhisperModel
from moviepy import VideoFileClip
import os

def transcribe_only(video_path, output_text_path):
    audio_path = "temp_audio.wav"
    
    print("[1] Mengekstrak audio...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    
    print("[2] Memuat model Faster-Whisper...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print("[3] Memulai proses transkripsi offline (sangat cepat)...")
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    
    with open(output_text_path, "w", encoding="utf-8") as file:
        for segment in segments:
            start_min, start_sec = divmod(segment.start, 60)
            end_min, end_sec = divmod(segment.end, 60)
            timestamp = f"[{int(start_min):02d}:{int(start_sec):02d} -> {int(end_min):02d}:{int(end_sec):02d}]"
            
            line = f"{timestamp} {segment.text}\n"
            file.write(line)
            
            print(line.strip())
            
    os.remove(audio_path)
    print(f"\n[4] Selesai! File bahasa Inggris tersimpan di: {output_text_path}")

transcribe_only(r"routing\WhatsApp Video 2026-10-28 at 8.34.23 PM.mp4", " 5.10_en.txt")