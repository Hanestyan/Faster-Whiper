from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from moviepy import VideoFileClip
import os
import time

def transcribe_and_translate(video_path, output_text_path):
    audio_path = "temp_audio.wav"
    
    # 1. Ekstraksi Audio
    print("[1] Mengekstrak audio dari video...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    
    # 2. Load Model
    print("[2] Memuat Faster-Whisper...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    # 3. Inisialisasi Penerjemah ke Bahasa Indonesia
    translator = GoogleTranslator(source='auto', target='id')
    
    # 4. Transkripsi dan Translasi
    print("[3] Memulai proses transkripsi dan translasi ")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    with open(output_text_path, "w", encoding="utf-8") as file:
        for segment in segments:
            # Mengonversi waktu dari detik ke menit:detik
            start_min, start_sec = divmod(segment.start, 60)
            end_min, end_sec = divmod(segment.end, 60)
            timestamp = f"[{int(start_min):02d}:{int(start_sec):02d} -> {int(end_min):02d}:{int(end_sec):02d}]"
            
            # Menerjemahkan teks asli ke bahasa Indonesia
            translated_text = " "
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    translated_text = translator.translate(segment.text)
                    break # Keluar dari loop retry jika berhasil
                except Exception as e:
                    if attempt < max_retries - 1:
                        print("  [Google limit tercapai, menunggu 3 detik sebelum coba lagi...]")
                        time.sleep(3)
                    else:
                        translated_text = f"[GAGAL DITERJEMAHKAN: {segment.text}]"
            # Menulis ke file text
            line = f"{timestamp} {translated_text}\n"
            file.write(line)
            
            # Menampilkan progress di terminal
            print(line.strip())
            time.sleep(0.5)
            
    os.remove(audio_path)
    print(f"\n[4] Hasil terjemahan tersimpan di: {output_text_path}")

# Eksekusi fungsi
transcribe_and_translate(r"routing\WhatsApp Video 2026-10-28 at 8.03.54 PM.mp4", "what is routing.txt")