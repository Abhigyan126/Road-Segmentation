import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from utils import VideoStitcherOpenCV
import threading

class VideoStitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentation")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        self.video_path = tk.StringVar()
        self.limit_frames = tk.IntVar()
        self.skip_frames = tk.IntVar()
        self.video_preview = None
        self.video_canvas = None
        self.video_capture = None
        self.playback_thread = None
        self.is_playing = False

        self.create_widgets()

    def create_widgets(self):
        # Video Upload Button
        upload_btn = tk.Button(self.root, text="Upload Video", command=self.upload_video, bg="#404040", fg="white")
        upload_btn.pack(pady=10)

        # Video Frame
        video_frame = tk.Frame(self.root, bg="black")
        video_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Video Canvas
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=400, height=300)
        self.video_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Play/Pause Buttons
        controls_frame = tk.Frame(video_frame, bg="#1e1e1e")
        controls_frame.pack(side=tk.BOTTOM, fill="x")

        play_btn = tk.Button(controls_frame, text="Play", command=self.play_video, bg="#404040", fg="white")
        play_btn.pack(side=tk.LEFT, padx=10)
        pause_btn = tk.Button(controls_frame, text="Pause", command=self.stop_video_playback, bg="#404040", fg="white")
        pause_btn.pack(side=tk.LEFT)

        # Limit Frames Entry
        limit_frames_label = tk.Label(controls_frame, text="Limit Frames:", bg="#1e1e1e", fg="white")
        limit_frames_label.pack(side=tk.LEFT, padx=(10, 5))
        limit_frames_entry = tk.Entry(controls_frame, textvariable=self.limit_frames, bg="#404040", fg="white")
        limit_frames_entry.pack(side=tk.LEFT)

        # Skip Frames Entry
        skip_frames_label = tk.Label(controls_frame, text="Skip Frames:", bg="#1e1e1e", fg="white")
        skip_frames_label.pack(side=tk.LEFT, padx=(10, 5))
        skip_frames_entry = tk.Entry(controls_frame, textvariable=self.skip_frames, bg="#404040", fg="white")
        skip_frames_entry.pack(side=tk.LEFT)

        # Start Processing Button
        start_btn = tk.Button(controls_frame, text="Start Processing", command=self.process_video, bg="#404040", fg="white")
        start_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Save Panorama Button
        save_btn = tk.Button(controls_frame, text="Save Panorama", command=self.save_panorama, bg="#404040", fg="white")
        save_btn.pack(side=tk.RIGHT, padx=(0, 10))
   
    def upload_video(self):
        self.video_path.set(filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")]))

        # Preview the first frame of the video
        cap = cv2.VideoCapture(self.video_path.get())
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame.thumbnail((400, 300))
            self.video_preview = ImageTk.PhotoImage(frame)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_preview)
            self.video_capture = cap

    def process_video(self):
        if self.video_capture is None:
            print("Please upload a video first.")
            return

        # Call the OpenCV part to process the video
        video_stitcher_opencv = VideoStitcherOpenCV()
        video_stitcher_opencv.process_video(self.video_path.get(), self.limit_frames.get(), self.skip_frames.get())

    def save_panorama(self):
        print("Save Panorama button clicked.")

    def start_video_playback(self):
        self.is_playing = True
        while self.is_playing:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame.thumbnail((400, 300))
                video_frame = ImageTk.PhotoImage(frame)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=video_frame)
                self.video_canvas.image = video_frame
                self.root.update() 
                break  


    def stop_video_playback(self):
        self.is_playing = False

    def play_video(self):
        if self.video_capture is None:
            print("Please upload a video first.")
            return

        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_video_playback()
            self.playback_thread.join()

        self.playback_thread = threading.Thread(target=self.start_video_playback)
        self.playback_thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStitcherApp(root)
    root.mainloop()
