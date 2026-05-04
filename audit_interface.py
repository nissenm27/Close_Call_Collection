import os, csv, re, threading, time, shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from queue import Queue, Empty

import pandas as pd
import cv2
from PIL import Image, ImageTk


DOWNLOADS_FOLDER = os.path.expanduser("~/Downloads")

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

EVENT_MAPPING = {0: "Conflict", 1: "Bump", 2: "Hard Brake", 3: "Not SCE"}
NAME_TO_IDX   = {v: k for k, v in EVENT_MAPPING.items()}

LABEL_COLORS = {
    "Conflict":   "#ff4d4d",
    "Bump":       "#ff9933",
    "Hard Brake": "#ffcc00",
    "Not SCE":    "#33cc66",
}


class VideoPlayer(tk.Frame):
    W, H = 720, 405   

    def __init__(self, parent, **kw):
        kw.pop("bg", None)
        super().__init__(parent, bg="#000", **kw)

        self.canvas = tk.Canvas(self, width=self.W, height=self.H,
                                bg="#000", highlightthickness=0)
        self.canvas.pack()

        ctrl = tk.Frame(self, bg="#1a1a1a")
        ctrl.pack(fill="x")

        btn = dict(bg="#2a2a2a", fg="white", relief="flat",
                   padx=14, pady=4, font=("Helvetica", 13), cursor="hand2")
        tk.Button(ctrl, text="▶", command=self.play,   **btn).pack(side="left", padx=2, pady=4)
        tk.Button(ctrl, text="⏸", command=self.pause,  **btn).pack(side="left", padx=2)
        tk.Button(ctrl, text="↺", command=self.replay, **btn).pack(side="left", padx=2)

        self._progress = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                   command=self._on_seek)
        self._progress.pack(side="left", fill="x", expand=True, padx=10, pady=6)
        self._time_lbl = tk.Label(ctrl, text="0:00", bg="#1a1a1a", fg="#888",
                                  width=6, font=("Courier", 9))
        self._time_lbl.pack(side="right", padx=8)

        self._cap       = None
        self._playing   = False
        self._frame_idx = 0
        self._total     = 0
        self._fps       = 30.0
        self._photo     = None
        self._lock      = threading.Lock()
        self._seeking   = False
        self._q: Queue  = Queue(maxsize=2)
        self._poll_frames()

    def load(self, path: str):
        self.stop()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._msg("Could not open video.")
            return
        with self._lock:
            self._cap       = cap
            self._total     = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            self._fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._frame_idx = 0
        self._grab(0)

    def play(self):
        with self._lock:
            if self._cap is None or self._playing:
                return
            self._playing = True
        threading.Thread(target=self._loop, daemon=True).start()

    def pause(self):
        with self._lock:
            self._playing = False

    def stop(self):
        with self._lock:
            self._playing = False
        while not self._q.empty():
            try: self._q.get_nowait()
            except Empty: break
        time.sleep(0.06)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    def replay(self):
        self.stop()
        time.sleep(0.05)
        with self._lock:
            self._frame_idx = 0
        self.play()

    def _grab(self, idx):
        with self._lock:
            if self._cap is None: return
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
        if ret:
            try: self._q.put_nowait((frame, idx))
            except Exception: pass

    def _loop(self):
        with self._lock:
            if self._cap is None: return
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
            fps = self._fps
        delay = 1.0 / fps
        while True:
            with self._lock:
                if not self._playing or self._cap is None: break
                ret, frame = self._cap.read()
                idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                if not ret:
                    self._playing = False; break
                self._frame_idx = idx
            try: self._q.put_nowait((frame, idx))
            except Exception: pass
            time.sleep(delay)

    def _poll_frames(self):
        try:
            frame, idx = self._q.get_nowait()
            self._draw(frame, idx)
        except Empty:
            pass
        self.after(16, self._poll_frames)

    def _draw(self, frame, idx):
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((self.W, self.H), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._photo = photo
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self._seeking = True
            if self._total > 0:
                self._progress.set(100.0 * idx / self._total)
                s = idx / max(self._fps, 1)
                self._time_lbl.config(text=f"{int(s//60)}:{int(s%60):02d}")
            self._seeking = False
        except Exception:
            self._seeking = False

    def _on_seek(self, val):
        if self._seeking: return
        with self._lock:
            total = self._total
        if total > 0:
            t = int(float(val) / 100.0 * total)
            with self._lock:
                self._frame_idx = t
            if not self._playing:
                self._grab(t)

    def _msg(self, text):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.W, self.H, fill="#000")
        self.canvas.create_text(self.W // 2, self.H // 2, text=text,
                                fill="#444", font=("Helvetica", 13), justify="center")




def build_index(folder: str):
    files = sorted(fp for fp in Path(folder).rglob("*")
                   if fp.suffix.lower() in VIDEO_EXTENSIONS)
    index = {}
    for fp in files:
        stem = fp.stem.lower()
        index[stem] = str(fp)
        for num in re.findall(r"\d+", stem):
            index.setdefault(num.lstrip("0") or "0", str(fp))
            index.setdefault(num.zfill(6), str(fp))
    print(f"[Index] {len(files)} videos indexed from {folder}")
    for f in files[:10]:
        print(f"  {f.name}")
    return index, files


def find_video(row, index, files, row_idx):
    bdd_id   = str(row.get("BDD_ID", "")).lower()
    event_id = str(row.get("EVENT_ID", row_idx)).lower()
    for key in [bdd_id, event_id, event_id.zfill(6), event_id.lstrip("0") or "0"]:
        if key in index:
            print(f"[Match] row {row_idx} → {Path(index[key]).name}")
            return index[key]
    if files:
        pos = row_idx % len(files)
        print(f"[Fallback] row {row_idx} → position {pos} → {files[pos].name}")
        return str(files[pos])
    return None


class CompleteLog:
    
    def __init__(self, path: str):
        self.path = path
       
        if os.path.exists(path):
            self._df = pd.read_csv(path)
        else:
            self._df = pd.DataFrame(columns=[
                "EVENT_ID", "BDD_ID", "EVENT_TYPE", "EVENT_TYPE_NAME",
                "CONFLICT_TYPE", "BDD_START", "AUDIT_CORRECTED"
            ])

        self._reviewed: set = set(self._df["EVENT_ID"].astype(str).tolist()
                                  if "EVENT_ID" in self._df.columns else [])

    def write(self, event_id, bdd_id, event_type_idx, event_type_name):
        """Append or update a row. Writes to disk immediately."""
        new_row = {
            "EVENT_ID":        event_id,
            "BDD_ID":          bdd_id,
            "EVENT_TYPE":      event_type_idx,
            "EVENT_TYPE_NAME": event_type_name,
            "CONFLICT_TYPE":   "",
            "BDD_START":       0.0,
            "AUDIT_CORRECTED": True,
        }
       
        if "EVENT_ID" in self._df.columns:
            self._df = self._df[self._df["EVENT_ID"].astype(str) != str(event_id)]
        self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
        self._df.to_csv(self.path, index=False)
        self._reviewed.add(str(event_id))

    def already_done(self, event_id) -> bool:
        return str(event_id) in self._reviewed

    def export_to_downloads(self) -> str:
        dest = os.path.join(DOWNLOADS_FOLDER, "complete_report_with_audit.csv")
        shutil.copy2(self.path, dest)
        return dest



class SetupScreen(tk.Tk):
   
    def __init__(self):
        super().__init__()
        self.title("BDD100K Audit Review — Setup")
        self.configure(bg="#0d0d0d")
        self.resizable(False, False)

        self._video_folder = tk.StringVar()
        self._audit_csv    = tk.StringVar(
            value=os.path.join(DOWNLOADS_FOLDER, "audit_log_full.csv"))
        self._complete_csv = tk.StringVar(
            value=os.path.join(DOWNLOADS_FOLDER, "sponsor_report_full.csv"))

        self._build()

    def _build(self):
        # Title
        tk.Label(self, text="BDD100K  AUDIT  REVIEW",
                 bg="#0d0d0d", fg="white",
                 font=("Courier", 18, "bold")
                 ).pack(pady=(32, 4))
        tk.Label(self, text="Select your files to begin",
                 bg="#0d0d0d", fg="#555", font=("Helvetica", 10)
                 ).pack(pady=(0, 24))

        frame = tk.Frame(self, bg="#0d0d0d", padx=40)
        frame.pack(fill="x")

        def row(label, var, is_folder=False):
            tk.Label(frame, text=label, bg="#0d0d0d", fg="#aaa",
                     font=("Helvetica", 9, "bold"), anchor="w"
                     ).pack(fill="x", pady=(10, 2))
            r = tk.Frame(frame, bg="#0d0d0d")
            r.pack(fill="x")
            tk.Entry(r, textvariable=var, bg="#1a1a1a", fg="white",
                     insertbackground="white", relief="flat",
                     font=("Courier", 9), width=54
                     ).pack(side="left", ipady=6, padx=(0, 6))
            cmd = (lambda v=var: v.set(filedialog.askdirectory(title=label))
                   if is_folder else
                   lambda v=var: v.set(
                       filedialog.askopenfilename(
                           title=label,
                           filetypes=[("CSV files","*.csv"),("All","*.*")])))
            tk.Button(r, text="Browse", command=cmd(),
                      bg="#2a2a2a", fg="white", relief="flat",
                      padx=10, pady=4, cursor="hand2"
                      ).pack(side="left")

        row(" Video Folder  (your annotated_videos_only folder)",
            self._video_folder, is_folder=True)
        row(" Audit CSV  (audit_log_full.csv)", self._audit_csv)
        row(" Complete CSV  (sponsor_report_full.csv)", self._complete_csv)

        tk.Button(self, text="Start Review  →",
                  command=self._start,
                  bg="#27ae60", fg="white", relief="flat",
                  font=("Helvetica", 13, "bold"),
                  padx=24, pady=12, cursor="hand2"
                  ).pack(pady=32)

    def _start(self):
        vf = self._video_folder.get().strip()
        ac = self._audit_csv.get().strip()
        cc = self._complete_csv.get().strip()

        if not vf or not os.path.isdir(vf):
            messagebox.showerror("Missing", "Please select a valid video folder.")
            return
        if not ac or not os.path.isfile(ac):
            messagebox.showerror("Missing", "Please select a valid audit CSV file.")
            return
        if not cc:
            messagebox.showerror("Missing", "Please select the complete CSV file.")
            return

        self.destroy()
        app = ReviewApp(vf, ac, cc)
        app.mainloop()



class ReviewApp(tk.Tk):

    def __init__(self, video_folder: str, audit_csv: str, complete_csv: str):
        super().__init__()
        self.title("BDD100K Audit Review")
        self.configure(bg="#0d0d0d")
        self.resizable(False, False)


        self.audit_df  = pd.read_csv(audit_csv)
        self.rows      = self.audit_df.to_dict("records")
        self.idx       = 0
        self._log      = CompleteLog(complete_csv)
        self._complete_csv = complete_csv

       
        self._index, self._files = build_index(video_folder)

  
        self._advance()
        self._build_ui()
        self.after(100, self._load)

    def _advance(self):
        while self.idx < len(self.rows):
            eid = self.rows[self.idx].get("EVENT_ID", self.idx)
            if not self._log.already_done(eid):
                break
            self.idx += 1



    def _build_ui(self):
        total    = len(self.rows)
        reviewed = sum(1 for r in self.rows
                       if self._log.already_done(r.get("EVENT_ID", "")))

        top = tk.Frame(self, bg="#111")
        top.pack(fill="x")
        self._prog_lbl = tk.Label(
            top,
            text=f"0 / {total} reviewed",
            bg="#111", fg="white", font=("Courier", 10)
        )
        self._prog_lbl.pack(side="left", padx=14, pady=8)

        tk.Button(top, text="⬇  Download Complete CSV",
                  command=self._download,
                  bg="#1a5276", fg="white", relief="flat",
                  font=("Helvetica", 9, "bold"), padx=12, pady=4, cursor="hand2"
                  ).pack(side="right", padx=10, pady=6)


        self._status = tk.Label(self, text="", bg="#0d0d0d", fg="#f39c12",
                                font=("Courier", 9), anchor="w", padx=14)
        self._status.pack(fill="x")


        self._player = VideoPlayer(self)
        self._player.pack(padx=0, pady=0)

        
        lf = tk.Frame(self, bg="#0d0d0d")
        lf.pack(fill="x", padx=20, pady=(14, 4))

        tk.Label(lf, text="SELECT CORRECT EVENT TYPE",
                 bg="#0d0d0d", fg="#444",
                 font=("Courier", 8, "bold")
                 ).pack(anchor="w", pady=(0, 8))

        btn_row = tk.Frame(lf, bg="#0d0d0d")
        btn_row.pack(fill="x")

        self._choice = tk.StringVar()
        self._label_btns = {}
        for name, color in LABEL_COLORS.items():
            b = tk.Radiobutton(
                btn_row, text=name,
                variable=self._choice, value=name,
                font=("Helvetica", 11, "bold"),
                fg=color, bg="#0d0d0d",
                selectcolor="#1a1a1a",
                activebackground="#0d0d0d",
                indicatoron=0,
                relief="flat",
                bd=2,
                padx=18, pady=10,
                cursor="hand2",
                width=12,
            )
            b.pack(side="left", padx=4)
            self._label_btns[name] = b

        act = tk.Frame(self, bg="#0d0d0d")
        act.pack(fill="x", padx=20, pady=(8, 16))

        tk.Button(act, text="✅   Save & Next",
                  command=self._save_next,
                  bg="#27ae60", fg="white", relief="flat",
                  font=("Helvetica", 11, "bold"),
                  padx=20, pady=10, cursor="hand2"
                  ).pack(side="left", padx=(0, 8))

        tk.Button(act, text="⏭  Skip",
                  command=self._skip,
                  bg="#2a2a2a", fg="#aaa", relief="flat",
                  font=("Helvetica", 10),
                  padx=14, pady=10, cursor="hand2"
                  ).pack(side="left", padx=4)

        tk.Button(act, text="⬅  Back",
                  command=self._previous,
                  bg="#2a2a2a", fg="#aaa", relief="flat",
                  font=("Helvetica", 10),
                  padx=14, pady=10, cursor="hand2"
                  ).pack(side="left", padx=4)

        self._save_lbl = tk.Label(act, text="", fg="#27ae60",
                                  bg="#0d0d0d", font=("Courier", 9))
        self._save_lbl.pack(side="right")


    def _load(self):
        if self.idx >= len(self.rows):
            messagebox.showinfo(
                "All Done! 🎉",
                f"All {len(self.rows)} audit rows reviewed.\n\n"
                f"Complete CSV saved to:\n{self._complete_csv}"
            )
            self.destroy()
            return

        row      = self.rows[self.idx]
        event_id = row.get("EVENT_ID", self.idx)
        total    = len(self.rows)
        reviewed = sum(1 for r in self.rows
                       if self._log.already_done(r.get("EVENT_ID", "")))

        self._prog_lbl.config(
            text=f"{reviewed} / {total} reviewed  │  row {self.idx+1}"
        )
        self._choice.set("")
        self._save_lbl.config(text="")

        pred_name = row.get("MODEL_PREDICTION_NAME", "")
        if pred_name in NAME_TO_IDX:
            self._choice.set(pred_name)

        self._status.config(text=f"Predicted: {pred_name}")

        threading.Thread(
            target=self._fetch, args=(row, self.idx), daemon=True
        ).start()

    def _fetch(self, row, row_idx):
        path = find_video(row, self._index, self._files, row_idx)
        if path:
            p = path
            self.after(0, lambda: self._player.load(p))
            self.after(0, lambda: self._player.play())
        else:
            self.after(0, lambda: self._player._msg(
                "No video found for this row.\n"
                "You can still select a label and save."))


    def _save_next(self):
        sel = self._choice.get()
        if not sel:
            messagebox.showwarning("No Selection",
                                   "Please select the correct event type first.")
            return
        row      = self.rows[self.idx]
        event_id = row.get("EVENT_ID", self.idx)
        bdd_id   = row.get("BDD_ID", f"bdd_{str(event_id).zfill(6)}")
        self._log.write(event_id, bdd_id, NAME_TO_IDX[sel], sel)
        self._save_lbl.config(text=f"✓ Saved as {sel}")
        self.idx += 1
        self._advance()
        self._load()

    def _skip(self):
        self.idx += 1
        self._load()

    def _previous(self):
        if self.idx > 0:
            self.idx -= 1
            self._load()

    def _download(self):
        dest = self._log.export_to_downloads()
        messagebox.showinfo("Downloaded",
                            f"Complete CSV saved to:\n{dest}")

    def mainloop(self):
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        super().mainloop()

    def _on_close(self):
        self._player.stop()
        self.destroy()


if __name__ == "__main__":
    app = SetupScreen()
    app.mainloop()