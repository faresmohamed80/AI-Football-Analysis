"""
Football Action Labeler  — Tkinter GUI
=======================================
1. Open video → File dialog
2. Drag the SEEK SLIDER to jump to any moment
3. Press MARK START  then seek forward  then MARK END
4. (Optional) draw BBox on the video by mouse drag
5. Click a label button  → clip saved automatically
"""

import cv2, os, threading, time
from datetime import datetime
from collections import deque

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "action_dataset")

LABELS = [
    ("PASS",                     "#00DD00"),
    ("SHOT",                     "#FF3333"),
    ("HEADER",                   "#FF8800"),
    ("CROSS",                    "#FF00CC"),
    ("HIGH_PASS",                "#00CCCC"),
    ("THROW_IN",                 "#FFDD00"),
    ("BALL_PLAYER_BLOCK",        "#6688FF"),
    ("PLAYER_SUCCESSFUL_TACKLE", "#00CC66"),
]
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]

THUMB_W, THUMB_H = 100, 65
MAX_REVIEW       = 8
VIDEO_MAX_W      = 860
VIDEO_MAX_H      = 560


# ── App ───────────────────────────────────────────────────────────────────────
class Labeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚽ Football Action Labeler")
        self.configure(bg="#12121C")
        self.resizable(True, True)

        # Video state
        self.cap         = None
        self.total       = 0
        self.fps         = 25.0
        self.orig_w      = 1
        self.orig_h      = 1
        self.disp_w      = 1
        self.disp_h      = 1
        self.scale       = 1.0
        self.frame_idx   = 0
        self.cur_frame   = None     # np array (orig size)
        self.playing     = False
        self.speed_idx   = 2        # 1.0x
        self._play_job   = None

        # Annotation state
        self.start_f     = None
        self.end_f       = None
        self.bbox        = None     # (x1,y1,x2,y2) in DISPLAY coords
        self.drag_start  = None
        self.saved       = []

        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg="#0A0A14", pady=4)
        top.pack(fill="x")
        tk.Button(top, text="📂  Open Video", command=self.open_video,
                  bg="#1E90FF", fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=14, pady=6).pack(side="left", padx=8)
        self.file_lbl = tk.Label(top, text="No file loaded",
                                 bg="#0A0A14", fg="#888", font=("Segoe UI", 9))
        self.file_lbl.pack(side="left", padx=8)

        # ── Main content ─────────────────────────────────────────────────────
        content = tk.Frame(self, bg="#12121C")
        content.pack(fill="both", expand=True)

        # Left  = video + controls
        left = tk.Frame(content, bg="#12121C")
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        # Video canvas
        self.canvas = tk.Canvas(left, bg="#000", cursor="crosshair",
                                width=VIDEO_MAX_W, height=VIDEO_MAX_H,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._drag_start)
        self.canvas.bind("<B1-Motion>",       self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self._img_ref = None   # keep reference

        # Seek slider
        seek_row = tk.Frame(left, bg="#12121C")
        seek_row.pack(fill="x", pady=(4, 0))
        self.seek_var = tk.IntVar(value=0)
        self.seek_bar = ttk.Scale(seek_row, from_=0, to=1000,
                                  orient="horizontal", variable=self.seek_var,
                                  command=self._on_seek)
        self.seek_bar.pack(fill="x", padx=4)

        # Frame label under slider
        self.frame_lbl = tk.Label(left, text="Frame: 0 / 0   |   00:00.0",
                                  bg="#12121C", fg="#888", font=("Consolas", 9))
        self.frame_lbl.pack()

        # Playback controls
        ctrl = tk.Frame(left, bg="#12121C", pady=4)
        ctrl.pack()

        btn_cfg = dict(bg="#1A1A2E", fg="white", relief="flat",
                       font=("Segoe UI", 10), padx=10, pady=5, cursor="hand2")

        tk.Button(ctrl, text="⏮ –30",  command=lambda: self.jump(-30),  **btn_cfg).pack(side="left", padx=3)
        tk.Button(ctrl, text="◀ –1",   command=lambda: self.step(-1),   **btn_cfg).pack(side="left", padx=3)
        self.play_btn = tk.Button(ctrl, text="▶  Play",
                                  command=self.toggle_play,
                                  bg="#1E90FF", fg="white", relief="flat",
                                  font=("Segoe UI", 10, "bold"),
                                  padx=14, pady=5, cursor="hand2")
        self.play_btn.pack(side="left", padx=4)
        tk.Button(ctrl, text="▶ +1",   command=lambda: self.step(+1),   **btn_cfg).pack(side="left", padx=3)
        tk.Button(ctrl, text="+30 ⏭",  command=lambda: self.jump(+30),  **btn_cfg).pack(side="left", padx=3)

        # Speed buttons
        spd_frame = tk.Frame(left, bg="#12121C", pady=4)
        spd_frame.pack()
        tk.Label(spd_frame, text="Speed:", bg="#12121C", fg="#888",
                 font=("Segoe UI", 9)).pack(side="left")
        self.spd_btns = []
        for i, s in enumerate(SPEEDS):
            b = tk.Button(spd_frame, text=f"{s}x", command=lambda i=i: self.set_speed(i),
                          bg="#222232", fg="#aaa", relief="flat",
                          font=("Segoe UI", 9), padx=8, pady=3, cursor="hand2")
            b.pack(side="left", padx=2)
            self.spd_btns.append(b)
        self._highlight_speed()

        # Mark buttons
        mark_row = tk.Frame(left, bg="#12121C", pady=4)
        mark_row.pack()
        self.start_btn = tk.Button(mark_row, text="📍 MARK START",
                                   command=self.mark_start,
                                   bg="#006622", fg="white", relief="flat",
                                   font=("Segoe UI", 10, "bold"),
                                   padx=14, pady=6, cursor="hand2")
        self.start_btn.pack(side="left", padx=6)
        self.end_btn = tk.Button(mark_row, text="📍 MARK END",
                                 command=self.mark_end,
                                 bg="#002299", fg="white", relief="flat",
                                 font=("Segoe UI", 10, "bold"),
                                 padx=14, pady=6, cursor="hand2")
        self.end_btn.pack(side="left", padx=6)
        tk.Button(mark_row, text="✖ Reset",
                  command=self.reset_sel,
                  bg="#442222", fg="#ccc", relief="flat",
                  font=("Segoe UI", 9), padx=10, pady=6, cursor="hand2").pack(side="left", padx=6)

        # Selection info bar
        self.sel_lbl = tk.Label(left, text="START: —    END: —    BBox: none",
                                bg="#0A0A14", fg="#aaa", font=("Consolas", 9),
                                anchor="w", padx=8, pady=4)
        self.sel_lbl.pack(fill="x", pady=(2, 0))

        # Status bar
        self.status = tk.Label(left, text="Open a video to begin",
                               bg="#0A0A14", fg="#1E90FF",
                               font=("Segoe UI", 9), anchor="w", padx=8, pady=4)
        self.status.pack(fill="x")

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(content, bg="#0A0A14", width=220)
        right.pack(side="right", fill="y", padx=(0, 6), pady=6)
        right.pack_propagate(False)

        tk.Label(right, text="SAVE CLIP AS:", bg="#0A0A14", fg="#888",
                 font=("Segoe UI", 8, "bold")).pack(pady=(8, 2))

        for lbl, color in LABELS:
            key_num = str(LABELS.index((lbl, color)) + 1)
            tk.Button(right, text=f"[{key_num}]  {lbl}",
                      command=lambda l=lbl: self.save_clip(l),
                      bg=color, fg="black" if lbl in ("THROW_IN","HIGH_PASS","BALL_PLAYER_BLOCK") else "black",
                      activebackground=color,
                      relief="flat", font=("Segoe UI", 9, "bold"),
                      padx=8, pady=6, cursor="hand2",
                      anchor="w").pack(fill="x", padx=8, pady=2)

        tk.Button(right, text="↩ Undo last clip",
                  command=self.undo_last,
                  bg="#2A1010", fg="#FF6666", relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2").pack(fill="x", padx=8, pady=(6,2))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6, padx=8)

        tk.Label(right, text="SAVED CLIPS:", bg="#0A0A14", fg="#888",
                 font=("Segoe UI", 8, "bold")).pack(pady=(0, 2))

        self.review_frame = tk.Frame(right, bg="#0A0A14")
        self.review_frame.pack(fill="both", expand=True, padx=4)

        # ── Keyboard bindings ────────────────────────────────────────────────
        self.bind("<space>",     lambda e: self.toggle_play())
        self.bind("<Left>",      lambda e: self.step(-1))
        self.bind("<Right>",     lambda e: self.step(+1))
        self.bind("<a>",         lambda e: self.jump(-30))
        self.bind("<d>",         lambda e: self.jump(+30))
        self.bind("<A>",         lambda e: self.jump(-30))
        self.bind("<D>",         lambda e: self.jump(+30))
        self.bind("<s>",         lambda e: self.mark_start())
        self.bind("<e>",         lambda e: self.mark_end())
        self.bind("<r>",         lambda e: self.reset_sel())
        self.bind("<bracketleft>",  lambda e: self.set_speed(max(0, self.speed_idx - 1)))
        self.bind("<bracketright>", lambda e: self.set_speed(min(len(SPEEDS)-1, self.speed_idx + 1)))
        self.bind("<Delete>",    lambda e: self.undo_last())
        self.bind("<BackSpace>", lambda e: self.undo_last())
        for i, (lbl, _) in enumerate(LABELS):
            self.bind(str(i+1), lambda e, l=lbl: self.save_clip(l))

    # ── Video loading ─────────────────────────────────────────────────────────
    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select Match Video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not path: return
        if self.cap: self.cap.release()
        self.cap      = cv2.VideoCapture(path)
        self.total    = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps      = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.orig_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale    = min(1.0, VIDEO_MAX_W / self.orig_w, VIDEO_MAX_H / self.orig_h)
        self.disp_w   = int(self.orig_w * self.scale)
        self.disp_h   = int(self.orig_h * self.scale)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self.seek_bar.config(to=self.total - 1)
        self.frame_idx = 0
        self.start_f = self.end_f = self.bbox = None
        self.file_lbl.config(text=os.path.basename(path))
        self._read_frame(0)
        self._set_status(f"Loaded: {os.path.basename(path)}  ({self.total} frames @ {self.fps:.1f}fps)", "#00DDFF")

    # ── Frame reading ─────────────────────────────────────────────────────────
    def _read_frame(self, idx):
        if not self.cap: return
        idx = max(0, min(self.total - 1, idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = self.cap.read()
        if not ret: return
        self.cur_frame  = frm
        self.frame_idx  = idx
        self._show_frame(frm)
        self._update_ui()

    def _show_frame(self, frm):
        disp = cv2.resize(frm, (self.disp_w, self.disp_h))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(disp)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self._img_ref = imgtk
        # Draw bbox
        if self.bbox:
            x1,y1,x2,y2 = self.bbox
            self.canvas.create_rectangle(x1,y1,x2,y2, outline="#00FF00", width=2, tags="bbox")
        # Selection border
        if self.start_f is not None and self.end_f is not None:
            if self.start_f <= self.frame_idx <= self.end_f:
                self.canvas.create_rectangle(2,2,self.disp_w-2,self.disp_h-2,
                                             outline="#1E90FF", width=4, tags="border")
        elif self.start_f is not None and self.frame_idx >= self.start_f:
            self.canvas.create_rectangle(2,2,self.disp_w-2,self.disp_h-2,
                                         outline="#00CC00", width=4, tags="border")

    def _update_ui(self):
        t = self.frame_idx / self.fps
        m, s = int(t // 60), t % 60
        self.frame_lbl.config(text=f"Frame: {self.frame_idx} / {self.total-1}   |   {m:02d}:{s:05.2f}")
        self.seek_var.set(self.frame_idx)
        # Selection label
        s_str = f"F{self.start_f}" if self.start_f is not None else "—"
        e_str = f"F{self.end_f}"   if self.end_f   is not None else "—"
        b_str = f"({self.bbox[0]},{self.bbox[1]})→({self.bbox[2]},{self.bbox[3]})" if self.bbox else "none"
        self.sel_lbl.config(text=f"START: {s_str}    END: {e_str}    BBox: {b_str}")

    # ── Playback ──────────────────────────────────────────────────────────────
    def toggle_play(self):
        if not self.cap: return
        self.playing = not self.playing
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶  Play",
                             bg="#FF6600" if self.playing else "#1E90FF")
        if self.playing:
            self._schedule_next()

    def _schedule_next(self):
        if not self.playing: return
        speed    = SPEEDS[self.speed_idx]
        interval = max(1, int(1000 / self.fps / speed))
        self._play_job = self.after(interval, self._play_tick)

    def _play_tick(self):
        if not self.playing or not self.cap: return
        nxt = self.frame_idx + 1
        if nxt >= self.total:
            self.playing = False
            self.play_btn.config(text="▶  Play", bg="#1E90FF")
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, nxt)
        ret, frm = self.cap.read()
        if not ret:
            self.playing = False; return
        self.cur_frame = frm
        self.frame_idx = nxt
        self._show_frame(frm)
        self._update_ui()
        self._schedule_next()

    def step(self, delta):
        self.playing = False
        self.play_btn.config(text="▶  Play", bg="#1E90FF")
        self._read_frame(self.frame_idx + delta)

    def jump(self, delta):
        self.playing = False
        self.play_btn.config(text="▶  Play", bg="#1E90FF")
        self._read_frame(self.frame_idx + delta)

    def set_speed(self, idx):
        self.speed_idx = idx
        self._highlight_speed()

    def _highlight_speed(self):
        for i, b in enumerate(self.spd_btns):
            b.config(bg="#1E90FF" if i == self.speed_idx else "#222232",
                     fg="white"   if i == self.speed_idx else "#aaa")

    def _on_seek(self, val):
        if not self.cap: return
        idx = int(float(val))
        if idx != self.frame_idx:
            self.playing = False
            self.play_btn.config(text="▶  Play", bg="#1E90FF")
            self._read_frame(idx)

    # ── Annotations ──────────────────────────────────────────────────────────
    def mark_start(self):
        self.start_f = self.frame_idx
        self.end_f   = None
        self.start_btn.config(bg="#009933")
        self._set_status(f"✓ Start marked → Frame {self.start_f}", "#00DD00")
        self._update_ui()

    def mark_end(self):
        if self.start_f is None:
            self._set_status("⚠ Mark START first!", "#FF4444"); return
        if self.frame_idx <= self.start_f:
            self._set_status("⚠ Move FORWARD before marking END!", "#FF4444"); return
        self.end_f = self.frame_idx
        self.end_btn.config(bg="#0044CC")
        self._set_status(f"✓ End marked → Frame {self.end_f}  ({self.end_f-self.start_f} frames)", "#4488FF")
        self._update_ui()

    def reset_sel(self):
        self.start_f = self.end_f = self.bbox = None
        self.start_btn.config(bg="#006622")
        self.end_btn.config(bg="#002299")
        self._set_status("Selection reset", "#888")
        self._update_ui()

    # ── BBox drawing ──────────────────────────────────────────────────────────
    def _drag_start(self, e):
        self.drag_start = (e.x, e.y)
        self.bbox = None

    def _drag_move(self, e):
        if not self.drag_start: return
        self.canvas.delete("drag_rect")
        self.canvas.create_rectangle(self.drag_start[0], self.drag_start[1],
                                     e.x, e.y, outline="#FFFF00", width=2, tags="drag_rect")

    def _drag_end(self, e):
        if not self.drag_start: return
        x1 = min(self.drag_start[0], e.x); y1 = min(self.drag_start[1], e.y)
        x2 = max(self.drag_start[0], e.x); y2 = max(self.drag_start[1], e.y)
        if x2 - x1 > 10 and y2 - y1 > 10:
            self.bbox = (x1, y1, x2, y2)
            self._set_status(f"✓ BBox set: ({x1},{y1})→({x2},{y2})", "#00CC00")
        else:
            self._set_status("BBox too small — drag again", "#FF8800")
        self.drag_start = None
        self.canvas.delete("drag_rect")
        self._show_frame(self.cur_frame)
        self._update_ui()

    # ── Save ─────────────────────────────────────────────────────────────────
    def save_clip(self, label):
        if not self.cap:
            self._set_status("⚠ No video loaded!", "#FF4444"); return
        if self.start_f is None or self.end_f is None:
            self._set_status("⚠ Mark START and END first!", "#FF4444"); return
        if self.end_f <= self.start_f:
            self._set_status("⚠ END must be after START!", "#FF4444"); return

        out_dir  = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        out_path = os.path.join(out_dir, f"{ts}.mp4")

        # Crop region in original pixel coords
        if self.bbox:
            x1 = int(self.bbox[0] / self.scale); y1 = int(self.bbox[1] / self.scale)
            x2 = int(self.bbox[2] / self.scale); y2 = int(self.bbox[3] / self.scale)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(self.orig_w,x2), min(self.orig_h,y2)
            ow, oh = x2-x1, y2-y1
        else:
            x1=y1=0; x2=self.orig_w; y2=self.orig_h
            ow, oh = self.orig_w, self.orig_h

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.fps, (ow, oh))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_f)
        thumb_np = None
        for _ in range(self.end_f - self.start_f + 1):
            ret, frm = self.cap.read()
            if not ret: break
            crop = frm[y1:y2, x1:x2]
            writer.write(crop)
            if thumb_np is None:
                thumb_np = cv2.resize(crop, (THUMB_W, THUMB_H))
        writer.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

        # Thumb for review
        thumb_img = None
        if thumb_np is not None:
            thumb_img = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(thumb_np, cv2.COLOR_BGR2RGB)))

        self.saved.append({'label': label, 'path': out_path,
                           'thumb': thumb_img, 'fs': self.start_f, 'fe': self.end_f})
        self._set_status(f"✓ Saved [{label}] → {os.path.basename(out_path)}", "#00DD00")
        self.reset_sel()
        self._refresh_review()

    def undo_last(self):
        if not self.saved:
            self._set_status("Nothing to undo", "#888"); return
        last = self.saved.pop()
        try: os.remove(last['path'])
        except: pass
        self._set_status(f"↩ Deleted: {os.path.basename(last['path'])}", "#FF8844")
        self._refresh_review()

    # ── Review panel ─────────────────────────────────────────────────────────
    def _refresh_review(self):
        for w in self.review_frame.winfo_children():
            w.destroy()
        for item in reversed(self.saved[-MAX_REVIEW:]):
            row = tk.Frame(self.review_frame, bg="#0A0A14")
            row.pack(fill="x", pady=2)
            color = dict(LABELS).get(item['label'], "#888")
            if item.get('thumb'):
                lbl = tk.Label(row, image=item['thumb'], bg="#0A0A14")
                lbl.image = item['thumb']
                lbl.pack(side="left", padx=(0,4))
            info = tk.Frame(row, bg="#0A0A14")
            info.pack(side="left", fill="x", expand=True)
            tk.Label(info, text=item['label'], bg="#0A0A14", fg=color,
                     font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x")
            tk.Label(info, text=f"F{item['fs']}–{item['fe']}",
                     bg="#0A0A14", fg="#666", font=("Consolas", 7), anchor="w").pack(fill="x")
            tk.Label(info, text=os.path.basename(item['path'])[:22],
                     bg="#0A0A14", fg="#555", font=("Consolas", 7), anchor="w").pack(fill="x")

    # ── Status ────────────────────────────────────────────────────────────────
    def _set_status(self, msg, color="#aaa"):
        self.status.config(text=msg, fg=color)


if __name__ == "__main__":
    app = Labeler()
    app.mainloop()
    if app.cap:
        app.cap.release()
    print(f"\nDone! {len(app.saved)} clips saved to: {OUTPUT_DIR}")
    for item in app.saved:
        print(f"  [{item['label']}]  {item['path']}")
