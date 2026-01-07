import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import os
import threading
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# -------------------------
# Definicja modelu (Final_Model)
# -------------------------
class Final_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(55296, 256)   # (24x24x96)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.fc2(F.relu(self.fc1(x)))
        return x

# -------------------------
# Konfiguracja
# -------------------------
MODEL_PATH = "pth_brain_tumor_detection_model.pth" # Ścieżka do pliku modelu
REPORTS_DIR = os.path.join(os.getcwd(), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------------
# Dodanie modelu
# -------------------------
model = None
try:
    model = Final_Model()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Warning: failed to load model:", e)

# -------------------------
# UI
# -------------------------
class AppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Scanner")
        
        try:
            root.state("zoomed")
        except Exception:
            pass

        
        self.image_folder = None
        self.image_list = []             
        self.analysis_results = []       
        self.current_index = 0
        self.tk_image = None
        self.analysis_thread = None

        
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        
        left = tk.Frame(root, padx=8, pady=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        
        controls = tk.Frame(left)
        controls.grid(row=0, column=0, sticky="ew", pady=(0,8))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.btn_folder = tk.Button(controls, text="Wybierz folder (Scan Folder)", command=self.load_folder)
        self.btn_folder.grid(row=0, column=0, sticky="ew", padx=4)

        self.btn_generate_pdf = tk.Button(controls, text="Generuj PDF (Report)", command=self.generate_report)
        self.btn_generate_pdf.grid(row=0, column=1, sticky="ew", padx=4)

        
        self.status_label = tk.Label(left, text="Folder: --    Images: 0", anchor="w")
        self.status_label.grid(row=1, column=0, sticky="ew")

        
        preview_frame = tk.Frame(left, bg="#f1f1f1", relief="groove", bd=2)
        preview_frame.grid(row=2, column=0, sticky="nsew", pady=(8,0))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.image_label = tk.Label(preview_frame, text="Podgląd obrazu", bg="#f1f1f1")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

       
        nav = tk.Frame(left)
        nav.grid(row=3, column=0, pady=8, sticky="ew")
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        nav.columnconfigure(2, weight=1)

        preview_frame.grid_propagate(False)
        preview_frame.config(width=350, height=300)   


        self.prev_btn = tk.Button(nav, text="<< Poprzedni", command=self.prev_image)
        self.prev_btn.grid(row=0, column=0, sticky="ew", padx=6)

        self.info_btn = tk.Button(nav, text="Pokaż szczegóły", command=self.show_current_details)
        self.info_btn.grid(row=0, column=1, sticky="ew", padx=6)

        self.next_btn = tk.Button(nav, text="Następny >>", command=self.next_image)
        self.next_btn.grid(row=0, column=2, sticky="ew", padx=6)

        
        right = tk.Frame(root, padx=8, pady=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        
        lbl = tk.Label(right, text="Wyniki analizy (kliknij aby przejść)", font=("Segoe UI", 11, "bold"))
        lbl.grid(row=0, column=0, sticky="w")

        
        self.results_box = ScrolledText(right, width=40, height=20)
        self.results_box.grid(row=1, column=0, sticky="nsew", pady=(6,6))
        self.results_box.config(state="disabled")
        self.results_box.bind("<Button-1>", self.on_results_click)

       
        bottom = tk.Frame(right)
        bottom.grid(row=2, column=0, sticky="ew", pady=(6,0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=0)

        self.threshold_label = tk.Label(bottom, text="Próg (show images with prob >=):")
        self.threshold_label.grid(row=0, column=0, sticky="w")

        self.threshold_var = tk.DoubleVar(value=0.60)
        self.threshold_entry = tk.Entry(bottom, textvariable=self.threshold_var, width=6)
        self.threshold_entry.grid(row=0, column=1, sticky="e")

        
        self._write_results_text("Wybierz folder, analiza rozpocznie się automatycznie.\n")

    # -------------------------
    # Funkcja rezultatu
    # -------------------------
    def _write_results_text(self, text, clear=False):
        self.results_box.config(state="normal")
        if clear:
            self.results_box.delete("1.0", tk.END)
        self.results_box.insert(tk.END, text)
        self.results_box.config(state="disabled")

    # -------------------------
    # Ładowanie folderu + autoanaliza
    # -------------------------
    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
        if not files:
            messagebox.showwarning("Brak obrazów", "Wybrany folder nie zawiera obrazów (png/jpg/jpeg/bmp).")
            return

        self.image_folder = folder
        self.image_list = files
        self.analysis_results = []
        self.current_index = 0
        self.status_label.config(text=f"Folder: {os.path.basename(folder)}    Images: {len(self.image_list)}")

        self.show_thumbnail(self.image_list[0])

        self._write_results_text("Analiza rozpoczęta automatycznie...\n", clear=True)
        self.analysis_thread = threading.Thread(target=self._analyze_folder_worker, daemon=True)
        self.analysis_thread.start()

    # -------------------------
    # Obraz
    # -------------------------
    def show_thumbnail(self, path):
        try:
            img = Image.open(path).convert("RGB")

            fixed_height = 500

            w, h = img.size
            scale = fixed_height / h
            new_w = int(w * scale)
            new_h = fixed_height

            img = img.resize((new_w, new_h), Image.LANCZOS)

            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image, text="")

        except Exception as e:
            self.image_label.config(image="", text=f"Could not show image:\n{e}")


    # -------------------------
    # analiza folderu (worker thread)
    # -------------------------
    def _analyze_folder_worker(self):
        if not self.image_list:
            self.root.after(0, lambda: messagebox.showwarning("Brak obrazów", "Brak obrazów do analizy."))
            return

        if model is None:
            self.root.after(0, lambda: messagebox.showerror("Brak modelu", "Model nie został załadowany. Sprawdź MODEL_PATH."))
            return

        # zmatchowanie transformacji użytej podczas treningu
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((196, 196)),
            transforms.ToTensor()
        ])

        results = []
        for path in self.image_list:
            try:
                img = Image.open(path).convert("L")
                tensor_img = transform(img).unsqueeze(0)  # [1,1,196,196]
                with torch.no_grad():
                    out = model(tensor_img)
                    prob = float(torch.sigmoid(out).item())
                label = "Tumor" if prob >= 0.5 else "Healthy"
                certainty = prob if prob >= 0.5 else (1 - prob)
                results.append({"path": path, "prob": prob, "label": label, "certainty": certainty})
            except Exception as e:
                results.append({"path": path, "prob": None, "label": "Error", "certainty": 0, "error": str(e)})

        # sortowanie wyników
        results.sort(key=lambda x: x["prob"] or 0, reverse=True)
        self.analysis_results = results

        # index startu po progu
        try:
            thr = float(self.threshold_var.get())
        except Exception:
            thr = 0.6
        start_idx = 0
        for i, it in enumerate(self.analysis_results):
            if (it["prob"] or 0) >= thr:
                start_idx = i
                break
        self.current_index = start_idx

        # aktualizacja UI w głównym wątku
        self.root.after(10, self._update_results_ui)

    # -------------------------
    # Aktualizacja wyników
    # -------------------------
    def _update_results_ui(self):
        self.results_box.config(state="normal")
        self.results_box.delete("1.0", tk.END)

        for i, r in enumerate(self.analysis_results):
            name = os.path.basename(r["path"])
            prob_text = f"{(r['prob']*100):.1f}%" if r["prob"] is not None else "ERR"
            label = r["label"]
            line = f"{i+1:03d}. {name} — {label} ({prob_text})\n"
            self.results_box.insert(tk.END, line)

        self.results_box.tag_configure("sel_line", background="#ffd", foreground="#000")
        self.results_box.tag_remove("sel_line", "1.0", tk.END)

        self.results_box.config(state="disabled")

        if self.analysis_results:
            self.show_current_image()
        else:
            messagebox.showinfo("Analiza", "Brak wyników analizy.")

    # -------------------------
    # Pokazywanie wyników
    # -------------------------
    def show_current_image(self):
        if not self.analysis_results:
            return
        r = self.analysis_results[self.current_index]
        self.show_thumbnail(r["path"])
        prob_text = f"{(r['prob']*100):.1f}%" if r["prob"] is not None else "ERR"
        self.status_label.config(text=f"Folder: {os.path.basename(self.image_folder)}    Showing: {os.path.basename(r['path'])}    Prob: {prob_text}")

        # highlit wyniku obrazu
        line_no = self.current_index + 1
        start = f"{line_no}.0"
        end = f"{line_no}.end"
        self.results_box.config(state="normal")
        self.results_box.tag_remove("sel_line", "1.0", tk.END)
        try:
            self.results_box.tag_add("sel_line", start, end)
            self.results_box.see(start)
        except Exception:
            pass
        self.results_box.config(state="disabled")

    # -------------------------
    # Nawigacja między wynikami obrazów
    # -------------------------
    def on_results_click(self, event):
        try:
            index = self.results_box.index(f"@{event.x},{event.y}")
            line_no = int(float(index))
            if 1 <= line_no <= len(self.analysis_results):
                self.current_index = line_no - 1
                self.show_current_image()
        except Exception:
            pass

    # -------------------------
    # Nawigacja między obrazami
    # -------------------------
    def next_image(self):
        if not self.analysis_results:
            return
        self.current_index = (self.current_index + 1) % len(self.analysis_results)
        self.show_current_image()

    def prev_image(self):
        if not self.analysis_results:
            return
        self.current_index = (self.current_index - 1) % len(self.analysis_results)
        self.show_current_image()

    # -------------------------
    # Szczegóły do obrazu
    # -------------------------
    def show_current_details(self):
        if not self.analysis_results:
            return
        r = self.analysis_results[self.current_index]
        prob_text = f"{(r['prob']*100):.2f}%" if r["prob"] is not None else "ERR"
        msg = (
            f"Plik: {os.path.basename(r['path'])}\n"
            f"Etykieta: {r['label']}\n"
            f"Prawdopodobieństwo guza: {prob_text}\n"
            f"Pełna ścieżka: {r['path']}"
        )
        messagebox.showinfo("Szczegóły obrazu", msg)

    # -------------------------
    # Generacja raportu
    # -------------------------
    def generate_report(self):
        if not self.analysis_results:
            messagebox.showwarning("Brak wyników", "Najpierw przeprowadź analizę folderu.")
            return

        # Nazwa bazowa — raport + folder + data
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"raport_{os.path.basename(self.image_folder)}_{date_str}.pdf"
        save_path = os.path.join(REPORTS_DIR, base_name)

        try:
            self._create_pdf(save_path)
            messagebox.showinfo("Sukces", f"Raport zapisano:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać raportu:\n{e}")


    def _create_pdf(self, path):
        c = canvas.Canvas(path, pagesize=A4)
        width, height = A4

        # Nagłówek
        t = c.beginText(2*cm, height - 2*cm)
        t.setFont("Helvetica-Bold", 14)
        t.textLine("Raport analizy skanów mózgu")
        t.moveCursor(0, 14)

        # Informacje o pliku i dacie
        t.setFont("Helvetica", 11)
        t.moveCursor(0, 12)
        t.textLine(f"Folder: {os.path.basename(self.image_folder)}")
        t.textLine(f"Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        t.moveCursor(0, 8)
        t.textLine("Wyniki analizy:")

        # Wyniki
        t.setFont("Helvetica", 10)
        if not self.analysis_results:
            t.textLine("Brak danych analizy")
        else:
            for r in self.analysis_results:
                name = os.path.basename(r['path'])
                prob = f"{(r['prob']*100):.1f}%" if r['prob'] is not None else "ERR"
                t.textLine(f"{name} — {r['label']} ({prob})")

        # Informacja o modelu
        t.moveCursor(0, 10)
        t.textLine("Model: Final_Model (PyTorch)")

        c.drawText(t)
        c.showPage()
        c.save()

# -------------------------
# Start
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()
