import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

from reed_solomon import ReedSolomon
from decode import DecodeReedSolomon
from qr_decode import (
    extract_aligned_qr, unmask_qr_matrix,
    extract_data_bits,
    heal_qr_data, extract_and_decode_text
)


class RSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RS Simulator")
        self.root.geometry("600x400")
        self._qr_path = None

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        nb.add(self._text_tab(nb), text="Текст")
        nb.add(self._qr_tab(nb),   text="QR-код")

    #Вкладка1
    def _text_tab(self, parent):
        f = tk.Frame(parent)

        tk.Label(f, text="Повідомлення:").pack(pady=(10, 0))
        self.entry_in = tk.Entry(f, width=55)
        self.entry_in.insert(0, "Hello")
        self.entry_in.pack()

        tk.Button(f, text="Закодувати", command=self.step_encode).pack(pady=5)

        tk.Label(f, text="Кодове слово (змініть для симуляції помилки):").pack()
        self.entry_cw = tk.Entry(f, width=55, fg="blue")
        self.entry_cw.pack()

        tk.Button(f, text="Виправити помилки", command=self.step_decode).pack(pady=5)

        self.lbl_result = tk.Label(f, text="—", font=("Courier", 10, "bold"))
        self.lbl_result.pack()
        self.lbl_pos = tk.Label(f, text="", fg="orange")
        self.lbl_pos.pack()

        return f

    def step_encode(self):
        text = self.entry_in.get()
        rs = ReedSolomon(8, 251)
        for i, ch in enumerate(text):
            rs[i] = ord(ch)
        rs.encode()

        bits = " ".join(f"{rs[i].coeffs:08b}" for i in range(len(rs.poly)))
        self.entry_cw.delete(0, tk.END)
        self.entry_cw.insert(0, bits)
        self.lbl_result.config(text="Закодовано", fg="black")
        self._rs_encoded = rs


    def step_decode(self):
        try:
            received = ReedSolomon(8, 251)
            for i, b in enumerate(self.entry_cw.get().split()):
                received[i] = int(b, 2)

            fixed = DecodeReedSolomon(received).decode()
            fixed.get_original()

            text = "".join(chr(fixed[i].coeffs) for i in range(len(self.entry_in.get())))
            self.lbl_result.config(text=f"Результат: {text}", fg="green")
        except ValueError as e:
            self.lbl_result.config(text=str(e), fg="red")
        except Exception as e:
            messagebox.showerror("Помилка декодування", str(e))

    # Вкладка 2
    def _qr_tab(self, parent):
        f = tk.Frame(parent)

        tk.Button(f, text="Відкрити та декодувати", command=self.load_and_decode).pack(pady=12)

        self.lbl_preview = tk.Label(f, text="", fg="gray", relief=tk.GROOVE)
        self.lbl_preview.pack()

        self.lbl_qr_result = tk.Label(f, text="", font=("Courier", 11, "bold"),
                                    wraplength=450)
        self.lbl_qr_result.pack(pady=10)
        return f

    def load_and_decode(self):
        path = filedialog.askopenfilename()

        img = Image.open(path)
        img.thumbnail((180, 180))
        photo = ImageTk.PhotoImage(img)
        self.lbl_preview.config(image=photo, text="")
        self.lbl_preview.image = photo

        aligned = extract_aligned_qr(path)
        print(f"aligned: {aligned}")

        for mask in range(8):
            unmasked  = unmask_qr_matrix(aligned, mask)
            bits = extract_data_bits(unmasked)
            codewords = [
                int("".join(map(str, bits[i:i+8])), 2)
                for i in range(0, len(bits), 8)
                if len(bits[i:i+8]) == 8
]
            clean = heal_qr_data(codewords)
            if clean is None:
                continue
            try:
                text = extract_and_decode_text(clean)
                print(f"Текст: {text}")
                self.lbl_qr_result.config(text=f"{text}", fg="green")
                return
            except (ValueError, IndexError) as e:
                print(f"Помилка: {e}")
                continue

        self.lbl_qr_result.config(text="Не вдалося декодувати", fg="red")




if __name__ == "__main__":
    root = tk.Tk()
    RSApp(root)
    root.mainloop()
