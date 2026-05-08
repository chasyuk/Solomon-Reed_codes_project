"""
Entry point for the Reed–Solomon QR lab.

Run from the project root:

    python main.py                          # open GUI

    python main.py encode "text" -o qr.png
    python main.py damage "text" -e 10 -s 42
    python main.py decode qr.png
    python main.py demo  "text" -e 10 --save

    python main.py --help                   # list all commands
    python main.py <command> --help         # help for a specific command
"""

from gui.rs_interface import _cli, ReedSolomonGUI

if __name__ == "__main__":
    result = _cli()
    if result and result[0] == "gui":
        app = ReedSolomonGUI()
        prefill = result[1]
        if prefill:
            app.ent_message.delete(0, "end")
            app.ent_message.insert(0, prefill)
        app.mainloop()
