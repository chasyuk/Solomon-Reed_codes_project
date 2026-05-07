import qrcode

def generate_qr_code(message, filename="my_qrcode.png"):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )

    qr.add_data(message)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    img.save(filename)
    print(f"QR-код збережено як '{filename}'!")
    return filename

data = "Never gonna give you up..."
generate_qr_code(data, "gen.jpg")
