from qr_decode import process_qr_pipeline

files = [
    "1.jpg",
    "2_jpg.rf.00c7185f53334d82e2ef67f2cab93604.jpg",
    "2_jpg.rf.55f587ba304a3d26891d8a82902f35e2.jpg",
    "124.jpg","138.jpg","143.jpg","145.jpg","157.jpg",
    "157_jpg.rf.5793b40259f3bfd2cee448341594ff51.jpg",
    "157_jpg.rf.b799b2cf13375399c018deeafc954a2c.jpg",
    "162.jpg","163.jpg","164.jpg","165.jpg","173.jpg","174.jpg",
    "191.jpg","192.jpg","194.jpg","200.jpg","206.jpg","228.jpg","229.jpg",

    "348.jpg","404.jpg","415.jpg","416.jpg","457.jpg","510.jpg","511.jpg",
    "512.jpg","516.jpg","581.jpg","643.jpg","646.jpg","656.jpg","657.jpg",
    "658.jpg","667.jpg","703.jpg","714.jpg","720.jpg","778.jpg",
    "image001.jpg","image002.jpg",

    "image006.jpg","image011.jpg","image019.jpg","image022.jpg","image025.jpg",
    "image027.jpg","image028.jpg","image029.jpg","image037.jpg","image038.jpg",
    "image039.jpg","image040.jpg","image041.jpg","image042.jpg","image043.jpg",
    "image044.jpg","image045.jpg","image046.jpg","image047.jpg","image048.jpg",
    "image049.jpg","image050.jpg",

    "image051.jpg","image052.jpg","image053.jpg","image054.jpg","image055.jpg",
    "image056.jpg","image057.jpg","image058.jpg","image059.jpg","image060.jpg",
    "image061.jpg","image062.jpg","image063.jpg","image064.jpg",

    "image342 (1).jpg","image342 (2).jpg","image342 (3).jpg","image342 (4).jpg",
    "image342 (5).jpg","image342 (6).jpg","image342 (7).jpg","image342 (8).jpg",
    "image342 (9).jpg","image342 (10).jpg",

    "qr2-413_jpg.rf.7ac7b14c1b1faac1c658afcbeff6c84e.jpg",
    "qr2-413_jpg.rf.10f79c520a8c2b3131e6acd513421a0c.jpg",

    "WhatsApp Image 2022-06-29 at 13.33.15 (1).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (1).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (6).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (7).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (8).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (9).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (10).jpg",
    "WhatsApp Image 2022-06-29 at 13.33.16 (11).jpg"
]

for name in files:
    # print(f"\n--- Сканування {name} ---")
    process_qr_pipeline(f"test_data/{name}")
