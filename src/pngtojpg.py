from PIL import Image
im = Image.open(r"2.png")
bg = Image.new("RGB", im.size, (255,255,255))
bg.paste(im,im)
bg.save(r"jk2.jpg")
