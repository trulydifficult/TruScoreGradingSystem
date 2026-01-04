import os
from fontTools.ttLib import TTFont

def get_font_family(font_path):
    try:
        font = TTFont(font_path)
        # Name ID 1 = Font Family
        name = font['name'].getName(1, 3, 1) or font['name'].getName(1, 1, 0)
        return str(name) if name else "Unknown"
    except:
        return "Unreadable / Not a font"

def scan_fonts(folder_path, output_file="/home/dewster/Projects/Vanguard/src/shared/essentials/appfonts/font_families.txt"):
    font_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.ttf', '.otf'))]

    with open(output_file, "w", encoding="utf-8") as f:
        for font_file in font_files:
            font_path = os.path.join(folder_path, font_file)
            family = get_font_family(font_path)
            f.write(f"{font_file} -> {family}\n")

    print(f"Done. Output saved to: {output_file}")

# Example use:
scan_fonts(r"/home/dewster/Projects/Vanguard/src/shared/essentials/appfonts/")
