import io
import re
from typing import Any
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.axes import Axes
from pathlib import Path


def text_to_image(text: str, scale=1.6):
    lines = text.split('\n')
    max_line_len = max(len(l) for l in lines) if lines else 1

    width = int(max(120, max_line_len * 6.2) * scale)
    height = int(max(120, len(lines) * 13) * scale)
    
    font_size = int(8 * scale) 
    
    text_spans = ""
    for line in lines:
        text_spans += f'<tspan x="10" dy="1.05em">{line}</tspan>'
    
    svg = \
    f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <rect width="100%" height="100%" fill="white"/>
    <text x="10" y="8" font-family="DejaVu Sans, Liberation Sans, sans-serif" font-size="{font_size}px" fill="#333333">
    {text_spans}
    </text>
    </svg>"""
    return svg


def smiles_to_image(smiles: str | None, scale: float) -> str | None:
    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(smiles) or Chem.Mol()

    d2d = Draw.MolDraw2DSVG(int(150 * scale), int(150 * scale))
    d2d.drawOptions().legendFontSize = 40
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    
    return d2d.GetDrawingText()


def ax_to_image(ax: Axes) -> str:
    buf = io.StringIO()
    ax.figure.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.1)
    return buf.getvalue()


def make_vector_grid(columns: list, bg_color: str = "white") -> str:
    if not columns:
        return f'<svg xmlns="http://www.w3.org/2000/svg" fill="{bg_color}"></svg>'
    
    if isinstance(columns, str):
        columns = [[columns]]

    if not isinstance(columns[0], list):
        columns = [columns]

    nr_of_columns = len(columns)
    nr_of_rows = len(columns[0])
    
    svg_elements = []
    for column in columns:
        for item in column:
            svg_elements.append(to_svg_string(item))

    max_widths = [0 for _ in range(nr_of_columns)]
    max_heights = [0 for _ in range(nr_of_rows)]

    for i, svg_str in enumerate(svg_elements):
        col, row = divmod(i, nr_of_rows)
        w, h = get_dims(svg_str)
        max_widths[col] = max(max_widths[col], w)
        max_heights[row] = max(max_heights[row], h)

    total_width = sum(max_widths)
    total_height = sum(max_heights)

    grid_svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">',
        f'<rect width="100%" height="100%" fill="{bg_color}"/>'
    ]

    y_offset = 0
    for r in range(nr_of_rows):
        x_offset = 0
        for c in range(nr_of_columns):
            index = c * nr_of_rows + r
            svg_str = svg_elements[index]

            if svg_str is None:
                x_offset += max_widths[c]
                continue

            w, h = get_dims(svg_str)
            
            padded_w = w + 10
            padded_h = h + 10

            cell_x = x_offset + (max_widths[c] - padded_w) // 2
            cell_y = y_offset + (max_heights[r] - padded_h) // 2
            
            clean_content = re.sub(r'<\?xml[^>]*\?>', '', svg_str)
            clean_content = re.sub(r'<!DOCTYPE[^>]*>', '', clean_content, flags=re.IGNORECASE)
            clean_content = re.sub(r'<svg[^>]*>', strip_fixed_dims, clean_content, count=1)
            
            grid_svg.append(f'<svg x="{cell_x}" y="{cell_y}" width="{padded_w}" height="{padded_h}" style="overflow: visible;">')
            grid_svg.append(clean_content)
            grid_svg.append('</svg>')

            x_offset += max_widths[c] + 10 
        y_offset += max_heights[r]

    grid_svg.append('</svg>')
    return "\n".join(grid_svg)


def strip_fixed_dims(match):
    tag = match.group(0)
    tag = re.sub(r'\bwidth=["\'][^"\']*["\']', '', tag)
    tag = re.sub(r'\bheight=["\'][^"\']*["\']', '', tag)
    return tag


def get_dims(svg_str):
    if svg_str is None:
        return 0, 0
    
    viewbox_match = re.search(r'viewBox=["\']\s*0\s+0\s+([\d\.]+)\s+([\d\.]+)["\']', svg_str)
    
    if viewbox_match:
        w_val = int(float(viewbox_match.group(1)))
        h_val = int(float(viewbox_match.group(2)))
        return w_val, h_val

    w_match = re.search(r'width=["\']([\d\.]+)(?:px|pt)?["\']', svg_str)
    h_match = re.search(r'height=["\']([\d\.]+)(?:px|pt)?["\']', svg_str)

    if w_match and h_match:
        return int(float(w_match.group(1))), int(float(h_match.group(1)))
    
    return 300, 300


def to_svg_string(img: Any) -> str | None:
    if img is None:
        return None
    
    if isinstance(img, str):
        clean_str = img.strip()
        if clean_str.lower().startswith("<svg") or "<svg" in clean_str:
            return clean_str
        return img
    
    if hasattr(img, "savefig"): 
        buf = io.StringIO()
        img.savefig(buf, format="svg")
        return buf.getvalue()
    
    if hasattr(img, "extract"):  
        return str(img)
        
    raise TypeError(f"Raster types like arrays/PIL images cannot be implicitly converted to SVG: {type(img)}")


def save_svg(svg: str, path: str | Path) -> None:
    path = str(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)