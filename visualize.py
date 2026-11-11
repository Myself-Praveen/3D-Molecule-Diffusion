"""Custom Visualizer for 3D Diffusion Outputs.
Generates a 2D grid PNG and a beautiful interactive 3D HTML viewer.
"""

import argparse
import base64
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw

def generate_2d_grid(sdf_path: str, output_png: str, max_mols: int = 50):
    """Generates a 2D grid image of the first N molecules."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    mols = []
    for i, mol in enumerate(suppl):
        if i >= max_mols:
            break
        if mol is not None:
            # We compute 2D coords just for the drawing so it lays flat and pretty
            try:
                Chem.rdDepictor.Compute2DCoords(mol)
                mols.append(mol)
            except Exception:
                pass
                
    if not mols:
        print("No valid molecules found to draw.")
        return

    # Draw grid
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), 
                               useSVG=False, returnPNG=False)
    img.save(output_png)
    print(f"Success! Saved 2D Grid Image to: {output_png}")


def generate_3d_html(sdf_path: str, output_html: str):
    """Generates an HTML file with an embedded 3Dmol.js viewer."""
    with open(sdf_path, "r") as f:
        sdf_data = f.read()

    # Escape the SDF data to embed it safely in Javascript
    sdf_b64 = base64.b64encode(sdf_data.encode("utf-8")).decode("utf-8")

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>3D Molecule Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: white; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }}
        #viewer-container {{ flex-grow: 1; position: relative; }}
        .controls {{ padding: 20px; background-color: #2d2d2d; display: flex; justify-content: center; align-items: center; gap: 20px; box-shadow: 0 -2px 10px rgba(0,0,0,0.5); z-index: 10; }}
        button {{ padding: 10px 25px; font-size: 16px; cursor: pointer; background-color: #007acc; color: white; border: none; border-radius: 5px; transition: background 0.2s; }}
        button:hover {{ background-color: #005999; }}
        #info {{ font-size: 18px; font-weight: bold; width: 150px; text-align: center; }}
    </style>
</head>
<body>
    <div id="viewer-container"></div>
    <div class="controls">
        <button onclick="prevMol()">&#8592; Previous</button>
        <div id="info">Molecule 1</div>
        <button onclick="nextMol()">Next &#8594;</button>
    </div>

    <script>
        // Decode the base64 SDF data
        const sdfData = atob("{sdf_b64}");
        
        // Split SDF by the '$$$$' delimiter and trim whitespace to fix SDF header alignment
        const sdfModels = sdfData.split("$$$$").map(m => m.trim()).filter(m => m.length > 0).map(m => m + "\\n$$$$");
        let currentIndex = 0;
        
        // Initialize 3Dmol viewer
        let viewer = $3Dmol.createViewer("viewer-container", {{
            backgroundColor: "#1e1e1e"
        }});

        function showMolecule(index) {{
            if (index < 0 || index >= sdfModels.length) return;
            viewer.clear();
            viewer.addModel(sdfModels[index], "sdf");
            viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
            viewer.zoomTo();
            viewer.render();
            document.getElementById("info").innerText = "Molecule " + (index + 1) + " of " + sdfModels.length;
        }}

        function nextMol() {{
            if (currentIndex < sdfModels.length - 1) {{
                currentIndex++;
                showMolecule(currentIndex);
            }}
        }}

        function prevMol() {{
            if (currentIndex > 0) {{
                currentIndex--;
                showMolecule(currentIndex);
            }}
        }}

        // Render first molecule on load
        showMolecule(currentIndex);
        
        // Keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowRight') nextMol();
            if (event.key === 'ArrowLeft') prevMol();
        }});
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Success! Saved Interactive 3D Viewer to: {output_html}")
    print(f"   -> Double click this file in File Explorer to open it in your browser!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=str, required=True, help="Path to SDF file")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save visualizer files")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    generate_2d_grid(args.sdf, str(out_path / "molecules_grid.png"))
    generate_3d_html(args.sdf, str(out_path / "3D_viewer.html"))
