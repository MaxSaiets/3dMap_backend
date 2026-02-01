
import trimesh
import numpy as np
import os
import sys

# Add directory to path to import services
sys.path.append(os.getcwd())

from services.model_exporter import export_3mf

def test_3mf_export():
    print("Testing 3MF export...")
    
    # Create simple box
    mesh1 = trimesh.creation.box(extents=[10, 10, 10])
    mesh1.visual.face_colors = [255, 0, 0, 255]
    
    # Create simple sphere at LARGE OFFSET
    mesh2 = trimesh.creation.icosphere(radius=5)
    # Simulate large offset (e.g. 500 units) to check centering
    mesh2.apply_translation([500.0, 500.0, 0.0])
    mesh2.visual.face_colors = [0, 255, 0, 255]
    
    items = [
        ("Box", mesh1),
        ("Sphere", mesh2)
    ]
    
    filename = "test_export.3mf"
    if os.path.exists(filename):
        os.remove(filename)
        
    try:
        export_3mf(filename, items, model_size_mm=50)
        print("Export finished.")
        
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"File created: {filename}, size: {size} bytes")
            if size > 1000:
                print("SUCCESS: File seems valid.")
            else:
                print("FAILURE: File too small.")
        else:
            print("FAILURE: File not created.")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3mf_export()
