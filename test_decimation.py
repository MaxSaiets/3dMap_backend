
import trimesh
import numpy as np

def test_decimation():
    print("Testing decimation capabilities...")
    
    # Create a high-res sphere
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    print(f"Original: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    
    try:
        # Decimate to 1000 faces
        simplified = mesh.simplify_quadratic_decimation(1000)
        print(f"Simplified: {len(simplified.faces)} faces, {len(simplified.vertices)} vertices")
        
        if len(simplified.faces) <= 1000:
            print("SUCCESS: Decimation worked.")
        else:
            print("FAILURE: Did not reduce face count enough.")
            
    except AttributeError:
        print("ERROR: simplify_quadratic_decimation not available.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_decimation()
