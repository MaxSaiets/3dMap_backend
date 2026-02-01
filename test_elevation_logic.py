
import sys
import os
import numpy as np
import time

# Add cwd to path
sys.path.append(os.getcwd())

from services.heightmap import get_elevation_data
from services.water_processor import process_water_surface

def test_elevation_logic():
    print("Testing elevation logic...")
    
    # Create dummy X, Y
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    X, Y = np.meshgrid(x, y)
    
    # Dummy latlon_bbox (should be ignored if we don't have API keys or tiles, fallback to synthetic)
    latlon_bbox = (50.45, 50.44, 30.52, 30.51) # Kyiv
    
    print("Calling get_elevation_data with synthetic/fallback expectations...")
    try:
        Z, zmin = get_elevation_data(
            X, Y, latlon_bbox, 
            z_scale=1.5,
            terrarium_zoom=10, 
            elevation_ref_m=None # Local relative mode
        )
        print(f"Z shape: {Z.shape}")
        print(f"Z range: {np.min(Z):.2f} to {np.max(Z):.2f}")
        print(f"zmin: {zmin}")
        
        if np.all(Z == 0):
            print("WARNING: Z is all zeros!")
        else:
            print("Z has variation. Good.")
            
    except Exception as e:
        print(f"get_elevation_data failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_elevation_logic()
