import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def inspect_3mf(file_path):
    print(f"Inspecting 3MF structure: {file_path}")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            print(f"Contents:")
            for name in z.namelist():
                print(f" - {name}")
                if name.endswith(".model"):
                    print(f"   Parsing {name}...")
                    with z.open(name) as f:
                        tree = ET.parse(f)
                        root = tree.getroot()
                        # Remove namespace for easier searching
                        for elem in root.iter():
                            if '}' in elem.tag:
                                elem.tag = elem.tag.split('}', 1)[1]
                        
                        resources = root.find('resources')
                        if resources is not None:
                            objects = resources.findall('object')
                            print(f"   Found {len(objects)} objects in resources.")
                            for obj in objects:
                                obj_id = obj.get('id')
                                obj_type = obj.get('type')
                                mesh = obj.find('mesh')
                                v_count = len(mesh.find('vertices').findall('vertex')) if mesh is not None and mesh.find('vertices') is not None else 0
                                t_count = len(mesh.find('triangles').findall('triangle')) if mesh is not None and mesh.find('triangles') is not None else 0
                                print(f"    - Object ID={obj_id}, Type={obj_type}, V={v_count}, T={t_count}")
                            
                        build = root.find('build')
                        if build is not None:
                            items = build.findall('item')
                            print(f"   Found {len(items)} items in build.")
                            for item in items:
                                print(f"    - Item ObjectID={item.get('objectid')}, Transform={item.get('transform')}")
                        else:
                            print("   [WARN] No build section found!")

    except Exception as e:
        print(f"Error reading 3MF: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_3mf(sys.argv[1])
    else:
        print("Usage: python inspect_3mf.py <file>")
