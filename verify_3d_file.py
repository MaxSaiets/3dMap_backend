import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
import sys

def validate_3mf(file_path):
    print(f"🔍 Аналіз файлу: {file_path}")
    
    if not zipfile.is_zipfile(file_path):
        print("❌ ПОМИЛКА: Це не валідний ZIP-архів (3MF має бути архівом).")
        return

    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            # 1. Перевірка наявності головного файлу моделі
            model_path = '3D/3dmodel.model'
            if model_path not in z.namelist():
                print(f"❌ КРИТИЧНО: Файл {model_path} відсутній в архіві!")
                return

            print("✅ Структура архіву OK")

            # 2. Аналіз XML
            with z.open(model_path) as f:
                xml_content = f.read()
            
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                print(f"❌ КРИТИЧНО: XML файл пошкоджений. Слайсер не зможе його прочитати.")
                print(f"   Деталі: {e}")
                return

            # Простір імен 3MF (обов'язковий для пошуку тегів)
            ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
            
            # 3. Пошук об'єктів та перевірка ID
            objects = root.findall('.//m:object', ns)
            ids = [obj.get('id') for obj in objects]
            
            print(f"📊 Знайдено об'єктів: {len(objects)}")
            
            # Шукаємо дублікати
            counts = Counter(ids)
            duplicates = [id for id, count in counts.items() if count > 1]
            
            if duplicates:
                print(f"❌ КРИТИЧНО: Знайдено дублікати ID об'єктів! Це ламає Bambu Studio.")
                print(f"   Дубльовані ID: {duplicates}")
            else:
                print("✅ ID об'єктів унікальні (OK)")

            # 4. Перевірка наявності геометрії
            empty_objects = 0
            for obj in objects:
                oid = obj.get('id')
                mesh = obj.find('m:mesh', ns)
                components = obj.find('m:components', ns)
                
                if mesh is not None:
                    vertices = mesh.find('m:vertices', ns)
                    triangles = mesh.find('m:triangles', ns)
                    v_count = len(list(vertices)) if vertices is not None else 0
                    t_count = len(list(triangles)) if triangles is not None else 0
                    
                    if v_count == 0 or t_count == 0:
                        print(f"⚠️ ПОПЕРЕДЖЕННЯ: Об'єкт ID={oid} має меш, але 0 вершин/трикутників.")
                        empty_objects += 1
                elif components is not None:
                    # Це збірка, це норм
                    pass
                else:
                    print(f"❌ ПОМИЛКА: Об'єкт ID={oid} пустий (немає ні мешу, ні компонентів).")
                    empty_objects += 1

            if empty_objects == 0:
                print("✅ Всі об'єкти містять дані (OK)")

    except Exception as e:
        print(f"❌ СИСТЕМНА ПОМИЛКА при читанні: {e}")

# Вкажіть тут ім'я вашого файлу
file_name = "model_93994bb5.3mf" 
validate_3mf(file_name)