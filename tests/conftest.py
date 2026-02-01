"""
Pytest конфігурація та фікстури
"""
import pytest
import os
import sys
from pathlib import Path

# Додаємо корінь проекту до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

# Тестові дані
TEST_BBOX = {
    "north": 50.455,
    "south": 50.450,
    "east": 30.530,
    "west": 30.520
}

@pytest.fixture
def test_bbox():
    """Тестовий bounding box (Київ, невелика область)"""
    return TEST_BBOX

@pytest.fixture
def output_dir(tmp_path):
    """Тимчасова директорія для виводу"""
    output = tmp_path / "output"
    output.mkdir()
    return output

@pytest.fixture
def mock_osm_data():
    """Моковані OSM дані"""
    import geopandas as gpd
    from shapely.geometry import Polygon, LineString, Point
    
    # Моковані будівлі
    buildings = gpd.GeoDataFrame({
        'building': ['residential', 'commercial'],
        'height': [10.0, 15.0],
        'geometry': [
            Polygon([(30.520, 50.450), (30.525, 50.450), (30.525, 50.455), (30.520, 50.455)]),
            Polygon([(30.525, 50.450), (30.530, 50.450), (30.530, 50.455), (30.525, 50.455)])
        ]
    })
    
    # Моковані дороги (граф)
    import networkx as nx
    G = nx.Graph()
    G.add_edge(1, 2, geometry=LineString([(30.520, 50.450), (30.530, 50.455)]), highway='primary')
    
    # Мокована вода
    water = gpd.GeoDataFrame({
        'natural': ['water'],
        'geometry': [
            Polygon([(30.522, 50.451), (30.523, 50.451), (30.523, 50.452), (30.522, 50.452)])
        ]
    })
    
    return buildings, water, G

