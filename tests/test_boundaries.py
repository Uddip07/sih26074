"""
Test Boundary Ingestion & Spatial Join
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import geopandas as gpd
import pandas as pd
from src.ingest.fetch_boundaries import load_blocks, load_panchayats, build_boundary_lookup


def test_load_blocks():
    blocks = load_blocks()
    assert len(blocks) == 14, f"Expected 14 blocks in Pune, got {len(blocks)}"
    assert "block_name" in blocks.columns
    assert "geometry" in blocks.columns
    assert blocks.crs.to_epsg() == 4326


def test_load_panchayats():
    panchayats = load_panchayats()
    assert len(panchayats) >= 1000, f"Expected >1000 panchayats, got {len(panchayats)}"
    assert "gp_code" in panchayats.columns
    assert "gp_name" in panchayats.columns
    assert panchayats.crs.to_epsg() == 4326


def test_spatial_join_coverage():
    lookup, joined = build_boundary_lookup()
    assert len(lookup) >= 1000
    assert "assigned_block_name" in lookup.columns
    assert lookup["assigned_block_name"].isna().sum() == 0, "No panchayats should have null block assignment"
    assert lookup["assigned_block_name"].nunique() == 14, "All 14 blocks should have assigned panchayats"
