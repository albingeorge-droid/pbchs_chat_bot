# utils/map.py
from __future__ import annotations

import re
from typing import Tuple, List, Dict, Any

from langsmith import traceable

from db import run_select


def parse_plot_road_from_text(text: str) -> Tuple[str | None, str | None]:
    """
    Extract (plot_no, road_no) from a natural-language question.
    
    Supports:
    - "plot 30 road 14" → ("30", "14")
    - "plot 30/14" → ("30", "14")
    - "plot 30a road 14b" → ("30a", "14b")
    - "plot 28/North Avenue Road" → ("28", "north avenue road")
    """
    if not text:
        return None, None

    s = text.lower()
    plot = None
    road = None

    # 1) Explicit "plot X" pattern - just get the alphanumeric after "plot"
    m_plot = re.search(r"plot\s+(?:number\s+)?([a-z0-9]+)\b", s)
    if m_plot:
        plot = m_plot.group(1).strip()

    # 2) Explicit "road Y" pattern - capture alphanumeric + optional words
    m_road = re.search(r"road\s+(?:number\s+)?([a-z0-9]+(?:\s+[a-z]+)*)\b", s)
    if m_road:
        road = m_road.group(1).strip()

    # 3) Pattern "X/Y" where Y can be multi-word (like "North Avenue Road")
    if not (plot and road):
        # Match "30/14", "28/North Avenue Road", etc.
        # Captures everything after / until end of string or specific delimiters
        m_pair = re.search(r"\b([a-z0-9]+)\s*/\s*([a-z0-9]+(?:\s+[a-z]+)*)", s)
        if m_pair:
            if not plot:
                plot = m_pair.group(1).strip()
            if not road:
                road = m_pair.group(2).strip()

    return plot, road
@traceable(run_type="chain", name="map_lookup_pra")
def lookup_pra_for_plot_road(plot: str, road: str) -> tuple[str, List[Dict[str, Any]]]:
    """
    Given canonical plot/road numbers, look up matching PRAs.

    Returns (sql, rows).
    """
    safe_plot = plot.replace("'", "''")
    safe_road = road.replace("'", "''")

    sql = f"""
SELECT p.pra_
FROM properties p
JOIN property_addresses pa
  ON pa.property_id = p.id
WHERE LOWER(TRIM(pa.plot_no)) = LOWER('{safe_plot}')
  AND LOWER(TRIM(pa.road_no)) = LOWER('{safe_road}')
LIMIT 1;
""".strip()

    rows = run_select(sql, preserve_limit=True)  # ✅ Keep the LIMIT 1
    return sql, rows


@traceable(run_type="chain", name="map_fetch_geometry")
def fetch_map_for_pra(pra: str) -> tuple[str, List[Dict[str, Any]]]:
    """
    Fetch GeoJSON features for a given PRA from pbchs_map.

    Returns (sql, rows). Each row has:
      - id
      - feature: { "type": "Feature", "geometry": {...}, "properties": {...} }
    """
    safe_pra = pra.replace("'", "''")

    sql = f"""
SELECT
  id,
  jsonb_build_object(
    'type', 'Feature',
    'geometry', ST_AsGeoJSON(geom)::jsonb,
    'properties', properties
  ) AS feature
FROM pbchs_map
WHERE properties->>'pra_id' = '{safe_pra}';
""".strip()

    rows = run_select(sql, preserve_limit=True)  # ✅ Preserve any LIMIT if added
    return sql, rows