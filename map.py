# utils/map.py
from __future__ import annotations

import re
from typing import Tuple, List, Dict, Any

from langsmith import traceable

from db import run_select


def parse_plot_road_from_text(text: str) -> Tuple[str | None, str | None]:
    """
    Extract (plot_no, road_no) from a natural-language question like:
      - "show the map of plot 30 road 15"
      - "map 30/14"
      - "map for plot 42 on road 7"

    Returns (plot, road) or (None, None) if not found.
    """
    if not text:
        return None, None

    s = text.lower()

    plot = None
    road = None

    # 1) explicit "plot 30" / "road 15"
    m_plot = re.search(r"plot\s+(\d{1,4})", s)
    if m_plot:
        plot = m_plot.group(1)

    m_road = re.search(r"road\s+(\d{1,4})", s)
    if m_road:
        road = m_road.group(1)

    # 2) pattern "30/14" or "30 14" if one of them is still missing
    if not (plot and road):
        m_pair = re.search(r"\b(\d{1,4})\s*[/ ]\s*(\d{1,4})\b", s)
        if m_pair:
            if not plot:
                plot = m_pair.group(1)
            if not road:
                road = m_pair.group(2)

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
SELECT p.pra
FROM properties p
JOIN property_addresses pa
  ON pa.property_id = p.id
WHERE TRIM(pa.plot_no) = '{safe_plot}'
  AND TRIM(pa.road_no) = '{safe_road}'
LIMIT 5;
""".strip()

    rows = run_select(sql) or []
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

    rows = run_select(sql) or []
    return sql, rows
