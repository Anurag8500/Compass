"""Offline OpenStreetMap (OSM) extract preparation and road network graph builder (Phase 12).

Provides:
1. Extraction of driveable road networks from Overpass API (offline build time only)
   or offline OSM JSON/XML files.
2. Conversion into a deterministic NetworkX directed graph (`nx.DiGraph`).
3. Preservation of node coordinates (lat, lon), edge polylines, directionality (oneway vs two-way),
   road classifications, street names, and lengths in meters.
4. Projection into session-local East-North-Up (ENU) Cartesian coordinates using the active
   `GeoReference` so that road coordinates and ESKF filter outputs share the identical frame.
5. Deterministic local JSON serialization and loading for zero-internet runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.request

import networkx as nx
import numpy as np

from navigation.frames.local_geo import GeoReference, R_EARTH_METERS

# Permitted driveable highway classes
DRIVEABLE_HIGHWAY_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "living_street",
    "service",
}


@dataclass
class RoadSegment:
    """A directed linear subsegment between two consecutive polyline vertices."""
    edge_id: str
    u: int
    v: int
    segment_idx: int
    p0_enu: np.ndarray  # (2,) [east, north]
    p1_enu: np.ndarray  # (2,) [east, north]
    length_m: float
    tangent: np.ndarray  # (2,) unit tangent [tx, ty]
    normal: np.ndarray  # (2,) unit normal [-ty, tx]
    azimuth_rad: float  # [0, 2pi) bearing clockwise from North
    cum_dist_start_m: float
    p0_lat_lon: Tuple[float, float]
    p1_lat_lon: Tuple[float, float]


@dataclass
class RoadEdgeData:
    """Attributes stored for each directed edge in the road graph."""
    edge_id: str
    u: int
    v: int
    highway: str
    name: str
    oneway: bool
    length_m: float
    geometry_latlon: List[Tuple[float, float]]  # [(lat, lon), ...]
    geometry_enu: Optional[List[Tuple[float, float]]] = None  # [(east, north), ...]
    azimuth_rad: float = 0.0


class RoadNetworkGraph:
    """Container for the offline road network directed graph with local ENU projection."""

    def __init__(
        self,
        graph: nx.DiGraph,
        metadata: Optional[Dict[str, Any]] = None,
        geo_ref: Optional[GeoReference] = None,
    ) -> None:
        self.graph = graph
        self.metadata = metadata or {}
        self.geo_ref: Optional[GeoReference] = None
        self._segments: List[RoadSegment] = []
        if geo_ref is not None:
            self.set_geo_reference(geo_ref)

    def set_geo_reference(self, geo_ref: GeoReference) -> None:
        """Project node and edge coordinates into the active session-local ENU frame."""
        self.geo_ref = geo_ref
        self._segments.clear()

        # Update node ENU coordinates
        for node, data in self.graph.nodes(data=True):
            lat = data["lat"]
            lon = data["lon"]
            e, n, _ = geo_ref.geodetic_to_enu(lat, lon, 0.0)
            data["x_enu"] = float(e)
            data["y_enu"] = float(n)

        # Update edge ENU polylines and extract linear segments
        for u, v, data in self.graph.edges(data=True):
            geom_latlon = data["geometry_latlon"]
            geom_enu: List[Tuple[float, float]] = []
            cum_dist = 0.0

            for lat, lon in geom_latlon:
                e, n, _ = geo_ref.geodetic_to_enu(lat, lon, 0.0)
                geom_enu.append((float(e), float(n)))

            data["geometry_enu"] = geom_enu

            # Build linear segments along this edge
            edge_id = data.get("edge_id", f"{u}_{v}")
            for idx in range(len(geom_enu) - 1):
                p0 = np.array(geom_enu[idx], dtype=np.float64)
                p1 = np.array(geom_enu[idx + 1], dtype=np.float64)
                delta = p1 - p0
                seg_len = float(np.linalg.norm(delta))
                if seg_len < 1e-6:
                    continue

                tangent = delta / seg_len
                # Road normal: rotated 90 deg counterclockwise [-ty, tx]
                normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
                # Azimuth: angle clockwise from North (Y-axis) in [0, 2pi)
                azimuth = float(math.atan2(tangent[0], tangent[1])) % (2.0 * math.pi)

                self._segments.append(
                    RoadSegment(
                        edge_id=edge_id,
                        u=u,
                        v=v,
                        segment_idx=idx,
                        p0_enu=p0,
                        p1_enu=p1,
                        length_m=seg_len,
                        tangent=tangent,
                        normal=normal,
                        azimuth_rad=azimuth,
                        cum_dist_start_m=cum_dist,
                        p0_lat_lon=geom_latlon[idx],
                        p1_lat_lon=geom_latlon[idx + 1],
                    )
                )
                cum_dist += seg_len

    @property
    def segments(self) -> List[RoadSegment]:
        """List of all directed linear road segments in the active session ENU frame."""
        return self._segments

    def to_dict(self) -> Dict[str, Any]:
        """Serialize road network graph to deterministic dictionary."""
        nodes_data = []
        for n, d in sorted(self.graph.nodes(data=True), key=lambda item: item[0]):
            nodes_data.append({
                "id": n,
                "lat": float(d["lat"]),
                "lon": float(d["lon"]),
            })

        edges_data = []
        for u, v, d in sorted(self.graph.edges(data=True), key=lambda item: (item[0], item[1], item[2].get("edge_id", ""))):
            edges_data.append({
                "edge_id": d.get("edge_id", f"{u}_{v}"),
                "u": u,
                "v": v,
                "highway": d.get("highway", "residential"),
                "name": d.get("name", ""),
                "oneway": bool(d.get("oneway", False)),
                "length_m": float(d.get("length_m", 0.0)),
                "azimuth_rad": float(d.get("azimuth_rad", 0.0)),
                "geometry_latlon": [[float(pt[0]), float(pt[1])] for pt in d["geometry_latlon"]],
            })

        return {
            "metadata": self.metadata,
            "nodes": nodes_data,
            "edges": edges_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], geo_ref: Optional[GeoReference] = None) -> RoadNetworkGraph:
        """Deserialize road network graph from dictionary."""
        g = nx.DiGraph()
        for nd in data["nodes"]:
            g.add_node(nd["id"], lat=nd["lat"], lon=nd["lon"])

        for ed in data["edges"]:
            geom = [(pt[0], pt[1]) for pt in ed["geometry_latlon"]]
            g.add_edge(
                ed["u"],
                ed["v"],
                edge_id=ed["edge_id"],
                highway=ed.get("highway", "residential"),
                name=ed.get("name", ""),
                oneway=ed.get("oneway", False),
                length_m=ed.get("length_m", 0.0),
                azimuth_rad=ed.get("azimuth_rad", 0.0),
                geometry_latlon=geom,
            )

        return cls(graph=g, metadata=data.get("metadata", {}), geo_ref=geo_ref)

    def save_json(self, filepath: str | Path) -> None:
        """Save the road graph to a local JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str | Path, geo_ref: Optional[GeoReference] = None) -> RoadNetworkGraph:
        """Load the road graph from a local JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, geo_ref=geo_ref)


def build_graph_from_overpass_json(
    osm_json: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> RoadNetworkGraph:
    """Construct a directed road graph from Overpass API JSON elements.

    Filters driveable highways and handles one-way vs two-way directionality.
    """
    nodes: Dict[int, Tuple[float, float]] = {}
    ways: List[Dict[str, Any]] = []

    for element in osm_json.get("elements", []):
        el_type = element.get("type")
        if el_type == "node":
            nodes[element["id"]] = (float(element["lat"]), float(element["lon"]))
        elif el_type == "way":
            tags = element.get("tags", {})
            highway = tags.get("highway")
            if highway in DRIVEABLE_HIGHWAY_TYPES:
                ways.append(element)

    g = nx.DiGraph()

    def haversine_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return 2.0 * R_EARTH_METERS * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    for way in ways:
        way_id = way["id"]
        way_nodes = way.get("nodes", [])
        if len(way_nodes) < 2:
            continue

        # Check that all nodes are present
        valid_nodes = [nid for nid in way_nodes if nid in nodes]
        if len(valid_nodes) < 2:
            continue

        tags = way.get("tags", {})
        highway = tags.get("highway", "residential")
        name = tags.get("name", "")
        oneway_tag = tags.get("oneway", "no")
        is_motorway = highway in {"motorway", "motorway_link"}
        oneway = oneway_tag in {"yes", "1", "true"} or is_motorway
        is_reverse_oneway = oneway_tag in {"-1", "reverse"}

        # Build forward geometry
        forward_geom = [nodes[nid] for nid in valid_nodes]
        total_len_m = 0.0
        for i in range(len(forward_geom) - 1):
            total_len_m += haversine_dist_m(
                forward_geom[i][0], forward_geom[i][1],
                forward_geom[i + 1][0], forward_geom[i + 1][1],
            )

        # Initial heading calculation from start to end
        dlat = math.radians(forward_geom[-1][0] - forward_geom[0][0])
        dlon = math.radians(forward_geom[-1][1] - forward_geom[0][1])
        azimuth = float(math.atan2(dlon, dlat)) % (2.0 * math.pi)

        # Add nodes to graph
        for nid in valid_nodes:
            lat, lon = nodes[nid]
            g.add_node(nid, lat=lat, lon=lon)

        u = valid_nodes[0]
        v = valid_nodes[-1]

        if is_reverse_oneway:
            # Reversed oneway
            rev_geom = list(reversed(forward_geom))
            rev_azimuth = (azimuth + math.pi) % (2.0 * math.pi)
            g.add_edge(
                v,
                u,
                edge_id=f"way_{way_id}_rev",
                highway=highway,
                name=name,
                oneway=True,
                length_m=total_len_m,
                azimuth_rad=rev_azimuth,
                geometry_latlon=rev_geom,
            )
        elif oneway:
            # Forward oneway
            g.add_edge(
                u,
                v,
                edge_id=f"way_{way_id}_fwd",
                highway=highway,
                name=name,
                oneway=True,
                length_m=total_len_m,
                azimuth_rad=azimuth,
                geometry_latlon=forward_geom,
            )
        else:
            # Bidirectional street: add forward and reverse edges
            g.add_edge(
                u,
                v,
                edge_id=f"way_{way_id}_fwd",
                highway=highway,
                name=name,
                oneway=False,
                length_m=total_len_m,
                azimuth_rad=azimuth,
                geometry_latlon=forward_geom,
            )
            rev_geom = list(reversed(forward_geom))
            rev_azimuth = (azimuth + math.pi) % (2.0 * math.pi)
            g.add_edge(
                v,
                u,
                edge_id=f"way_{way_id}_rev",
                highway=highway,
                name=name,
                oneway=False,
                length_m=total_len_m,
                azimuth_rad=rev_azimuth,
                geometry_latlon=rev_geom,
            )

    meta = metadata or {}
    meta.update({
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "coordinate_system": "WGS84 lat/lon + Session ENU Cartesian projection",
    })

    return RoadNetworkGraph(graph=g, metadata=meta)


def fetch_osm_road_network(
    south: float,
    west: float,
    north: float,
    east: float,
    output_json_path: Optional[str | Path] = None,
    timeout_s: int = 60,
) -> RoadNetworkGraph:
    """Query Overpass API for driveable roads in bounding box (offline extract preparation only).

    Saves the extracted graph locally so subsequent navigation runs require zero internet connectivity.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:{timeout_s}];
    (
      way["highway"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    req = urllib.request.Request(
        overpass_url,
        data=query.encode("utf-8"),
        headers={"User-Agent": "COMPASS-MapMatch-Phase12/1.0 (offline-extract-tool)"},
    )

    with urllib.request.urlopen(req, timeout=timeout_s + 10) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Overpass API returned status {resp.status}")
        raw_data = json.loads(resp.read().decode("utf-8"))

    metadata = {
        "source": "OpenStreetMap via Overpass API",
        "bbox": [south, west, north, east],
        "extract_timestamp_utc": "2026-09-11T15:00:00Z",
    }

    graph = build_graph_from_overpass_json(raw_data, metadata=metadata)

    if output_json_path is not None:
        graph.save_json(output_json_path)

    return graph
