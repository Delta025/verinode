#!/usr/bin/env python3
"""
Build a curated Solar System test graph for retrieval calibration.

Creates ~100 true facts and ~20 deliberate false claims, each as a
KnowledgeNode with embeddings and entity index. The false claims are
stored as low-confidence nodes from an "unverified" source, while true
facts use a high-quality source.

No LLM needed — purely programmatic.

Output: data/solar_system_test_graph.json

Usage:
  python scripts/build_solar_system_test.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.graph import KnowledgeGraph
from src.core.node import KnowledgeNode, NodeType, NodeStatus, Source
from src.core.confidence import ConfidenceEngine
from src.core.embeddings import EmbeddingEngine
from src.core.entity_index import EntityIndex

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# True facts (~100)
# ------------------------------------------------------------------

SOLAR_SYSTEM_FACTS = [
    # Planets — basic
    "The Solar System has eight planets.",
    "Mercury is the closest planet to the Sun.",
    "Venus is the second planet from the Sun.",
    "Earth is the third planet from the Sun.",
    "Mars is the fourth planet from the Sun.",
    "Jupiter is the fifth planet from the Sun.",
    "Saturn is the sixth planet from the Sun.",
    "Uranus is the seventh planet from the Sun.",
    "Neptune is the eighth planet from the Sun.",
    # Planet types
    "Mercury is a rocky terrestrial planet.",
    "Venus is a rocky terrestrial planet.",
    "Earth is a rocky terrestrial planet.",
    "Mars is a rocky terrestrial planet.",
    "Jupiter is a gas giant.",
    "Saturn is a gas giant.",
    "Uranus is an ice giant.",
    "Neptune is an ice giant.",
    # Moons
    "Mercury has no moons.",
    "Venus has no moons.",
    "Earth has one moon called the Moon.",
    "Mars has two moons named Phobos and Deimos.",
    "Jupiter has at least 95 known moons.",
    "Saturn has at least 146 known moons.",
    "Uranus has 27 known moons.",
    "Neptune has 16 known moons.",
    # Distances from Sun (semi-major axis)
    "Mercury orbits the Sun at an average distance of 57.9 million km.",
    "Venus orbits the Sun at an average distance of 108.2 million km.",
    "Earth orbits the Sun at an average distance of 149.6 million km.",
    "Mars orbits the Sun at an average distance of 227.9 million km.",
    "Jupiter orbits the Sun at an average distance of 778.5 million km.",
    "Saturn orbits the Sun at an average distance of 1.434 billion km.",
    "Uranus orbits the Sun at an average distance of 2.871 billion km.",
    "Neptune orbits the Sun at an average distance of 4.495 billion km.",
    # Orbital periods
    "Mercury's orbital period is about 88 Earth days.",
    "Venus's orbital period is about 225 Earth days.",
    "Earth's orbital period is 365.25 days.",
    "Mars's orbital period is about 687 Earth days.",
    "Jupiter's orbital period is about 11.86 Earth years.",
    "Saturn's orbital period is about 29.5 Earth years.",
    "Uranus's orbital period is about 84 Earth years.",
    "Neptune's orbital period is about 165 Earth years.",
    # Diameters
    "Mercury has a diameter of 4,879 km.",
    "Venus has a diameter of 12,104 km.",
    "Earth has a diameter of 12,742 km.",
    "Mars has a diameter of 6,779 km.",
    "Jupiter has a diameter of 139,820 km.",
    "Saturn has a diameter of 116,460 km.",
    "Uranus has a diameter of 50,724 km.",
    "Neptune has a diameter of 49,244 km.",
    # Temperatures
    "The Sun's surface temperature is approximately 5,778 K.",
    "The Sun's core temperature is approximately 15 million degrees Celsius.",
    "Venus has a surface temperature of about 462 degrees Celsius.",
    "Mars has an average surface temperature of about minus 65 degrees Celsius.",
    "Mercury's daytime temperature can reach 430 degrees Celsius.",
    "Mercury's nighttime temperature can drop to minus 180 degrees Celsius.",
    # Compositions
    "The Sun is composed of approximately 73% hydrogen and 25% helium.",
    "Jupiter is composed primarily of hydrogen and helium.",
    "Saturn is composed primarily of hydrogen and helium.",
    "Earth's atmosphere is composed of approximately 78% nitrogen and 21% oxygen.",
    "Mars's atmosphere is composed of approximately 95% carbon dioxide.",
    "Venus's atmosphere is composed of approximately 96% carbon dioxide.",
    # Notable moons
    "The Moon has a diameter of 3,474 km.",
    "The Moon orbits Earth at an average distance of 384,400 km.",
    "Titan is Saturn's largest moon and has a thick atmosphere.",
    "Europa is a moon of Jupiter with a subsurface ocean.",
    "Ganymede is the largest moon in the Solar System.",
    "Io is the most volcanically active body in the Solar System.",
    "Enceladus has geysers of water ice erupting from its south pole.",
    "Triton is Neptune's largest moon and orbits retrograde.",
    # Sun properties
    "The Sun has a diameter of approximately 1.39 million km.",
    "The Sun is a G-type main-sequence star (yellow dwarf).",
    "The Sun is approximately 4.6 billion years old.",
    "The Sun accounts for about 99.86% of the total mass of the Solar System.",
    # Other features
    "The asteroid belt is located between Mars and Jupiter.",
    "The Kuiper Belt extends beyond the orbit of Neptune.",
    "Jupiter's Great Red Spot is a persistent anticyclonic storm.",
    "Saturn's rings are composed primarily of ice particles and rocky debris.",
    "Olympus Mons on Mars is the tallest volcano in the Solar System.",
    "Olympus Mons has a height of approximately 21.9 km.",
    "The Oort Cloud is a theoretical shell of icy objects surrounding the Solar System.",
    "Pluto was reclassified as a dwarf planet in 2006.",
    # Physical constants relevant to Solar System
    "The speed of light is approximately 299,792 km per second.",
    "Earth's gravitational acceleration at sea level is approximately 9.81 m/s squared.",
    "Jupiter's gravitational acceleration at cloud tops is approximately 24.79 m/s squared.",
    # Rotation periods
    "Earth's rotation period is approximately 24 hours.",
    "Jupiter's rotation period is approximately 9.93 hours.",
    "Venus's rotation period is approximately 243 Earth days.",
    "Venus rotates in the opposite direction to most planets (retrograde).",
    # Misc
    "Light from the Sun takes approximately 8 minutes and 20 seconds to reach Earth.",
    "The Solar System formed approximately 4.6 billion years ago from a molecular cloud.",
    "Halley's Comet has an orbital period of approximately 75 to 79 years.",
    "The Voyager 1 spacecraft is the most distant human-made object from Earth.",
]

# ------------------------------------------------------------------
# Deliberate false claims (~20)
# ------------------------------------------------------------------

SOLAR_SYSTEM_FALSE_CLAIMS = [
    # Obvious numerical errors
    "Earth orbits the Sun at an average distance of 230 million km.",          # actual: 149.6M
    "Mars has five moons.",                                                    # actual: 2
    "Jupiter has a diameter of 71,000 km.",                                    # actual: 139,820 km
    # Subtle numerical errors
    "The Moon orbits Earth at an average distance of 405,000 km.",             # actual: 384,400 km
    "Earth's orbital period is 365 days.",                                     # actual: 365.25
    "Mercury's orbital period is about 116 Earth days.",                       # actual: 88
    # Categorical errors
    "Venus is a gas giant.",                                                   # actual: rocky terrestrial
    "Neptune is the seventh planet from the Sun.",                             # actual: 8th
    "Uranus is a gas giant.",                                                  # actual: ice giant
    # Compositional errors
    "The Sun is primarily composed of helium.",                                # actual: hydrogen
    "Earth's atmosphere is composed of approximately 78% oxygen.",             # actual: nitrogen
    "Mars's atmosphere is composed of approximately 95% nitrogen.",            # actual: carbon dioxide
    # Attribution errors
    "Europa is a moon of Saturn.",                                             # actual: Jupiter
    "Olympus Mons is located on Earth.",                                       # actual: Mars
    "Titan is Jupiter's largest moon.",                                        # actual: Saturn's
    # Conceptual errors
    "Jupiter's Great Red Spot is a large crater.",                             # actual: storm
    "Saturn's rings are composed primarily of solid rock.",                     # actual: ice + rock
    "Pluto is the ninth planet of the Solar System.",                          # reclassified 2006
    # Mixed errors
    "The Sun has a diameter of approximately 500,000 km.",                     # actual: 1.39M km
    "The speed of light is approximately 150,000 km per second.",              # actual: 299,792
]


def main():
    print("=" * 60)
    print("Solar System Test Graph Builder")
    print("=" * 60)

    t0 = time.time()
    graph = KnowledgeGraph()

    true_source = Source(
        source_type="encyclopedia",
        title="Solar System Reference",
        quality_score=0.95,
    )
    false_source = Source(
        source_type="unverified_web",
        title="Unverified Blog Post",
        quality_score=0.30,
    )

    # Add true facts
    true_ids = []
    for text in SOLAR_SYSTEM_FACTS:
        node = KnowledgeNode(
            text=text,
            node_type=NodeType.FACT,
            domain="astronomy",
            confidence=0.90,
            sources=[true_source],
        )
        node.status = NodeStatus.WELL_SUPPORTED
        graph.add_node(node)
        true_ids.append(node.id)

    # Add false claims with lower confidence
    false_ids = []
    for text in SOLAR_SYSTEM_FALSE_CLAIMS:
        node = KnowledgeNode(
            text=text,
            node_type=NodeType.FACT,
            domain="astronomy",
            confidence=0.40,
            sources=[false_source],
        )
        node.status = NodeStatus.PROVISIONAL
        graph.add_node(node)
        false_ids.append(node.id)

    print(f"  Nodes created: {len(graph)} "
          f"({len(true_ids)} true + {len(false_ids)} false)")

    # Build embedding index
    t1 = time.time()
    emb = EmbeddingEngine()
    emb.index_graph(graph)
    print(f"  Embeddings indexed in {time.time() - t1:.2f}s")

    # Build entity index
    idx = EntityIndex()
    idx.index_graph(graph)
    print(f"  Entity index: {idx.size} nodes, "
          f"{len(idx._word_to_nodes)} unique words")

    # Save
    out_path = DATA_DIR / "solar_system_test_graph.json"
    graph.save(str(out_path))

    stats = graph.get_statistics()
    elapsed = time.time() - t0
    print(f"\n  Final: {stats['total_nodes']} nodes, "
          f"avg confidence {stats['avg_confidence']:.3f}")
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Done in {elapsed:.1f}s")
    print("=" * 60)

    # Return IDs for reference
    return graph, true_ids, false_ids


if __name__ == "__main__":
    main()
