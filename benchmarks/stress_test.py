#!/usr/bin/env python3
"""
Stress Test - Verinode Memory Substrate

Tests the system at scale with 150 facts across 3 domains (chemistry,
physics, biology). Validates ingestion throughput, retrieval quality,
contradiction detection accuracy, and export/import round-trip fidelity.

Requires: Running Ollama server with qwen3:14b
Usage:    python benchmarks/stress_test.py

Recommended Ollama env vars for faster runtime:
    set OLLAMA_FLASH_ATTENTION=1
    set OLLAMA_KV_CACHE_TYPE=q8_0
    set OLLAMA_NUM_PARALLEL=4
"""

import sys
import time
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph import KnowledgeGraph
from src.core.node import Source
from src.core.confidence import ConfidenceEngine
from src.core.reasoning import ReasoningEngine
from src.core.pipeline import Pipeline
from src.core.fact_extraction import FactExtractionEngine
from src.core.embeddings import EmbeddingEngine
from src.core.verification import VerificationEngine
from src.core.llm_contradiction import create_contradiction_checker
from src.core.llm_memory import LLMMemory
from src.core.export import export_subgraph, export_to_json, import_from_json, import_subgraph


# ---------------------------------------------------------------
# Dataset: 150 facts across 3 domains
# ---------------------------------------------------------------

CHEMISTRY_FACTS = [
    # Melting/boiling points (10)
    "Gold melts at 1064 degrees Celsius.",
    "Silver melts at 961.8 degrees Celsius.",
    "Copper melts at 1085 degrees Celsius.",
    "Iron melts at 1538 degrees Celsius.",
    "Aluminum melts at 660.3 degrees Celsius.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Ethanol boils at 78.37 degrees Celsius.",
    "Mercury boils at 356.7 degrees Celsius.",
    "Nitrogen boils at minus 195.8 degrees Celsius.",
    "Oxygen boils at minus 183 degrees Celsius.",

    # Chemical symbols (10)
    "Gold has the chemical symbol Au on the periodic table.",
    "Silver has the chemical symbol Ag.",
    "Iron has the chemical symbol Fe.",
    "Sodium has the chemical symbol Na.",
    "Potassium has the chemical symbol K.",
    "Mercury has the chemical symbol Hg.",
    "Copper has the chemical symbol Cu.",
    "Lead has the chemical symbol Pb.",
    "Tin has the chemical symbol Sn.",
    "Tungsten has the chemical symbol W.",

    # Element properties (10)
    "Pure gold has a density of 19.3 grams per cubic centimeter.",
    "Gold is highly resistant to corrosion and oxidation.",
    "Lithium is the lightest metal with a density of 0.534 grams per cubic centimeter.",
    "Osmium is the densest naturally occurring element at 22.59 grams per cubic centimeter.",
    "Francium is the most electropositive element.",
    "Fluorine is the most electronegative element on the periodic table.",
    "Helium has the lowest boiling point of any element at minus 268.9 degrees Celsius.",
    "Carbon has four valence electrons and can form four covalent bonds.",
    "Sulfur is a yellow nonmetal that exists in several allotropic forms.",
    "Silicon is the second most abundant element in the Earth's crust after oxygen.",

    # Compound properties (10)
    "Water has the chemical formula H2O.",
    "Carbon dioxide has the chemical formula CO2.",
    "Table salt is sodium chloride with the formula NaCl.",
    "Sulfuric acid has the chemical formula H2SO4.",
    "Methane has the chemical formula CH4.",
    "Ammonia has the chemical formula NH3.",
    "Glucose has the molecular formula C6H12O6.",
    "Ethanol has the chemical formula C2H5OH.",
    "Hydrochloric acid has the chemical formula HCl.",
    "Nitric acid has the chemical formula HNO3.",

    # Reactions and processes (10)
    "Combustion of hydrogen in oxygen produces water.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "Rusting is the oxidation of iron in the presence of water and oxygen.",
    "Electrolysis of water produces hydrogen gas and oxygen gas.",
    "The Haber process synthesizes ammonia from nitrogen and hydrogen.",
    "Fermentation converts glucose into ethanol and carbon dioxide.",
    "Neutralization of an acid and a base produces a salt and water.",
    "Combustion of methane in oxygen produces carbon dioxide and water.",
    "The thermite reaction between aluminum and iron oxide produces molten iron.",
    "Saponification is the hydrolysis of a fat or oil to produce soap and glycerol.",
]

PHYSICS_FACTS = [
    # Physical constants and measurements (10)
    "The speed of light in a vacuum is approximately 299792 kilometers per second.",
    "The gravitational constant G is approximately 6.674 times 10 to the minus 11 cubic meters per kilogram per second squared.",
    "Planck's constant is approximately 6.626 times 10 to the minus 34 joule seconds.",
    "The charge of an electron is approximately 1.602 times 10 to the minus 19 coulombs.",
    "Absolute zero is minus 273.15 degrees Celsius or zero Kelvin.",
    "The speed of sound in air at room temperature is approximately 343 meters per second.",
    "Standard atmospheric pressure at sea level is 101325 Pascals.",
    "Avogadro's number is approximately 6.022 times 10 to the 23rd per mole.",
    "The Boltzmann constant is approximately 1.381 times 10 to the minus 23 joules per Kelvin.",
    "The mass of an electron is approximately 9.109 times 10 to the minus 31 kilograms.",

    # Mechanics (10)
    "Newton's first law states that an object at rest stays at rest unless acted upon by an external force.",
    "Newton's second law states that force equals mass times acceleration.",
    "Newton's third law states that every action has an equal and opposite reaction.",
    "Momentum is the product of mass and velocity.",
    "Kinetic energy equals one-half times mass times velocity squared.",
    "Gravitational potential energy equals mass times gravitational acceleration times height.",
    "Work is defined as force multiplied by displacement in the direction of the force.",
    "Power is the rate at which work is done or energy is transferred.",
    "Angular momentum is conserved in a closed system with no external torques.",
    "The coefficient of friction determines the force needed to slide one surface over another.",

    # Thermodynamics (10)
    "The first law of thermodynamics states that energy cannot be created or destroyed.",
    "The second law of thermodynamics states that entropy of an isolated system always increases.",
    "Heat flows spontaneously from a hotter body to a cooler body.",
    "The third law of thermodynamics states that entropy approaches zero as temperature approaches absolute zero.",
    "Specific heat capacity is the amount of heat needed to raise one gram of a substance by one degree Celsius.",
    "Water has a specific heat capacity of 4.184 joules per gram per degree Celsius.",
    "Latent heat of fusion is the energy required to change a substance from solid to liquid.",
    "An ideal gas follows the equation PV equals nRT.",
    "Thermal conductivity measures how well a material conducts heat.",
    "Convection transfers heat through the movement of fluids.",

    # Electromagnetism and optics (10)
    "Electric current is the flow of electric charge through a conductor.",
    "Ohm's law states that voltage equals current times resistance.",
    "Electromagnetic waves travel at the speed of light in a vacuum.",
    "Visible light has wavelengths between approximately 380 and 700 nanometers.",
    "Red light has the longest wavelength in the visible spectrum.",
    "Violet light has the shortest wavelength in the visible spectrum.",
    "A magnetic field is produced by moving electric charges.",
    "Faraday's law states that a changing magnetic field induces an electric field.",
    "The electromagnetic spectrum includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays.",
    "Total internal reflection occurs when light hits a boundary at an angle greater than the critical angle.",

    # Astronomy and cosmology (10)
    "The Earth orbits the Sun at an average distance of about 150 million kilometers.",
    "Light from the Sun takes approximately 8 minutes and 20 seconds to reach Earth.",
    "The Moon orbits the Earth at an average distance of about 384400 kilometers.",
    "Jupiter is the largest planet in our solar system.",
    "The observable universe is approximately 93 billion light-years in diameter.",
    "A light-year is the distance that light travels in one year, about 9.461 trillion kilometers.",
    "The Sun is a main-sequence star classified as a G-type yellow dwarf.",
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
    "The cosmic microwave background radiation has a temperature of about 2.725 Kelvin.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
]

BIOLOGY_FACTS = [
    # Cell biology (10)
    "Cells are the basic structural and functional units of all living organisms.",
    "Mitochondria are the powerhouses of the cell, producing ATP through cellular respiration.",
    "The cell membrane is a selectively permeable lipid bilayer.",
    "Ribosomes are cellular organelles responsible for protein synthesis.",
    "The nucleus contains the cell's genetic material in the form of DNA.",
    "The endoplasmic reticulum is involved in protein and lipid synthesis.",
    "The Golgi apparatus modifies, packages, and ships proteins to their destinations.",
    "Lysosomes contain digestive enzymes that break down waste materials.",
    "Chloroplasts are the sites of photosynthesis in plant cells.",
    "The cytoskeleton provides structural support and facilitates cell movement.",

    # Genetics and DNA (10)
    "DNA is a double helix composed of two complementary strands of nucleotides.",
    "The four DNA bases are adenine, thymine, guanine, and cytosine.",
    "Adenine pairs with thymine and guanine pairs with cytosine in DNA.",
    "RNA is single-stranded and uses uracil instead of thymine.",
    "Genes are segments of DNA that encode instructions for making proteins.",
    "The human genome contains approximately 20000 to 25000 protein-coding genes.",
    "Meiosis produces four haploid daughter cells from one diploid cell.",
    "Mitosis produces two genetically identical daughter cells.",
    "Mutations are changes in the DNA sequence that can be inherited.",
    "The genetic code is nearly universal across all known forms of life.",

    # Anatomy (10)
    "The adult human body contains 206 bones.",
    "The human heart beats approximately 100000 times per day.",
    "The human brain contains approximately 86 billion neurons.",
    "The femur is the longest and strongest bone in the human body.",
    "The human body has about 600 skeletal muscles.",
    "The small intestine is approximately 6 meters long in an adult human.",
    "The human skin is the largest organ of the body.",
    "Red blood cells carry oxygen from the lungs to the body's tissues.",
    "White blood cells are key components of the immune system.",
    "The human liver performs over 500 different functions.",

    # Ecology and evolution (10)
    "Natural selection is the process by which organisms with favorable traits survive and reproduce.",
    "An ecosystem includes all living organisms and their physical environment in a given area.",
    "Producers such as plants form the base of most food chains.",
    "Decomposers break down dead organisms and recycle nutrients back into the ecosystem.",
    "Biodiversity refers to the variety of life in a particular habitat or ecosystem.",
    "Symbiosis is a close and long-term biological interaction between two different species.",
    "Mutualism is a type of symbiosis where both species benefit from the interaction.",
    "Predation is an interaction where one organism hunts and feeds on another.",
    "Invasive species can disrupt ecosystems by outcompeting native organisms.",
    "The theory of evolution by natural selection was proposed by Charles Darwin.",

    # Physiology (10)
    "Hemoglobin in red blood cells binds to oxygen for transport in the bloodstream.",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar levels.",
    "The kidneys filter blood and produce urine to remove waste products.",
    "Neurons transmit electrical signals called action potentials.",
    "The human body maintains a core temperature of approximately 37 degrees Celsius.",
    "Enzymes are biological catalysts that speed up chemical reactions in living organisms.",
    "Antibodies are proteins produced by the immune system to fight infections.",
    "The diaphragm is the primary muscle used in breathing.",
    "Platelets are cell fragments that help blood clot at wound sites.",
    "Hormones are chemical messengers that regulate physiological processes.",
]


# ---------------------------------------------------------------
# Verification test data
# ---------------------------------------------------------------

FALSE_CLAIMS = [
    ("Gold melts at 500 degrees Celsius.", "chemistry"),
    ("Water boils at 50 degrees Celsius at standard pressure.", "chemistry"),
    ("Iron has the chemical symbol Ir.", "chemistry"),
    ("DNA is a triple helix composed of three strands.", "biology"),
    ("The human body contains 150 bones.", "biology"),
    ("Mitochondria are responsible for protein synthesis.", "biology"),
    ("The speed of light is approximately 150000 kilometers per second.", "physics"),
    ("Absolute zero is minus 100 degrees Celsius.", "physics"),
    ("Newton's second law states that force equals mass times velocity.", "physics"),
    ("The Moon orbits the Earth at a distance of about 38000 kilometers.", "physics"),
]

TRUE_CLAIMS = [
    ("Gold melts at 1064 degrees Celsius.", "chemistry"),
    ("Water has the chemical formula H2O.", "chemistry"),
    ("Iron has the chemical symbol Fe.", "chemistry"),
    ("DNA is a double helix.", "biology"),
    ("The adult human body contains 206 bones.", "biology"),
    ("Mitochondria produce ATP through cellular respiration.", "biology"),
    ("The speed of light is approximately 300000 kilometers per second.", "physics"),
    ("Absolute zero is minus 273 degrees Celsius.", "physics"),
    ("Force equals mass times acceleration.", "physics"),
    ("Jupiter is the largest planet in our solar system.", "physics"),
]


# ---------------------------------------------------------------
# Retrieval test queries
# ---------------------------------------------------------------

QUERIES = [
    ("What is the melting point of gold?", "chemistry"),
    ("What is the chemical formula for water?", "chemistry"),
    ("What are the noble gases?", "chemistry"),
    ("How does photosynthesis work?", None),
    ("What is Newton's second law?", "physics"),
    ("What is the speed of light?", "physics"),
    ("What is DNA made of?", "biology"),
    ("How many bones in the human body?", "biology"),
    ("What is entropy?", "physics"),
    ("What are the functions of mitochondria?", "biology"),
]


# ---------------------------------------------------------------
# Main test
# ---------------------------------------------------------------

def main():
    print("=" * 70)
    print("Verinode - Larger Dataset Stress Test")
    print("150 facts across 3 domains (chemistry, physics, biology)")
    print("=" * 70)

    # -----------------------------------------------------------
    # Setup
    # -----------------------------------------------------------
    print("\n--- Setup ---")

    graph = KnowledgeGraph()
    confidence_engine = ConfidenceEngine()
    reasoning_engine = ReasoningEngine()
    embedding_engine = EmbeddingEngine()

    try:
        extractor = FactExtractionEngine(model="qwen3:14b")
    except Exception as e:
        print(f"\nERROR: Could not initialize Ollama ({e})")
        print("Make sure Ollama is running: ollama serve")
        print("And the model is pulled: ollama pull qwen3:14b")
        sys.exit(1)

    pipeline = Pipeline(
        graph,
        extractor=extractor,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
    )

    print("  Pipeline ready.")

    # -----------------------------------------------------------
    # Phase 1: Ingestion (150 facts, 3 domains)
    # -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: INGESTION (150 facts)")
    print("=" * 70)

    domains = [
        ("chemistry", CHEMISTRY_FACTS),
        ("physics", PHYSICS_FACTS),
        ("biology", BIOLOGY_FACTS),
    ]

    total_added = 0
    total_dup = 0
    total_flagged = 0
    total_rejected = 0
    ingestion_start = time.time()

    for domain_name, facts in domains:
        print(f"\n  --- Domain: {domain_name} ({len(facts)} facts) ---")
        source = Source(
            source_type="textbook",
            title=f"{domain_name.title()} Reference",
            quality_score=0.9,
        )

        domain_start = time.time()
        domain_added = 0
        domain_dup = 0

        for i, fact_text in enumerate(facts):
            t0 = time.time()
            result = pipeline.ingest(fact_text, source=source, domain=domain_name)
            elapsed = time.time() - t0

            domain_added += result.nodes_added
            domain_dup += result.duplicates_merged
            total_added += result.nodes_added
            total_dup += result.duplicates_merged
            total_flagged += result.facts_flagged
            total_rejected += result.facts_rejected

            if (i + 1) % 10 == 0:
                domain_elapsed = time.time() - domain_start
                print(f"    {i+1}/{len(facts)} ingested "
                      f"({domain_elapsed:.1f}s elapsed, "
                      f"last={elapsed:.1f}s, "
                      f"+{domain_added} nodes, "
                      f"{domain_dup} dups)")

        domain_elapsed = time.time() - domain_start
        print(f"  {domain_name}: {domain_added} added, {domain_dup} dups "
              f"in {domain_elapsed:.1f}s ({domain_elapsed/len(facts):.1f}s/fact)")

    ingestion_time = time.time() - ingestion_start

    # Build embedding index
    print(f"\n  Building embedding index...")
    t0 = time.time()
    pipeline.build_index()
    index_time = time.time() - t0
    print(f"  Index built in {index_time:.2f}s")

    print(f"\n  INGESTION SUMMARY:")
    print(f"    Total nodes: {len(graph)}")
    print(f"    Added: {total_added}, Duplicates: {total_dup}, "
          f"Flagged: {total_flagged}, Rejected: {total_rejected}")
    print(f"    Time: {ingestion_time:.1f}s ({ingestion_time/150:.1f}s/fact avg)")

    # -----------------------------------------------------------
    # Phase 2: LLM Memory - Setup
    # -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: LLM MEMORY RETRIEVAL")
    print("=" * 70)

    checker = create_contradiction_checker(model="qwen3:14b")
    verification_engine = VerificationEngine(
        graph,
        embedding_engine=embedding_engine,
        confidence_engine=confidence_engine,
        reasoning_engine=reasoning_engine,
        contradiction_checker=checker,
    )

    memory = LLMMemory(
        graph,
        embedding_engine=embedding_engine,
        reasoning_engine=reasoning_engine,
        confidence_engine=confidence_engine,
        pipeline=pipeline,
        verification_engine=verification_engine,
    )

    # Query tests
    query_latencies = []
    recall_latencies = []
    query_coverages = []

    for question, domain in QUERIES:
        # Full query
        t0 = time.time()
        result = memory.query(question, domain=domain, k=5)
        q_elapsed = time.time() - t0
        query_latencies.append(q_elapsed)
        query_coverages.append(result.coverage)

        # Quick recall
        t0 = time.time()
        facts = memory.recall(question, domain=domain, k=5)
        r_elapsed = time.time() - t0
        recall_latencies.append(r_elapsed)

        print(f"\n  Q: {question}")
        print(f"    query: {q_elapsed:.2f}s | coverage: {result.coverage} "
              f"| confidence: {result.overall_confidence:.2f} "
              f"| facts: {len(result.facts)}")
        print(f"    recall: {r_elapsed:.2f}s | facts: {len(facts)}")
        if result.facts:
            print(f"    top: {result.facts[0].text[:70]}")

    avg_query = sum(query_latencies) / len(query_latencies)
    avg_recall = sum(recall_latencies) / len(recall_latencies)

    print(f"\n  RETRIEVAL SUMMARY:")
    print(f"    Avg query latency: {avg_query:.2f}s")
    print(f"    Avg recall latency: {avg_recall:.2f}s")
    print(f"    Coverage distribution: {dict((c, query_coverages.count(c)) for c in set(query_coverages))}")

    # -----------------------------------------------------------
    # Phase 3: Claim Verification
    # -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: CLAIM VERIFICATION (10 false + 10 true)")
    print("=" * 70)

    verify_start = time.time()

    # False claims
    false_correct = 0
    print("\n  --- False Claims ---")
    for claim, domain in FALSE_CLAIMS:
        t0 = time.time()
        v = memory.verify(claim)
        elapsed = time.time() - t0
        is_caught = v["verdict"] == "contradicted"
        if is_caught:
            false_correct += 1
        tag = "CAUGHT" if is_caught else "MISSED"
        print(f"  [{tag}] {claim[:55]}")
        print(f"         verdict={v['verdict']} conf={v['confidence']:.2f} ({elapsed:.1f}s)")

    # True claims
    true_correct = 0
    print("\n  --- True Claims ---")
    for claim, domain in TRUE_CLAIMS:
        t0 = time.time()
        v = memory.verify(claim)
        elapsed = time.time() - t0
        is_ok = v["verdict"] in ("supported", "uncertain", "insufficient_evidence")
        if is_ok:
            true_correct += 1
        tag = "OK" if is_ok else "FALSE_POS"
        print(f"  [{tag}] {claim[:55]}")
        print(f"         verdict={v['verdict']} conf={v['confidence']:.2f} ({elapsed:.1f}s)")

    verify_time = time.time() - verify_start

    print(f"\n  VERIFICATION SUMMARY:")
    print(f"    False claims caught: {false_correct}/{len(FALSE_CLAIMS)} "
          f"({false_correct/len(FALSE_CLAIMS)*100:.0f}%)")
    print(f"    True claims OK: {true_correct}/{len(TRUE_CLAIMS)} "
          f"({true_correct/len(TRUE_CLAIMS)*100:.0f}%)")
    print(f"    False positive rate: {(len(TRUE_CLAIMS)-true_correct)/len(TRUE_CLAIMS)*100:.0f}%")
    print(f"    Time: {verify_time:.1f}s ({verify_time/20:.1f}s/claim avg)")

    # -----------------------------------------------------------
    # Phase 4: Export/Import
    # -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: EXPORT/IMPORT")
    print("=" * 70)

    export_data = {}
    total_export_time = 0
    total_exported_nodes = 0

    for domain_name in ["chemistry", "physics", "biology"]:
        t0 = time.time()
        export = export_subgraph(graph, domain_name, confidence_threshold=0.3)
        json_str = export_to_json(export)
        elapsed = time.time() - t0
        total_export_time += elapsed
        total_exported_nodes += export.node_count
        export_data[domain_name] = json_str

        print(f"  Export {domain_name}: {export.node_count} nodes, "
              f"{export.edge_count} edges ({elapsed:.3f}s, "
              f"{len(json_str)/1024:.1f}KB)")

    # Import into fresh graph
    print(f"\n  Importing all 3 domains into fresh graph...")
    fresh_graph = KnowledgeGraph()
    total_import_time = 0
    total_import_added = 0

    for domain_name, json_str in export_data.items():
        t0 = time.time()
        reimported = import_from_json(json_str)
        stats = import_subgraph(fresh_graph, reimported)
        elapsed = time.time() - t0
        total_import_time += elapsed
        total_import_added += stats["added"]

        print(f"  Import {domain_name}: +{stats['added']} added, "
              f"{stats['merged']} merged, "
              f"{stats['conflicts']} conflicts ({elapsed:.3f}s)")

    print(f"\n  EXPORT/IMPORT SUMMARY:")
    print(f"    Exported: {total_exported_nodes} nodes in {total_export_time:.3f}s")
    print(f"    Imported: {total_import_added} nodes in {total_import_time:.3f}s")
    print(f"    Fresh graph: {len(fresh_graph)} nodes")
    print(f"    Fidelity: {len(fresh_graph)}/{total_exported_nodes} "
          f"({len(fresh_graph)/max(total_exported_nodes,1)*100:.1f}%)")

    # -----------------------------------------------------------
    # Phase 5: Graph Statistics
    # -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: GRAPH STATISTICS")
    print("=" * 70)

    stats = graph.get_statistics()
    print(f"  Total nodes: {stats['total_nodes']}")
    # Count edges manually (predecessor/successor relationships)
    total_edges = sum(len(n.successor_nodes) for n in graph.nodes.values())
    print(f"  Total edges: {total_edges}")
    print(f"  Contradictions: {stats.get('nodes_with_conflicts', 0)}")
    print(f"  Domains: {dict(stats.get('nodes_by_domain', {}))}")
    print(f"  Node types: {dict(stats.get('nodes_by_type', {}))}")
    print(f"  Status distribution: {dict(stats.get('nodes_by_status', {}))}")

    # Confidence stats
    nodes = list(graph.nodes.values())
    confs = [n.confidence for n in nodes]
    if confs:
        avg_conf = sum(confs) / len(confs)
        min_conf = min(confs)
        max_conf = max(confs)
        high = sum(1 for c in confs if c >= 0.8)
        med = sum(1 for c in confs if 0.5 <= c < 0.8)
        low = sum(1 for c in confs if c < 0.5)
        print(f"  Confidence: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
        print(f"  High (>=0.8): {high}, Medium (0.5-0.8): {med}, Low (<0.5): {low}")

    # -----------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------
    total_time = time.time() - ingestion_start

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Input: 150 facts across 3 domains")
    print(f"  Graph: {len(graph)} nodes, {total_edges} edges")
    print(f"  Ingestion: {ingestion_time:.1f}s ({ingestion_time/150:.1f}s/fact)")
    print(f"  Retrieval: {avg_query:.2f}s avg query, {avg_recall:.2f}s avg recall")
    print(f"  Verification: {false_correct}/{len(FALSE_CLAIMS)} false caught, "
          f"{true_correct}/{len(TRUE_CLAIMS)} true OK")
    print(f"  Export/Import: {total_exported_nodes} -> {len(fresh_graph)} nodes "
          f"({total_export_time+total_import_time:.2f}s)")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
