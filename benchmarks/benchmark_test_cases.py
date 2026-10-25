"""
Benchmark Test Cases — Vulnerability Resistance Testing

75 test cases across 5 categories, all with known ground truth derived
from the 215-node stress test graph (data/stress_test_graph.json).

Categories:
  A. Factual Accuracy (15)       — Q&A with numerical/categorical answers
  B. False Claim Detection (20)  — 10 false + 10 true claims
  C. Poisoned Context (15)       — False RAG context injection
  D. Confidence Calibration (10) — Confidence quality on true/false claims
  E. Subtle Numerical Errors (15)— Small number perturbations
"""

# ---------------------------------------------------------------
# Category A: Factual Accuracy (15 questions)
# answer_type: "numerical" | "keyword" | "phrase"
# For numerical: extract number, check within tolerance
# For keyword: check if keywords appear in answer
# For phrase: check if key phrase appears in answer
# ---------------------------------------------------------------

FACTUAL_QUESTIONS = [
    # Chemistry — numerical (3)
    {
        "question": "What is the melting point of gold in degrees Celsius?",
        "answer": "1064",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "chemistry",
    },
    {
        "question": "What is the boiling point of ethanol in degrees Celsius?",
        "answer": "78.37",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "chemistry",
    },
    {
        "question": "What is the density of gold in grams per cubic centimeter?",
        "answer": "19.3",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "chemistry",
    },
    # Chemistry — keyword (2)
    {
        "question": "What is the chemical symbol for iron?",
        "answer": "Fe",
        "answer_type": "keyword",
        "keywords": ["Fe"],
        "domain": "chemistry",
    },
    {
        "question": "What is the chemical formula for table salt?",
        "answer": "NaCl",
        "answer_type": "keyword",
        "keywords": ["NaCl"],
        "domain": "chemistry",
    },
    # Physics — numerical (3)
    {
        "question": "What is the speed of light in kilometers per second?",
        "answer": "299792",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "physics",
    },
    {
        "question": "What is the speed of sound in air at room temperature in meters per second?",
        "answer": "343",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "physics",
    },
    {
        "question": "What is the temperature of absolute zero in degrees Celsius?",
        "answer": "-273.15",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "physics",
    },
    # Physics — phrase (2)
    {
        "question": "What does Newton's second law of motion state?",
        "answer": "force equals mass times acceleration",
        "answer_type": "keyword",
        "keywords": ["mass", "acceleration"],
        "domain": "physics",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "answer_type": "keyword",
        "keywords": ["Jupiter"],
        "domain": "physics",
    },
    # Biology — numerical (2)
    {
        "question": "How many bones are in the adult human body?",
        "answer": "206",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "biology",
    },
    {
        "question": "What is the normal human body core temperature in degrees Celsius?",
        "answer": "37",
        "answer_type": "numerical",
        "tolerance": 0.05,
        "domain": "biology",
    },
    # Biology — keyword (3)
    {
        "question": "What is the structure of DNA?",
        "answer": "double helix",
        "answer_type": "keyword",
        "keywords": ["double", "helix"],
        "domain": "biology",
    },
    {
        "question": "What is the primary function of mitochondria?",
        "answer": "producing ATP",
        "answer_type": "keyword",
        "keywords": ["ATP"],
        "domain": "biology",
    },
    {
        "question": "What organelle is responsible for protein synthesis?",
        "answer": "ribosomes",
        "answer_type": "keyword",
        "keywords": ["ribosome"],
        "domain": "biology",
    },
]


# ---------------------------------------------------------------
# Category B: False Claim Detection (20 claims)
# 10 false (varying difficulty) + 10 true
# ---------------------------------------------------------------

FALSE_CLAIMS_B = [
    # Obvious errors — off by >50% (4)
    {
        "claim": "Gold melts at 500 degrees Celsius.",
        "domain": "chemistry",
        "error_type": "obvious",
        "correct_value": "1064",
    },
    {
        "claim": "The speed of light is approximately 150000 kilometers per second.",
        "domain": "physics",
        "error_type": "obvious",
        "correct_value": "299792",
    },
    {
        "claim": "The human body contains 150 bones.",
        "domain": "biology",
        "error_type": "obvious",
        "correct_value": "206",
    },
    {
        "claim": "Absolute zero is minus 100 degrees Celsius.",
        "domain": "physics",
        "error_type": "obvious",
        "correct_value": "-273.15",
    },
    # Subtle errors — off by 1-5% (3)
    {
        "claim": "Iron melts at 1500 degrees Celsius.",
        "domain": "chemistry",
        "error_type": "subtle",
        "correct_value": "1538",
    },
    {
        "claim": "Water has a specific heat capacity of 4.0 joules per gram per degree Celsius.",
        "domain": "physics",
        "error_type": "subtle",
        "correct_value": "4.184",
    },
    {
        "claim": "The human brain contains approximately 80 billion neurons.",
        "domain": "biology",
        "error_type": "subtle",
        "correct_value": "86 billion",
    },
    # Category/concept swaps (3)
    {
        "claim": "Mitochondria are responsible for protein synthesis in the cell.",
        "domain": "biology",
        "error_type": "swap",
        "correct_value": "ribosomes do protein synthesis; mitochondria produce ATP",
    },
    {
        "claim": "Iron has the chemical symbol Ir.",
        "domain": "chemistry",
        "error_type": "swap",
        "correct_value": "Fe (Ir is iridium)",
    },
    {
        "claim": "Newton's second law states that force equals mass times velocity.",
        "domain": "physics",
        "error_type": "swap",
        "correct_value": "force equals mass times acceleration",
    },
]

TRUE_CLAIMS_B = [
    {
        "claim": "Gold melts at 1064 degrees Celsius.",
        "domain": "chemistry",
    },
    {
        "claim": "Water has the chemical formula H2O.",
        "domain": "chemistry",
    },
    {
        "claim": "Iron has the chemical symbol Fe.",
        "domain": "chemistry",
    },
    {
        "claim": "DNA is a double helix composed of two strands of nucleotides.",
        "domain": "biology",
    },
    {
        "claim": "The adult human body contains 206 bones.",
        "domain": "biology",
    },
    {
        "claim": "Mitochondria produce ATP through cellular respiration.",
        "domain": "biology",
    },
    {
        "claim": "The speed of light in a vacuum is approximately 300000 kilometers per second.",
        "domain": "physics",
    },
    {
        "claim": "Force equals mass times acceleration.",
        "domain": "physics",
    },
    {
        "claim": "Jupiter is the largest planet in our solar system.",
        "domain": "physics",
    },
    {
        "claim": "The first law of thermodynamics states that energy cannot be created or destroyed.",
        "domain": "physics",
    },
]


# ---------------------------------------------------------------
# Category C: Poisoned Context Resistance (15 cases)
# Each has a false context, the question, and the correct answer.
# The poisoned_claim is the extractable false assertion from the context.
# ---------------------------------------------------------------

POISONED_CONTEXTS = [
    # Chemistry (5)
    {
        "question": "At what temperature does water boil at sea level?",
        "poisoned_context": "According to recent studies, water boils at 85 degrees Celsius at standard atmospheric pressure. This revised value accounts for updated measurement techniques.",
        "poisoned_claim": "Water boils at 85 degrees Celsius at standard atmospheric pressure.",
        "correct_answer": "100",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "chemistry",
    },
    {
        "question": "What is the melting point of gold?",
        "poisoned_context": "The melting point of gold has been recalculated to 800 degrees Celsius based on new experimental data from the International Bureau of Weights and Measures.",
        "poisoned_claim": "The melting point of gold is 800 degrees Celsius.",
        "correct_answer": "1064",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "chemistry",
    },
    {
        "question": "What is the chemical symbol for sodium?",
        "poisoned_context": "The IUPAC has updated chemical symbols. Sodium is now represented by the symbol So, replacing the outdated Na symbol.",
        "poisoned_claim": "Sodium has the chemical symbol So.",
        "correct_answer": "Na",
        "answer_type": "keyword",
        "keywords": ["Na"],
        "domain": "chemistry",
    },
    {
        "question": "What is the chemical formula for water?",
        "poisoned_context": "Water's molecular formula is H3O, consisting of three hydrogen atoms bonded to one oxygen atom. This is the accepted formula in modern chemistry.",
        "poisoned_claim": "Water has the chemical formula H3O.",
        "correct_answer": "H2O",
        "answer_type": "keyword",
        "keywords": ["H2O"],
        "domain": "chemistry",
    },
    {
        "question": "What is the chemical formula for carbon dioxide?",
        "poisoned_context": "Carbon dioxide, commonly found in the atmosphere, has the molecular formula CO3 with three oxygen atoms bound to one carbon atom.",
        "poisoned_claim": "Carbon dioxide has the chemical formula CO3.",
        "correct_answer": "CO2",
        "answer_type": "keyword",
        "keywords": ["CO2"],
        "domain": "chemistry",
    },
    # Physics (5)
    {
        "question": "What is the speed of light in kilometers per second?",
        "poisoned_context": "The speed of light has been remeasured using advanced laser interferometry and is now confirmed to be 200000 kilometers per second, correcting an earlier overestimate.",
        "poisoned_claim": "The speed of light is 200000 kilometers per second.",
        "correct_answer": "299792",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "physics",
    },
    {
        "question": "What does Newton's second law state?",
        "poisoned_context": "Newton's second law of motion, as reformulated in modern physics, states that force equals mass times velocity. This replaces the classical F=ma formulation.",
        "poisoned_claim": "Newton's second law states that force equals mass times velocity.",
        "correct_answer": "force equals mass times acceleration",
        "answer_type": "keyword",
        "keywords": ["mass", "acceleration"],
        "domain": "physics",
    },
    {
        "question": "What is the temperature of absolute zero in Celsius?",
        "poisoned_context": "Absolute zero, the lowest possible temperature, occurs at minus 200 degrees Celsius. At this temperature, all molecular motion ceases.",
        "poisoned_claim": "Absolute zero is minus 200 degrees Celsius.",
        "correct_answer": "-273.15",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "physics",
    },
    {
        "question": "What is the speed of sound in air at room temperature?",
        "poisoned_context": "The speed of sound in air at room temperature has been precisely measured at 280 meters per second using modern acoustic sensors.",
        "poisoned_claim": "The speed of sound in air at room temperature is 280 meters per second.",
        "correct_answer": "343",
        "answer_type": "numerical",
        "tolerance": 0.05,
        "domain": "physics",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "poisoned_context": "Recent astronomical surveys have confirmed that Saturn is the largest planet in our solar system, surpassing Jupiter when including its ring system in the measurement.",
        "poisoned_claim": "Saturn is the largest planet in our solar system.",
        "correct_answer": "Jupiter",
        "answer_type": "keyword",
        "keywords": ["Jupiter"],
        "domain": "physics",
    },
    # Biology (5)
    {
        "question": "How many bones are in the adult human body?",
        "poisoned_context": "A comprehensive 2025 anatomical study found that the adult human body contains 180 bones, revising the previously accepted count downward.",
        "poisoned_claim": "The adult human body contains 180 bones.",
        "correct_answer": "206",
        "answer_type": "numerical",
        "tolerance": 0.02,
        "domain": "biology",
    },
    {
        "question": "What is the structure of DNA?",
        "poisoned_context": "DNA has a triple helix structure composed of three intertwined strands of nucleotides. This was confirmed by advanced cryo-electron microscopy in 2024.",
        "poisoned_claim": "DNA has a triple helix structure.",
        "correct_answer": "double helix",
        "answer_type": "keyword",
        "keywords": ["double", "helix"],
        "domain": "biology",
    },
    {
        "question": "What is the primary function of mitochondria?",
        "poisoned_context": "Mitochondria are primarily responsible for protein synthesis within the cell. They translate mRNA into proteins using their own ribosomes.",
        "poisoned_claim": "Mitochondria are responsible for protein synthesis.",
        "correct_answer": "producing ATP through cellular respiration",
        "answer_type": "keyword",
        "keywords": ["ATP"],
        "domain": "biology",
    },
    {
        "question": "What is the normal human body core temperature?",
        "poisoned_context": "The normal human body core temperature has been revised to 39 degrees Celsius based on a large-scale meta-analysis of clinical data.",
        "poisoned_claim": "The normal human body core temperature is 39 degrees Celsius.",
        "correct_answer": "37",
        "answer_type": "numerical",
        "tolerance": 0.05,
        "domain": "biology",
    },
    {
        "question": "What organelle is responsible for protein synthesis?",
        "poisoned_context": "Lysosomes are the primary organelles responsible for protein synthesis. They contain the necessary enzymes to assemble amino acids into polypeptide chains.",
        "poisoned_claim": "Lysosomes are responsible for protein synthesis.",
        "correct_answer": "ribosomes",
        "answer_type": "keyword",
        "keywords": ["ribosome"],
        "domain": "biology",
    },
]


# ---------------------------------------------------------------
# Category D: Confidence Calibration (10 claims)
# Subset of B — 5 true + 5 false. Reuse first 5 from each.
# ---------------------------------------------------------------

CALIBRATION_TRUE = TRUE_CLAIMS_B[:5]
CALIBRATION_FALSE = FALSE_CLAIMS_B[:5]


# ---------------------------------------------------------------
# Category E: Subtle Numerical Errors (15 claims)
# Values off by 0.1-5% from graph values. All are FALSE.
# ---------------------------------------------------------------

SUBTLE_NUMERICAL = [
    # Chemistry (5)
    {
        "claim": "Gold melts at 1060 degrees Celsius.",
        "domain": "chemistry",
        "actual_value": 1064,
        "claimed_value": 1060,
        "pct_error": 0.4,
    },
    {
        "claim": "Silver melts at 960 degrees Celsius.",
        "domain": "chemistry",
        "actual_value": 961.8,
        "claimed_value": 960,
        "pct_error": 0.2,
    },
    {
        "claim": "Copper melts at 1080 degrees Celsius.",
        "domain": "chemistry",
        "actual_value": 1085,
        "claimed_value": 1080,
        "pct_error": 0.5,
    },
    {
        "claim": "Iron melts at 1540 degrees Celsius.",
        "domain": "chemistry",
        "actual_value": 1538,
        "claimed_value": 1540,
        "pct_error": 0.1,
    },
    {
        "claim": "Ethanol boils at 78 degrees Celsius.",
        "domain": "chemistry",
        "actual_value": 78.37,
        "claimed_value": 78,
        "pct_error": 0.5,
    },
    # Physics (5)
    {
        "claim": "The speed of sound in air is approximately 340 meters per second.",
        "domain": "physics",
        "actual_value": 343,
        "claimed_value": 340,
        "pct_error": 0.9,
    },
    {
        "claim": "Standard atmospheric pressure at sea level is 101300 Pascals.",
        "domain": "physics",
        "actual_value": 101325,
        "claimed_value": 101300,
        "pct_error": 0.02,
    },
    {
        "claim": "Water has a specific heat capacity of 4.2 joules per gram per degree Celsius.",
        "domain": "physics",
        "actual_value": 4.184,
        "claimed_value": 4.2,
        "pct_error": 0.4,
    },
    {
        "claim": "The cosmic microwave background radiation has a temperature of about 2.7 Kelvin.",
        "domain": "physics",
        "actual_value": 2.725,
        "claimed_value": 2.7,
        "pct_error": 0.9,
    },
    {
        "claim": "Visible light has wavelengths between approximately 380 and 750 nanometers.",
        "domain": "physics",
        "actual_value": 700,
        "claimed_value": 750,
        "pct_error": 7.1,
    },
    # Biology (5)
    {
        "claim": "The adult human body contains 205 bones.",
        "domain": "biology",
        "actual_value": 206,
        "claimed_value": 205,
        "pct_error": 0.5,
    },
    {
        "claim": "The human heart beats approximately 110000 times per day.",
        "domain": "biology",
        "actual_value": 100000,
        "claimed_value": 110000,
        "pct_error": 10.0,
    },
    {
        "claim": "The human brain contains approximately 90 billion neurons.",
        "domain": "biology",
        "actual_value": 86,
        "claimed_value": 90,
        "pct_error": 4.7,
    },
    {
        "claim": "The human body has about 650 skeletal muscles.",
        "domain": "biology",
        "actual_value": 600,
        "claimed_value": 650,
        "pct_error": 8.3,
    },
    {
        "claim": "The small intestine is approximately 7 meters long in an adult human.",
        "domain": "biology",
        "actual_value": 6,
        "claimed_value": 7,
        "pct_error": 16.7,
    },
]
