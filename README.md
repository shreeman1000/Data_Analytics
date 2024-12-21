# Analytical Assignments Code Repository

This repository contains implementations for solving various analytical and computational problems across multiple domains, including network analysis, planetary motion, biology, cricket, and genomics. Each folder corresponds to a specific assignment.

---

## Contents

1. [Girvan-Newman Algorithm for Community Detection (Assignment 1)](#1-girvan-newman-algorithm-for-community-detection-assignment-1)
2. [Mars Equant Model for Planetary Motion (Assignment 2)](#2-mars-equant-model-for-planetary-motion-assignment-2)
3. [Smoking and Gender Interaction Analysis (Assignment 3)](#3-smoking-and-gender-interaction-analysis-assignment-3)
4. [Duckworth-Lewis Model for Cricket (Assignment 4)](#4-duckworth-lewis-model-for-cricket-assignment-4)
5. [Genome Sequence Alignment (Assignment 5)](#5-genome-sequence-alignment-assignment-5)

---

## 1. Girvan-Newman Algorithm for Community Detection (Assignment 1)

### Objective
Detect community structure in complex networks using the Girvan-Newman algorithm.

### Methodology
- **Concept**: Identifies communities by iteratively removing edges with the highest edge betweenness (i.e., the number of shortest paths passing through an edge).
- **Steps**:
  1. Calculate edge betweenness for all edges in the graph.
  2. Remove the edge with the highest betweenness.
  3. Repeat until the graph is divided into disconnected components (communities).

---

## 2. Mars Equant Model for Planetary Motion (Assignment 2)

### Objective
Model the position of Mars at specific observational times (oppositions) using the Mars Equant Model.

### Assumptions
- **Sun-Centric Orbit**:
  - The Sun is at the origin.
  - Mars's orbit is circular, centered at a distance of 1 unit from the Sun, with the center at an angle \(c\) degrees from the Sun-Aries reference line.
- **Orbit and Equant**:
  - The orbit radius is \(r\) (in units of the Sun-center distance).
  - The equant is located at \((e_1, e_2)\) in polar coordinates relative to the Sun.
  - The 'equant 0' angle \(z\) (degrees) represents the earliest opposition, taken as the reference time \(t=0\).
- **Motion**:
  - Mars moves around the equant with an angular velocity \(s\) degrees per day.

### Methodology
- Derived the equations governing the angular position of Mars at different times based on the equant model.
- Compared modeled positions with observed positions to validate the model.

---

## 3. Smoking and Gender Interaction Analysis (Assignment 3)

### Objective
Identify genes that respond differently to smoking in men vs. women using a 2-way ANOVA framework.

### Methodology
- **Data**: Expression data from white blood cells of 48 individuals.
- **Statistical Test**: Computed F-statistics to evaluate the significance of interactions between gender and smoking status.
- **Visualization**: Generated a histogram of p-values.

---

## 4. Duckworth-Lewis Model for Cricket (Assignment 4)

### Objective
Model and predict the number of runs remaining in cricket games using the Duckworth-Lewis (DL) method.

### Methodology
- **Data Preprocessing**: Loaded cricket data from 1999 to 2011, filtered missing entries, and transformed overs bowled to overs remaining.
- **Model**: Implemented a DL model with:
  \[
  Z = Z_0 \cdot \left(1 - e^{-\frac{L \cdot X}{Z_0}}\right)
  \]
  where \(Z_0\) is the initial possible score, \(L\) is the decay factor, and \(X\) is the overs remaining.
- **Training**: Optimized parameters \(Z_0\) and \(L\) using a custom logarithmic loss function.

### Results
- **Average Loss**: 58.891 (final run).
- **Parameters**:
  - \(Z_0\): [6.222, 18.811, ..., 273.848]
  - \(L\): 10.647.

---

## 5. Genome Sequence Alignment (Assignment 5)

### Objective
Align genome sequence reads to the human X chromosome to analyze genes associated with color vision and identify configurations linked to color blindness.

### Methodology
- **Alignment**: Used the Burrows-Wheeler Transform (BWT) to align 3 million reads to the X chromosome, allowing up to two mismatches.
- **Exon Mapping**: Counted reads mapping to exons of the red and green opsin genes, assigning weights to ambiguous reads.
- **Configuration Analysis**: Computed probabilities for different gene configurations to identify the most likely cause of observed read patterns.

### Results
- **Most Probable Configuration**: Configuration 3, with likelihood \(1.95 \times 10^{-17}\).

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
