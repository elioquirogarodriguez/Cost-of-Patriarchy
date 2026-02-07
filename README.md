# Technological Lag Visualization for "The Opportunity Cost of Patriarchy"

This repository contains Python code and output files for generating the visualizations presented in the paper **"The Opportunity Cost of Patriarchy: A Quantitative Analysis of Civilisational Inefficiency"**. The scripts create two complementary visual representations of the technological development gap between patriarchal and egalitarian civilizational trajectories, based on the mathematical model developed in the paper.

## 📊 Overview

The visualization consists of:
1. **Plot (PNG)**: A two-panel figure showing:
   - **Panel A**: Technological development level over historical time (3000 BCE to 2026 CE) for both patriarchal (actual) and egalitarian (counterfactual) scenarios
   - **Panel B**: Equivalent development years, illustrating the temporal lag between scenarios
2. **Summary (TXT)**: A text file with key quantitative findings derived from the model

## 🧮 Model Basis

The visualization implements the mathematical model from Section 9 of the paper:
- Innovation rate: `I(t) = k · P(t) · E(t) · D(t)`
- Patriarchal scenario operates at **15% of potential capacity** (0.5 population access × 0.3 diversity factor)
- Egalitarian scenario operates at **100% of potential capacity**
- Analysis period: 3000 BCE to 2026 CE (5,026 years)

## 📁 File Structure
├── plot_patriarchy_cost.py # Main visualization script
├── patriarchy_opportunity_cost_visualization.png # Generated plot
├── technological_lag_summary.txt # Generated summary statistics
└── README.md # This file

text

## 🛠️ Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install required packages:
```bash
pip install numpy matplotlib
▶️ Usage

Run the visualization script:

bash
python plot_patriarchy_cost.py
The script will:

Generate the two-panel visualization
Save it as patriarchy_opportunity_cost_visualization.png
Generate and save summary statistics as technological_lag_summary.txt
Display the plot in a window (if running interactively)
📈 Key Outputs

Visualization (patriarchy_opportunity_cost_visualization.png)

Panel A: Shows the growing divergence in technological development between scenarios
Panel B: Translates this into equivalent years, showing that in 2026:

Patriarchal reality: Development equivalent to year ~-474 CE in egalitarian timeline
Egalitarian counterfactual: Development equivalent to year 4526 CE
Grey areas: Represent the accumulated opportunity cost/lost development
Summary (technological_lag_summary.txt)

Contains:

Model parameters and assumptions
Key quantitative findings
Interpretation of the 2,500-year technological lag
Cumulative opportunity cost metrics
🔍 Key Findings from the Model

Technological Lag: Approximately 2,500 years of delayed development
Efficiency Loss: Patriarchal structures operate at ~15% of potential innovation capacity
Temporal Displacement: Current (2026) development in patriarchal reality corresponds to what would have been achieved by ~-474 CE under egalitarian conditions
Lost Potential: With gender equity, 2026 would exhibit development equivalent to year 4526 CE
📝 Model Assumptions & Limitations

Simplified Growth: Uses normalized exponential population and sigmoidal education quality models
Constant Rates: Assumes constant innovation rate ratios across entire historical period
Normalized Units: Technological development measured in arbitrary normalized units
Counterfactual Nature: Egalitarian scenario represents an idealized trajectory
📚 Citation

If using this visualization or model in your work, please cite:

Caja Moya, C., & Quiroga Rodríguez, E. (2026). The Opportunity Cost of Patriarchy: A Quantitative Analysis of Civilisational Inefficiency. Universidad del Atlántico Medio.
📄 License

This code and its outputs are provided for academic and research purposes. Please contact the authors for usage permissions beyond educational applications.

✉️ Contact

For questions about the model or visualization:

Elio Quiroga Rodríguez: elio.quiroga@pdi.atlanticomedio.es
Dra. Cristina Caja Moya: cristina.caja@pdi.atlanticomedio.es
