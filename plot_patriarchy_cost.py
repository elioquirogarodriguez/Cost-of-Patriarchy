#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of Technological Development: Patriarchal vs. Egalitarian Scenarios
Based on "The Opportunity Cost of Patriarchy" mathematical model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

# Define time range (years from 3000 BCE to 2026 CE)
start_year = -3000  # 3000 BCE
end_year = 2026
years = np.linspace(start_year, end_year, 1000)

# Model parameters based on the paper
# Under patriarchy: P ≈ 0.5 * P_potential, D ≈ 0.3 * D_potential
# Combined effect: 0.5 * 0.3 = 0.15 (operating at 15% capacity)
patriarchal_rate = 0.15
egalitarian_rate = 1.0

# Base innovation rate (arbitrary units, normalized)
k_base = 1.0

# Population growth (simplified exponential model)
# P(t) = P0 * exp(r*t) where r is growth rate
def population_factor(year):
    """Population growth over time (normalized)"""
    # Normalize to start_year = 0
    t_norm = (year - start_year) / (end_year - start_year)
    # Accelerating population growth
    return 0.1 + 0.9 * (1 - np.exp(-3 * t_norm))

# Educational infrastructure quality (improves over time)
def education_factor(year):
    """Educational infrastructure quality over time"""
    t_norm = (year - start_year) / (end_year - start_year)
    # Sigmoid growth for education quality
    return 1 / (1 + np.exp(-10 * (t_norm - 0.5)))

# Innovation rate: I(t) = k * P(t) * E(t) * scenario_rate
def innovation_rate(year, scenario_rate):
    """Calculate innovation rate at a given year"""
    P_t = population_factor(year)
    E_t = education_factor(year)
    return k_base * P_t * E_t * scenario_rate

# Calculate technological level as integral of innovation
# T(t) = T0 + ∫ I(τ) dτ
def technological_level(years_array, scenario_rate):
    """Calculate technological development level over time"""
    tech_level = np.zeros_like(years_array)
    dt = years_array[1] - years_array[0]
    
    for i, year in enumerate(years_array):
        if i == 0:
            tech_level[i] = 1.0  # Starting point (normalized)
        else:
            # Accumulate innovation
            innovation = innovation_rate(year, scenario_rate)
            # Simple accumulation without excessive compounding
            tech_level[i] = tech_level[i-1] + innovation * dt
    
    return tech_level

# Calculate both scenarios
tech_patriarchal = technological_level(years, patriarchal_rate)
tech_egalitarian = technological_level(years, egalitarian_rate)

# Normalize to make 2026 patriarchal scenario = 2026 "equivalent years"
current_tech_patriarchal = tech_patriarchal[-1]
current_tech_egalitarian = tech_egalitarian[-1]

# Based on the paper: ~2000-2500 year technological lag
# This means in 2026, we have the tech level that would have been reached
# around year 500 BCE to 26 CE in an egalitarian scenario
ESTIMATED_LAG_YEARS = 2500  # From the paper

# Calculate what year in egalitarian scenario has our current patriarchal tech
# We'll use the ratio approach
tech_ratio = current_tech_patriarchal / current_tech_egalitarian

# Find the year in egalitarian timeline that matches current patriarchal tech
egalitarian_year_equivalent = years[-1] - ESTIMATED_LAG_YEARS

# For the equivalent years plot, we scale based on development rate
# In egalitarian scenario, development is faster, so equivalent year > calendar year
advancement_factor = 1.0 / patriarchal_rate  # Should be ~6.67x faster

# The current year 2026 in egalitarian scenario would be equivalent to 
# a much more advanced year in terms of development
current_egal_equivalent = years[-1] + ESTIMATED_LAG_YEARS

# Convert technological levels to "equivalent years of development"
# Patriarchal scenario: maps to actual calendar years (1:1)
equivalent_years_patriarchal = years.copy()

# Egalitarian scenario: 
# We're asking "if we reach technology level X in year Y under egalitarianism,
# what year does that tech level X represent in terms of advancement?"
# Since we're advancing faster, we reach higher tech levels sooner
# So the "equivalent development year" is AHEAD of the calendar year

# The key insight: in 2026 under egalitarianism, we'd have the tech level
# that patriarchal society would only reach in year 2026 + LAG_YEARS
equivalent_years_egalitarian = years + ESTIMATED_LAG_YEARS

# Create the main visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('The Opportunity Cost of Patriarchy:\nTechnological Development Trajectories (3000 BCE - 2026 CE)',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Technological Level over Calendar Time
ax1.plot(years, tech_patriarchal, 'r-', linewidth=2.5, label='Patriarchal Scenario (Historical Reality)', alpha=0.8)
ax1.plot(years, tech_egalitarian, 'b-', linewidth=2.5, label='Egalitarian Scenario (Counterfactual)', alpha=0.8)

# Fill the gap between scenarios
ax1.fill_between(years, tech_patriarchal, tech_egalitarian, alpha=0.2, color='orange',
                 label='Lost Development (Opportunity Cost)')

# Mark 2026
ax1.axvline(x=2026, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Year 2026')

# Mark the technological lag
ax1.axhline(y=current_tech_patriarchal, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax1.axvline(x=egalitarian_year_equivalent, color='blue', linestyle=':', linewidth=1, alpha=0.5)

# Add annotation for the lag
ax1.annotate('', xy=(egalitarian_year_equivalent, current_tech_patriarchal * 0.95),
            xytext=(2026, current_tech_patriarchal * 0.95),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax1.text((egalitarian_year_equivalent + 2026) / 2, current_tech_patriarchal * 0.92,
         f'~{ESTIMATED_LAG_YEARS} year\ntechnological lag', ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')

# Mark key historical periods
periods = [
    (-3000, 'Neolithic'),
    (-800, 'Classical\nGreece'),
    (1100, 'Medieval'),
    (1650, 'Scientific\nRevolution'),
    (1800, 'Industrial\nRevolution'),
]
for year, label in periods:
    ax1.axvline(x=year, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.text(year, ax1.get_ylim()[1] * 0.05, label, rotation=0, 
            fontsize=8, ha='center', alpha=0.6)

ax1.set_xlabel('Calendar Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Technological Development Level (Arbitrary Units)', fontsize=12, fontweight='bold')
ax1.set_title('A. Technological Level vs. Calendar Time', fontsize=13, fontweight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Format x-axis to show BCE/CE
def format_year(x, p):
    if x < 0:
        return f'{int(-x)} BCE'
    elif x == 0:
        return '0'
    else:
        return f'{int(x)} CE'

ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_year))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Equivalent Development Years over Calendar Time
ax2.plot(years, equivalent_years_patriarchal, 'r-', linewidth=2.5, 
         label='Patriarchal Scenario (1:1 ratio)', alpha=0.8)
ax2.plot(years, equivalent_years_egalitarian, 'b-', linewidth=2.5,
         label='Egalitarian Scenario (Advanced)', alpha=0.8)

# Fill the gap
ax2.fill_between(years, equivalent_years_patriarchal, equivalent_years_egalitarian,
                 alpha=0.2, color='orange', label='Development Gap')

# Mark 2026
ax2.axvline(x=2026, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax2.axhline(y=2026, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# Calculate what equivalent year we'd be in egalitarian scenario
current_egal_equivalent = equivalent_years_egalitarian[-1]

# Add annotation for current status
ax2.scatter([2026], [2026], color='red', s=150, zorder=5, marker='o',
           edgecolors='darkred', linewidth=2, label='2026 Patriarchal Reality')
ax2.scatter([2026], [current_egal_equivalent], color='blue', s=150, zorder=5, marker='s',
           edgecolors='darkblue', linewidth=2, label='2026 Egalitarian Equivalent')

# Arrow showing the advancement
ax2.annotate('', xy=(2026, current_egal_equivalent), xytext=(2026, 2026),
            arrowprops=dict(arrowstyle='->', color='purple', lw=3))
ax2.text(2026 + 200, (2026 + current_egal_equivalent) / 2,
         f'+{ESTIMATED_LAG_YEARS} years\nof advancement\n(if egalitarian)',
         fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontweight='bold', va='center')

# Add diagonal reference line (1:1)
ax2.plot([start_year, end_year], [start_year, end_year], 'k--', 
         linewidth=1, alpha=0.3, label='1:1 Reference (No lag)')

ax2.set_xlabel('Calendar Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Equivalent Development Year', fontsize=12, fontweight='bold')
ax2.set_title('B. Equivalent Development Years: How Advanced We Would Be', 
              fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Format both axes
ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_year))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_year))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()

# Save the figure
output_path = '/mnt/user-data/outputs/patriarchy_opportunity_cost_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Visualization saved to: {output_path}")

# Create summary statistics text file
summary_path = '/mnt/user-data/outputs/technological_lag_summary.txt'
with open(summary_path, 'w') as f:
    f.write("TECHNOLOGICAL DEVELOPMENT ANALYSIS SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write("Based on the mathematical model from:\n")
    f.write("'The Opportunity Cost of Patriarchy: A Quantitative Analysis\n")
    f.write("of Civilisational Inefficiency'\n\n")
    f.write("MODEL PARAMETERS:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Patriarchal innovation rate: {patriarchal_rate:.0%} of potential\n")
    f.write(f"Egalitarian innovation rate: {egalitarian_rate:.0%} of potential\n")
    f.write(f"Analysis period: {start_year} to {end_year}\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Current year (calendar): 2026 CE\n")
    f.write(f"Technological lag: {ESTIMATED_LAG_YEARS} years (from paper estimate)\n")
    f.write(f"Egalitarian equivalent year: {int(current_egal_equivalent)} CE\n")
    f.write(f"Years of lost advancement: {int(current_egal_equivalent - 2026)}\n\n")
    
    f.write("INTERPRETATION:\n")
    f.write("-" * 60 + "\n")
    f.write("In 2026 under patriarchal conditions, our technological development\n")
    f.write(f"is equivalent to what the year ~{int(egalitarian_year_equivalent)} CE would have been\n")
    f.write("in an egalitarian scenario (approximately 2,500 years behind).\n\n")
    f.write(f"Conversely, if we had maintained gender equity throughout history,\n")
    f.write(f"our current (2026) development level would be equivalent to the\n")
    f.write(f"year {int(current_egal_equivalent)} CE - approximately {ESTIMATED_LAG_YEARS} years\n")
    f.write("more advanced than we currently are.\n\n")
    
    f.write("This represents a ~{:.1f}% reduction in development rate due to\n".format(
        (1 - patriarchal_rate/egalitarian_rate) * 100))
    f.write("systematic gender-based exclusion from innovation processes.\n\n")
    
    f.write("CUMULATIVE OPPORTUNITY COST:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Total lost development: {tech_egalitarian[-1] - tech_patriarchal[-1]:.2f} units\n")
    f.write(f"Lost development as % of potential: {((tech_egalitarian[-1] - tech_patriarchal[-1]) / tech_egalitarian[-1] * 100):.1f}%\n")
    f.write(f"Efficiency ratio (actual/potential): {(tech_patriarchal[-1] / tech_egalitarian[-1]):.2%}\n")

print(f"Summary statistics saved to: {summary_path}")

# Print summary to console
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print(f"\nTechnological lag (from paper): {ESTIMATED_LAG_YEARS} years")
print(f"2026 would be equivalent to year: {int(current_egal_equivalent)} CE")
print(f"Years of advancement lost: {ESTIMATED_LAG_YEARS}")
print(f"\nOperating at {patriarchal_rate:.0%} of potential capacity")
print(f"Development efficiency ratio: {patriarchal_rate/egalitarian_rate:.1%}")
print("=" * 60)

plt.show()
