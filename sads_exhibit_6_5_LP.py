# Consumer Preferences for Sporting Events---Conjoint Analysis (Python)

# prepare for Python version 3x features and functions
from __future__ import division, print_function

# import packages for analysis and modeling
import pandas as pd  # data frame operations
import numpy as np  # arrays and math functions
import statsmodels.formula.api as smf  # R-like model specification
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# read in conjoint survey profiles with respondent ranks
conjoint_data_frame = pd.read_csv('sporting_event_ranking.csv')

# set up sum contrasts for effects coding as needed for conjoint analysis
# using C(effect, Sum) notation within main effects model specification
main_effects_model = 'ranking ~ C(price, Sum) + C(seating, Sum) +  \
    C(boxvip, Sum) + C(frontrow, Sum) + C(promotion, Sum)'

# fit linear regression model using main effects only (no interaction terms)
main_effects_model_fit = \
    smf.ols(main_effects_model, data=conjoint_data_frame).fit()
print(main_effects_model_fit.summary())
conjoint_attributes = ['price', 'seating', 'boxvip', 'frontrow', 'promotion']

# build part-worth information one attribute at a time
level_name = []
part_worth = []
part_worth_range = []
end = 1  # initialize index for coefficient in params
for item in conjoint_attributes:
    level_set = set(conjoint_data_frame[item])
    nlevels = len(level_set)

    # Convert all elements to strings before sorting
    level_name.append(sorted(list(level_set), key=lambda x: str(x)))

    begin = end
    end = begin + nlevels - 1
    new_part_worth = list(main_effects_model_fit.params[begin:end])
    new_part_worth.append((-1) * sum(new_part_worth))
    part_worth_range.append(max(new_part_worth) - min(new_part_worth))
    part_worth.append(new_part_worth)
    # end set to begin next iteration

# compute attribute relative importance values from ranges
attribute_importance = []
for item in part_worth_range:
    attribute_importance.append(round(100 * (item / sum(part_worth_range)), 2))
# user-defined dictionary for printing descriptive attribute names
effect_name_dict = {'price': 'Ticket Price', \
                    'seating': 'Seating Area', 'boxvip': 'Box/VIP', \
                    'frontrow': 'Front Row', 'promotion': 'Promotion'}

# report conjoint measures to console
index = 0  # initialize for use in for-loop
for item in conjoint_attributes:
    print('\nAttribute:', effect_name_dict[item])
    print('    Importance:', attribute_importance[index])
    print('    Level Part-Worths')
    for level in range(len(level_name[index])):
        if level < len(part_worth[index]):
            print('       ', level_name[index][level], part_worth[index][level])
        else:
            # Calculate the last part-worth as the negative sum of the others
            last_part_worth = -sum(part_worth[index])
            print('       ', level_name[index][level], last_part_worth)
    index = index + 1

# Bar Charts for Attribute Importance
plt.figure(figsize=(10, 6))
plt.bar(effect_name_dict.values(), attribute_importance)
plt.title('Relative Importance of Attributes')
plt.xlabel('Attributes')
plt.ylabel('Importance (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'attribute_importance_bar_chart.png'))
plt.close()

# Heatmap for Part-Worth Utilities
max_levels = max(len(pw) for pw in part_worth)
part_worth_df = pd.DataFrame(
    [pw + [np.nan] * (max_levels - len(pw)) for pw in part_worth],
    index=effect_name_dict.values(),
    columns=[f'Level {i + 1}' for i in range(max_levels)]
)

plt.figure(figsize=(12, 8))
sns.heatmap(part_worth_df, annot=True, cmap='RdYlGn', center=0, fmt='.3f', cbar_kws={'label': 'Part-Worth Utility'})
plt.title('Part-Worth Utilities Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'part_worth_utilities_heatmap.png'))
plt.close()

# Grouped Bar Chart for Part-Worth Utilities
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(effect_name_dict))
width = 0.8 / max_levels

for i in range(max_levels):
    values = [pw[i] if i < len(pw) else np.nan for pw in part_worth]
    ax.bar(x + (i - max_levels / 2 + 0.5) * width, values, width, label=f'Level {i + 1}')

ax.set_xticks(x)
ax.set_xticklabels(effect_name_dict.values(), rotation=45, ha='right')
ax.legend()

plt.title('Part-Worth Utilities by Attribute Level')
plt.xlabel('Attributes')
plt.ylabel('Part-Worth Utility')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'part_worth_utilities_grouped_bar_chart.png'))
plt.close()

# Radar Chart for Attribute Importance
attributes = list(effect_name_dict.values())
importance = attribute_importance

angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False)
importance = np.concatenate((importance, [importance[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, importance)
ax.fill(angles, importance, alpha=0.25)
ax.set_thetagrids(angles[:-1] * 180 / np.pi, attributes)
ax.set_title('Attribute Importance')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'attribute_importance_radar_chart.png'))
plt.close()

# Horizontal Bar Chart for Part-Worth Utilities
fig, ax = plt.subplots(figsize=(12, 10))

utilities = []
labels = []
for attr, levels, pw in zip(effect_name_dict.values(), level_name, part_worth):
    for level, util in zip(levels, pw):
        utilities.append(util)
        labels.append(f"{attr} - {level}")

y_pos = np.arange(len(utilities))

ax.barh(y_pos, utilities)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Part-Worth Utility')
ax.set_title('Part-Worth Utilities for All Attribute Levels')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'part_worth_utilities_horizontal_bar_chart.png'))
plt.close()


# ************
# Set initial prices for each section
# ************

import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the attributes and their importance
attributes = {
    'Seating Area': attribute_importance[1],
    'Ticket Price': attribute_importance[0],
    'Promotion': attribute_importance[4],
    'Box/VIP': attribute_importance[2],
    'Front Row': attribute_importance[3]
}

# Define the part-worth utilities for each attribute level
seating_utilities = {
    'Field': part_worth[1][0],
    'Loge': part_worth[1][1],
    'Reserved': part_worth[1][2],
    'Top Deck': part_worth[1][3]
}

price_utilities = {
    'Price $20': part_worth[0][0],
    'Price $45': part_worth[0][1],
    'Price $70': part_worth[0][2],
    'Price $95': part_worth[0][3]
}

# Define sections based on Dodger Stadium's actual layout
sections = {
    'Field': {'base_price': 95, 'capacity': 15000},
    'Loge': {'base_price': 70, 'capacity': 15000},
    'Reserved': {'base_price': 45, 'capacity': 15000},
    'Top Deck': {'base_price': 20, 'capacity': 11000}
}

# Calculate the utility score for each section
for section in sections:
    utility_score = seating_utilities[section]
    sections[section]['utility_score'] = utility_score

# Normalize utility scores
max_utility = max(section['utility_score'] for section in sections.values())
min_utility = min(section['utility_score'] for section in sections.values())

for section in sections:
    sections[section]['normalized_utility'] = (sections[section]['utility_score'] - min_utility) / (
                max_utility - min_utility)

# Define coefficients for linear programming
# Objective function: maximize revenue
c = [-section['base_price'] * section['capacity'] for section in sections.values()]  # negative for maximization

# Bounds for each section's initial price
bounds = [(section['base_price'] * 0.5, section['base_price'] * 1.5) for section in sections.values()]

# Solve linear programming problem
result = opt.linprog(c,bounds=bounds, method='highs')
# result = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if result.success:
    optimal_prices = result.x
    section_names = list(sections.keys())
    for i, section in enumerate(section_names):
        sections[section]['optimal_price'] = optimal_prices[i]

    print("Optimal prices calculated:")
    for section, data in sections.items():
        print(f"{section}: ${data.get('optimal_price', 'Price not calculated'):.2f}")
else:
    print("Linear programming failed to find an optimal solution.")

# Calculate potential revenue with optimal prices
total_potential_revenue = sum(
    sections[section].get('optimal_price', 0) * sections[section]['capacity'] for section in sections
)
print(f"\nTotal potential revenue: ${total_potential_revenue:,.2f}")

# Plotting results
plt.figure(figsize=(10, 6))
plt.bar(sections.keys(), [data.get('optimal_price', 0) for data in sections.values()])
plt.title('Optimal Ticket Prices')
plt.xlabel('Section')
plt.ylabel('Price ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'optimal_ticket_prices_LP.png'))
plt.close()


# ************
# Dynamic Pricing Simulation
# ************

# Use initial prices from the conjoint analysis result for each section
# These initial prices were previously calculated in the conjoint analysis
# For simplicity, assume the initial_price is calculated and included in sections

# Sections from the original model (with initial_price now included)
sections = {
    'Field': {'initial_price': result.x[0], 'capacity': 15000, 'utility': part_worth[1][0]},  # example initial_price
    'Loge': {'initial_price': result.x[1], 'capacity': 15000, 'utility': part_worth[1][1]},  # example initial_price
    'Reserved': {'initial_price': result.x[2], 'capacity': 15000, 'utility': part_worth[1][2]},  # example initial_price
    'Top Deck': {'initial_price': result.x[3], 'capacity': 11000, 'utility': part_worth[1][3]},  # example initial_price
}

# Set up sales velocity simulation
time_to_event = 30  # Days until the event
expected_velocity = 0.05  # Expected percentage of tickets sold per day (5%)
scaling_factor = 0.1  # Scaling factor for price adjustments
price_history = {section: [] for section in sections}

# Sales velocity simulation for each section
np.random.seed(42)  # For reproducibility
for day in range(time_to_event):
    print(f"\nDay {day + 1}")

    for section in sections:
        # Adjust prices dynamically based on sales velocity
        sales_velocity = np.random.uniform(0.01, 0.1)  # Simulated percentage of tickets sold
        capacity = sections[section]['capacity']
        remaining_tickets = capacity if day == 0 else sections[section]['remaining_tickets']

        # Calculate price adjustment factor based on sales velocity
        adjustment_factor = 1 + (sales_velocity / expected_velocity) * scaling_factor

        # Adjust price based on time and velocity
        time_factor = (1 + 1 / (time_to_event - day + 1))  # Prices increase closer to the event
        new_price = sections[section]['initial_price'] * adjustment_factor * time_factor

        # Simulate ticket sales for the day
        tickets_sold = int(min(sales_velocity * remaining_tickets, remaining_tickets))

        # Update section details
        sections[section]['remaining_tickets'] = remaining_tickets - tickets_sold
        sections[section]['current_price'] = round(new_price, 2)

        # Store price and sales history
        price_history[section].append({
            'Day': day + 1,
            'Price': round(new_price, 2),
            'Tickets Sold': tickets_sold,
            'Remaining Tickets': sections[section]['remaining_tickets']
        })

        print(
            f"Section: {section}, Price: ${round(new_price, 2)}, Tickets Sold: {tickets_sold}, Remaining Tickets: {sections[section]['remaining_tickets']}")

        # Break loop if all tickets are sold
        if sections[section]['remaining_tickets'] <= 0:
            print(f"All tickets sold for {section} section!")
            break

# Convert price history to DataFrame for analysis
price_history_dfs = {section: pd.DataFrame(history) for section, history in price_history.items()}

# Print final data and summary
for section in sections:
    print(f"\nFinal summary for {section} section:")
    print(price_history_dfs[section])

# Optional: Calculate total potential revenue
total_potential_revenue = sum(
    sections[section]['current_price'] * sections[section]['remaining_tickets'] for section in sections
)
print(f"\nTotal potential revenue: ${total_potential_revenue:,.2f}")




