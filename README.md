This project conducts a conjoint analysis to evaluate consumer preferences for sporting events, focusing on key factors such as ticket price, seating area, VIP/box options, front row availability, and promotional offers. The customer preference data set and initial code are from Exhibit 6, “Sports Analytics and Data Science” by Thomas W. Miller. The analysis estimates part-worth utilities for these attributes and derives their relative importance in influencing consumer preferences. It includes a dynamic pricing simulation based on these preferences and projected ticket sales. Key features:

Conjoint Analysis: Uses linear regression to model consumer rankings of sporting event tickets based on key attributes.

Part-Worth Calculation: Extracts part-worth utilities for each level of the defined attributes and computes attribute importance.

Visualization: Generates bar charts, heatmaps, radar charts, and grouped bar charts for visualizing the importance and utilities of different attributes.

Base Case - Optimal Pricing: Calculates initial ticket prices based on consumer utility scores for different seating sections
(sads_exhibit_6_5_a.py).

Alternate Case - Optimal Pricing: Utilizes linear programming to determine optimal ticket prices for different stadium sections (sads_exhibit_6_5_LP.py).

Dynamic Pricing Simulation: Simulates dynamic pricing adjustments leading up to the event, factoring in ticket sales velocity
and remaining capacity.

Revenue Projections: Computes total potential revenue based on dynamic ticket prices and remaining inventory.
The scripts generates various plots for analysis and outputs data-driven pricing recommendations to optimize revenue for sporting events. The alternate case consistently results in higher simulated ticket revenue over a 30-day period.   
