CUSTOMER CHURN INTELLIGENCE PLATFORM

Business Objective
------------------
Identify customers at risk of becoming inactive so that
retention actions can be prioritized.

Prediction Unit
---------------
Customer snapshot at a defined prediction date.

Observation Window
------------------
180 days.

Prediction Window
-----------------
90 days.

Target
------
1 → No qualifying purchase during the following 90 days.
0 → At least one qualifying purchase during the following 90 days.

Churn Type
----------
Behavioral / inactivity-based churn.

Temporal Constraint
-------------------
A snapshot is eligible only when the complete 90-day
future observation period exists.

1. Load both workbook sheets.

2. Validate that both sheets contain the expected schema.

3. Standardize column names.

4. Normalize datatypes.

5. Combine the two sheets.

6. Remove exact duplicate records.

7. Identify cancellation/return/operational records.

8. Separate records with missing Customer IDs.

9. Construct a completed-purchase dataset using
   customer-identifiable, positive-quantity, positive-price
   purchase records.

10. Preserve the original raw data and all cleaning statistics.

11. Validate the resulting dataset.