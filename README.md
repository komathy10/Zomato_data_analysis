A simple end-to-end data analysis project using the Zomato Restaurants Dataset from Kaggle.
The project covers data cleaning, manipulation, feature engineering, and visualizations to understand restaurant ratings, pricing, and cuisine trends.

What’s Done-
1.Cleaned missing values (ratings, best seller items)
2.Standardized column names
3.Converted prices & ratings to numeric
3.Created new features - Feature engineering helps create more meaningful information (Created Avg_Rating,Total_Votes,Created High_Rated (binary),Extracted Primary_Cuisine)
4.Binning (Grouping continuous numbers into categories)
5.Encoding (Label Encoding categorical text values)
6.One-Hot Encoding (Dummy Variables)
7.Feature Scaling (Standardization & Min-Max Scaling)
We scaled numeric features like-
Prices
Votes
Ratings
8.Polynomial features help machine learning models learn non-linear patterns like curves, interactions between variables
For example, we create-
Squared terms:
Dining_Rating^2
Delivery_Votes^2
Interaction terms:
Dining_Rating * Delivery_Rating
Dining_Votes * Delivery_Votes
These help models capture complex relationships.
