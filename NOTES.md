Possible problems right now
- Treat the skill difference between each escape count the same
    - If implement ordinal regression, will need iterative optimization again instead of the current closed form solution
- Can't identify players who have been playing in the same team together (collinearity)
    - E.g: Persica has low rating while Koting has abnormally high rating
    - Tried difference penalty & team effects
    

Things I did
- Sum to zero constraint for a baseline reference
Run virtual environment command:
source venv/bin/activate