import polars as pl
import numpy as np

# Let's analyze the exact singleton math
total = 1732544
# In baseline (0.82), empty was 124,960 (7.21%)
# In second run (0.80), empty was 84,460 (4.87%)
# 40,500 entities lost their empty status!
diff = 124960 - 84460
print(f"Entities that lost empty status: {diff:,}")
print(f"Impact on Macro F0.5 if they were false positives on singletons: -{diff/total:.4f} (-{diff/total*100:.2f}%)")

# Now let's calculate what happens if we INCREASE precision:
# In Macro F0.5:
# If an entity has 1 true match:
#   - Pred 0 matches: Score = 0.0
#   - Pred 1 true match: Score = 1.0
#   - Pred 1 true + 1 false match: Score = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.555
#   - Pred 1 true + 2 false matches: Score = (1.25 * 0.33 * 1.0) / (0.25 * 0.33 + 1.0) = 0.385
# If an entity has 0 true matches (singleton):
#   - Pred 0 matches: Score = 1.0
#   - Pred 1 match (false positive): Score = 0.0 (A catastrophic 1.0 drop!)
