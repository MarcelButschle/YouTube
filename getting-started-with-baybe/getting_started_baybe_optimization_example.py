import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.recommenders import BotorchRecommender, TwoPhaseMetaRecommender, RandomRecommender
from baybe.surrogates import GaussianProcessSurrogate
from baybe.acquisition import qUCB, qLogEI

# 1. Objective function


def yield_fn(xi1, xi2):
    xi1, xi2 = float(xi1), float(xi2)
    x1_t, x2_t = 3 * xi1 - 15, xi2 / 50.0 - 13
    c, s = np.cos(0.5), np.sin(0.5)
    x1_r = c * x1_t - s * x2_t
    x2_r = s * x1_t + c * x2_t
    y = np.exp(-x1_r**2 / 80.0 - 0.5 * (x2_r + 0.03 * x1_r**2 - 40 * 0.03)**2)
    return 100.0 * y


# 2. Define the search space
parameters = [
    NumericalContinuousParameter(name="xi1", bounds=(0, 10)),
    NumericalContinuousParameter(name="xi2", bounds=(0, 1000)),
]
searchspace = SearchSpace.from_product(parameters=parameters)

# 3. Configure the Surrogate and Acquisition Function
# We choose a Gaussian Process and LogEI for exploration-exploitation balance
surrogate_model = GaussianProcessSurrogate()
acquisition_function = qLogEI()

# 4. Define the Strategy/Recommender
# We use a two-phase strategy:
# Phase 1: 3 random points to seed the model (TwoPhaseMetaRecommender)
# Phase 2: Bayesian optimization using our custom settings
recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),
    recommender=BotorchRecommender(
        surrogate_model=surrogate_model,
        acquisition_function=acquisition_function
    ),
    switch_after=3  # Switch to BO after 3 measurements are added
)

# 5. Create Campaign
target = NumericalTarget(name="yield")
campaign = Campaign(
    searchspace=searchspace,
    objective=target,
    recommender=recommender
)

# 6. Optimization loop
print("Starting customized BayBE Optimization...")
for i in range(15):
    rec = campaign.recommend(batch_size=1)
    xi1, xi2 = rec["xi1"].iloc[0], rec["xi2"].iloc[0]
    y = yield_fn(xi1, xi2)

    rec["yield"] = y
    campaign.add_measurements(rec)
    print(f"Trial {i+1:02d}: xi1={xi1:6.2f}, xi2={xi2:7.2f} | Yield = {y:6.2f}%")

# 7. Display best results
best_idx = campaign.measurements["yield"].idxmax()
print(f"\nBest Yield: {campaign.measurements.loc[best_idx, 'yield']:.4f}%")

# 8. Plotting the optimization progress
yield_values = campaign.measurements["yield"].values
cum_max_values = np.maximum.accumulate(yield_values)

plt.figure(figsize=(10, 6))
plt.scatter(range(1, len(yield_values) + 1), yield_values,
            color='green', alpha=0.4, label='Individual Trials (BayBE)')
plt.plot(range(1, len(cum_max_values) + 1), cum_max_values,
         color='green', marker='s', linewidth=2, label='Cumulative Best Value')

plt.title("Optimization Progress with BayBE", fontsize=14)
plt.xlabel("Trial Number", fontsize=12)
plt.ylabel("Yield [%]", fontsize=12)
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
