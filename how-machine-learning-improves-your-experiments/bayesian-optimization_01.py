import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from baybe import Campaign
from baybe.searchspace import SearchSpace
import numpy as np
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
import pandas as pd

from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
    RandomRecommender,
)
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
    NumericalContinuousParameter,
)

# Define the optimization objective
target = NumericalTarget(
    name="Y",
    mode="MAX",
)
objective = SingleTargetObjective(target=target)

# Generate data
X = np.random.choice(np.linspace(20, 100, 10000), size=200,
                     replace=False).reshape(-1, 1)


def generate_data(X):
    y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)
    return y


def generate_data(X):
    y = -0.05 * X**2 + 8*X + np.random.normal(scale=0.5, size=X.shape)
    # Normalize y to max 0.96
    y_normalized = y / np.max(y) * 0.96
    return y_normalized


def true_function(X):
    y = -0.05 * X**2 + 8*X
    y_normalized = y / np.max(y) * 0.96
    return y_normalized


y = generate_data(X)
# Plot X and y

plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of X and y')
plt.show()


# Define the search space
parameters = [
    NumericalContinuousParameter(
        name="X",
        bounds=(20, 100),
    ),
]
searchspace = SearchSpace.from_product(parameters)

# Define the optimization strategy
recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),  # farthest point sampling
    recommender=BotorchRecommender(
        acquisition_function="qEI"),  # Bayesian model-based optimization
)

# The optimization loop
campaign = Campaign(searchspace, objective, recommender)

batch_size = 2

# Initial recommendations
df = campaign.recommend(batch_size=batch_size)

df["Y"] = generate_data(df["X"])
print(df)
campaign.add_measurements(df)

sampled_data = df
for i in range(3):
    df = campaign.recommend(batch_size=batch_size)

    df["Y"] = generate_data(df["X"])
    sampled_data = pd.concat([sampled_data, df])

    campaign.add_measurements(df)

    data = campaign.measurements[[p.name for p in campaign.parameters]]
    def model(x): return campaign.get_surrogate().posterior(x).mean

    predict = pd.DataFrame(X, columns=["X"])
    posterior = campaign.posterior(predict)

    mean_predictions = posterior.mean.detach().numpy()
    variances = posterior.variance.detach().numpy()

    # Calculate the confidence intervals
    ci_upper = mean_predictions + 1.96 * np.sqrt(variances)
    ci_lower = mean_predictions - 1.96 * np.sqrt(variances)

    # Combine X, Y, mean_predictions, ci_upper, and ci_lower into a data frame
    df_combined = pd.DataFrame({
        'X': X.flatten(),
        'Y': y.flatten(),
        'Mean Predictions': mean_predictions.flatten(),
        'CI Upper': ci_upper.flatten(),
        'CI Lower': ci_lower.flatten()
    })

    # Sort the data frame by X values
    df_combined = df_combined.sort_values(by='X')

    # Plot the data
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # ax.scatter(df_combined['X'], df_combined['Y'], label='Actual')
    ax.scatter(sampled_data['X'], sampled_data['Y'],
               label='Sampled', color='r')
    df_combined = df_combined.sort_values(by='X')
    ax.plot(df_combined['X'], true_function(
        df_combined['X']), label='True Function')
    ax.plot(df_combined['X'], df_combined['Mean Predictions'],
            label='Predicted Function')
    ax.fill_between(df_combined['X'], df_combined['CI Lower'],
                    df_combined['CI Upper'], color='r', alpha=0.2, label='95% CI')
    ax.legend()
    plt.xlabel('Temperature / °C')
    plt.ylabel('Yield')
    plt.ylim(0.3, 1.2)
    plt.title(f"Run {batch_size*(i+1)+batch_size}")
    plt.savefig(f"YouTube Figures/run_{batch_size*(i+1)+batch_size}.svg")
    plt.show()
