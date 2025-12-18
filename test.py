import numpy as np
import matplotlib.pyplot as plt
import math

# making a change to test the commit now

# --- DATA ---
x_train = np.array([10, 15, 20, 25, 30])  # temperature
y_train = np.array([85, 70, 62, 48, 35])  # sold cups
number_of_examples = x_train.shape[0]

# show the digram of the axses
plt.scatter(x_train, y_train)
plt.title("The Shivering Barista")
plt.xlabel("temperature")
plt.ylabel("sold cups")
plt.show()


def linearRegression(weight, bias, x, examples_number):
    # Returns an array of all predictions (y_hat)
    y_hat = np.zeros(examples_number)
    for i in range(examples_number):
        y_hat[i] = weight * x[i] + bias
    return y_hat


def loss_function(examples_number, y_hat, y):
    loss_sum = 0
    for i in range(examples_number):
        error = y_hat[i] - y[i]
        loss_sum += error**2
    total_loss = (1 / (2 * examples_number)) * loss_sum
    return total_loss


def gradient(examples_number, x, y, weight, bias):
    d_dw = 0
    d_db = 0
    # Calculate predictions first to use in gradient
    y_hat = linearRegression(weight, bias, x, examples_number)

    for i in range(examples_number):
        # We use the prediction for the specific point [i]
        err_i = y_hat[i] - y[i]
        d_dw += err_i * x[i]
        d_db += err_i

    d_dw = d_dw / examples_number
    d_db = d_db / examples_number
    return d_dw, d_db


def gradient_descent(x, y, w_init, b_init, alpha, examples_number, iters):
    print("STARTING Gradient Descent...")
    j_history = []
    w = w_init
    b = b_init

    for i in range(iters):
        # We must calculate gradient INSIDE the loop every time w and b change
        d_dw, d_db = gradient(examples_number, x, y, w, b)

        # Update Parameters
        w = w - alpha * d_dw
        b = b - alpha * d_db

        # Save cost history
        current_y_hat = linearRegression(w, b, x, examples_number)
        cost = loss_function(examples_number, current_y_hat, y)
        j_history.append(cost)

        # Print progress 10 times during the run
        if i % (iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {cost:0.2e} | w: {w:0.3f}, b: {b:0.3f}")

    return w, b, j_history


def main_app():
    weight_init = 0
    bais_init = 0
    alpha = 0.001  # Increased slightly for better convergence
    iterations = 100000  # make itration 100k to reach the best possible point

    # Calculate final weights and bias
    weight_final, bais_final, j_history = gradient_descent(
        x_train, y_train, weight_init, bais_init, alpha, number_of_examples, iterations
    )

    print(f"\nFinal Results -> w: {weight_final:0.4f}, b: {bais_final:0.4f}")

    # --- GRAPHING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Cost History (The Suffering Plot)
    ax1.plot(j_history, color="purple")
    ax1.set_title("Cost Reduction Over Time")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")

    # Graph 2: The Result (The Barista's Model)
    ax2.scatter(x_train, y_train, marker="x", c="r", label="Actual Sales")
    y_final_line = weight_final * x_train + bais_final
    ax2.plot(x_train, y_final_line, label="Prediction Line", c="b")
    ax2.set_title("Temperature vs Cups Sold")
    ax2.set_xlabel("Temp (°C)")
    ax2.set_ylabel("Cups Sold")
    ax2.legend()

    plt.show()

    # The prediction formula: y = wx + b
    prediction_20 = weight_final * 20 + bais_final
    print(f"Prediction for 20°C: {prediction_20:0.2f} cups")

    # The prediction formula: y = wx + b

    prediction_20 = weight_final * 20 + bais_final
    print(f"Prediction for 20°C: {prediction_20:0.2f} cups")
    print(f"Prediction for 20°C: {weight_final} * 20 + {bais_final} = {prediction_20}")
