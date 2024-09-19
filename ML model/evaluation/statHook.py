class StatHook:
    """
    StatHook is a class designed to hook into a neural network layer during forward propagation
    and capture statistics such as the mean, standard deviation, and output distribution of the
    layer's activations. This can be useful for monitoring and diagnosing model behavior.
    """
    def __init__(self):
        """
        Initializes lists to store the means, standard deviations, and histogram data for each
        forward pass of the model.
        """
        self.means = []  # List to store mean values of the layer outputs
        self.stds = []   # List to store standard deviation values of the layer outputs
        self.hist_data = []  # List to store histogram data (layer outputs) for analysis

    def hook_fn(self, module, input, output):
        """
        Function that is called during the forward pass of a layer. This function calculates
        and stores the mean and standard deviation of the output data. Additionally, it stores
        the output data for potential histogram analysis.

        Args:
            module (nn.Module): The layer/module being hooked.
            input (Tensor): Input tensor to the layer.
            output (Tensor): Output tensor from the layer.
        """
        # Calculate and store the mean and standard deviation of the layer's output
        self.means.append(output.data.mean())  # Calculate and store mean
        self.stds.append(output.data.std())    # Calculate and store standard deviation

        # Optionally, store output data for histogram analysis (converted to NumPy)
        self.hist_data.append(output.data.cpu().numpy())

    def clear(self):
        """
        Clears all stored data, resetting the lists for means, standard deviations, and histogram data.
        """
        self.means.clear()
        self.stds.clear()
        self.hist_data.clear()

    def get_statistics(self):
        """
        Retrieves the stored statistics (means, standard deviations, and histogram data).

        Returns:
            tuple: A tuple containing three lists - means, standard deviations, and histogram data.
        """
        return self.means, self.stds, self.hist_data
