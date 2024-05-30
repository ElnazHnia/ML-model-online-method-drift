class StatHook:
    def __init__(self):
        self.means = []
        self.stds = []
        self.hist_data = []

    def hook_fn(self, module, input, output):
        print("Output of fc2:", output.detach())
        # Calculate mean and std dev
        self.means.append(output.data.mean())
        self.stds.append(output.data.std())

        # Optionally, collect data for histograms (distribution)
        self.hist_data.append(output.data.cpu().numpy())

    def clear(self):
        # Clear the stored data
        self.means.clear()
        self.stds.clear()
        self.hist_data.clear()

    def get_statistics(self):
        return self.means, self.stds, self.hist_data

