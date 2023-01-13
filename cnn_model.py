import torch


class CNN(torch.nn.Module):
    def __init__(self, n_in_channels: int, n_hidden_layers: int, n_kernels: int, kernel_size: int):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(CNN, self).__init__()

        layers = []
        for n in range(n_hidden_layers):
            layers.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                          bias=True, padding=int(kernel_size / 2)))
            layers.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        mask = x[:, 1].clone().to(dtype=torch.bool)

        return pred, mask


# Create an instance of our CNN
# cnn = CNN(n_in_channels=2, n_hidden_layers=3, n_kernels=32, kernel_size=7)
# device = torch.device("cpu")
# cnn.to(device=device)
# input_tensor = torch.arange(10 * 10 * 6, dtype=torch.float32,
#                             device=device).reshape((3, 2, 10, 10))
# print("\nCNN")
# print(f"input tensor shape: {input_tensor.shape}")
# output_tensor = cnn(input_tensor)
# for i in output_tensor:
#     print(i)
# print(f"output tensor shape: {output_tensor.shape}")
