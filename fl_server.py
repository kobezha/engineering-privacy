import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Call the FedAvg aggregation from superclass to compute new global model weights
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        # Save the aggregated model weights after each round
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            # Save the model weights to a .npz file
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        
        return aggregated_weights

# Initialize the custom federated averaging strategy
strategy = SaveModelStrategy()

# Start the Flower server
fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),  # Server address
    config=fl.server.ServerConfig(num_rounds=40),     # Configuration with 3 federated learning rounds
    grpc_max_message_length=1024 * 1024 * 1024,      # GRPC message length
    strategy=strategy                                # Use the custom strategy for federated averaging
)