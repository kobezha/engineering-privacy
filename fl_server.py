import flwr as fl
import sys
import numpy as np

save_weights = True 

#define a strategy for weight aggregation on the central server
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # fed averaging 
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        #for saving weights 
        if aggregated_weights is not None:
            if save_weights:
                print(f"Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        
        return aggregated_weights

# initializing a federated averaging strategy 
strategy = SaveModelStrategy()

# starting flower server
fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),  

    #set as 40 rounds for data collection
    config=fl.server.ServerConfig(num_rounds=40),    

    #grpc message length 
    grpc_max_message_length=1024 * 1024 * 1024,      
    strategy=strategy                                
)