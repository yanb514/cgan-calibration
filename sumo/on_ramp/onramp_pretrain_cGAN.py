"""
High-Level Description:
This script is designed to generate training data for a Conditional Generative Adversarial Network (cGAN) to calibrate traffic simulation models. It uses the SUMO (Simulation of Urban Mobility) traffic simulator to produce two types of data:
1. Real data: Generated with fixed parameters and mildly varying demands.
2. Simulated (fake) data: Generated with varying parameters and mildly varying demands.

The script saves the generated data in the following directories:
- /data/SCENARIO/real: Contains real traffic data.
- /data/SCENARIO/sim: Contains simulated traffic data.

The script also includes functionality for training a discriminator model to distinguish between real and simulated traffic patterns.

"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import xml.etree.ElementTree as ET
import shutil
import logging
from datetime import datetime
import sys
import glob
import json
import random
import multiprocessing
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
from onramp_calibrate import clear_directory
# optuna.logging.set_verbosity(optuna.logging.ERROR)


# ================ Configuration ====================
SCENARIO = "onramp"
FLOW_STD = 5  # standard deviation for simulated data
FLOW_STD_REAL = 5  # slightly larger standard deviation for real data to create more variation
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)
measurement_locations = ['upstream_0', 'upstream_1', 'merge_0', 'merge_1', 'merge_2', 'downstream_0', 'downstream_1']

# Define parameter ranges for calibration (simulated data)

param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 'lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain', 'lcKeepRight']
min_val = [30.0, 1.0, 1.0, 1.0, 0.5, 0, 0, 0.0001, 0, 0]
max_val = [35.0, 3.0, 4.0, 3.0, 2.0, 5, 1, 5,      5, 5]

# Define parameter ranges for real data generation with more overlapping
real_min_val = [29.5, 0.8, 0.8, 0.8, 0.4, 0, 0, 0.0001, 0, 0]  # slightly wider range
real_max_val = [35.5, 3.2, 4.2, 3.2, 2.2, 5.5, 1.2, 5.5, 5.5, 5.5]

# ================ Discriminator Model ====================

def simulate_real_worker(i, measurement_locations):
    """
    Runs SUMO for a single sample and extracts traffic patterns.
    """
    # Generate random driver parameters
    # driver_param = {
    #     param_name: random.uniform(real_min_val[i], real_max_val[i])
    #     for i, param_name in enumerate(param_names)
    # }
    temp_config_path, temp_path = create_temp_config(param=None, trial_number=i, type="real")
    run_sumo(temp_config_path)

    # Extract simulated traffic volumes
    real_meas = reader.extract_sim_meas(
        ["trial_" + location for location in measurement_locations],
        file_dir=temp_path
    )

    # 'real_meas' is a dictionary with keys "flow", "density", "speed", each holding a 2D array (time x space)
    flow = real_meas["flow"]
    density = real_meas["density"]
    speed = real_meas["speed"]

    # Stack the features (flow, density, speed) along a new dimension to form (time, space, features)
    traffic = np.stack([flow, density, speed], axis=-1)  # Shape: (time, space, features)
    demand = extract_od_matrix(temp_path)

    return traffic, demand


def generate_real_data(num_samples, suffix):
    """
    Generate real traffic data samples using SUMO in parallel.

    Args:
        num_samples (int): The number of real data samples to generate.
        suffix (str): The suffix to add to the saved file names.

    Description:
        This function uses the SUMO traffic simulator to generate real traffic data. It runs SUMO simulations in parallel using multiprocessing. The SUMO configuration for real data generation involves fixed vehicle parameters and mildly varying demands. The traffic patterns and demand data are extracted and saved as tensors.

        SUMO Configuration:
        - The route file (.rou.xml) is based on a ground truth file (`SCENARIO_gt.rou.xml`).
        - Noise is added to the vehicle flow rates using a Gaussian distribution with a standard deviation defined by `FLOW_STD`.

    Returns:
        None
    """
    """
    Generate real traffic data samples using SUMO in parallel.

    Args:
        num_samples (int): The number of real data samples to generate.
        suffix (str): The suffix to add to the saved file names.

    Returns:
        None
    """

    # Use multiprocessing to parallelize the SUMO runs
    with multiprocessing.Pool(processes=min(4, num_samples)) as pool:
        results = pool.map(partial(simulate_real_worker, measurement_locations=measurement_locations), range(num_samples))

    # Separate traffic and demand for saving
    traffic_patterns = [result[0] for result in results]  # Extract traffic data
    demand_data = [result[1] for result in results]      # Extract demand data

    # Convert the list of traffic patterns to a tensor
    traffic_patterns_tensor = torch.tensor(np.array(traffic_patterns), dtype=torch.float32)

    # Convert the list of demand data to a tensor
    demand_tensor = torch.tensor(np.array(demand_data), dtype=torch.float32)

    # Clear temporary directory
    clear_directory("temp")

    # Save both traffic_patterns_tensor and demand_tensor to "/data/SCENARIO/sim"
    save_traffic_path = os.path.abspath(f"../../data/{SCENARIO}/real/traffic_patterns_{suffix}.pt")
    save_demand_path = os.path.abspath(f"../../data/{SCENARIO}/real/demand_{suffix}.pt")
    
    os.makedirs(os.path.dirname(save_traffic_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_demand_path), exist_ok=True)
    
    torch.save(traffic_patterns_tensor, save_traffic_path)
    torch.save(demand_tensor, save_demand_path)
    
    print(f"Saved traffic patterns tensor to {save_traffic_path}")
    print(f"Saved demand tensor to {save_demand_path}")

    return 


def simulate_sim_worker(i, measurement_locations):
    """
    Runs SUMO for a single sample and extracts traffic patterns and demand.
    """
    # Generate random driver parameters
    # driver_param = {
    #     param_name: random.uniform(min_val[i], max_val[i])
    #     for i, param_name in enumerate(param_names)
    # }

    # Create temp configuration for the simulation
    temp_config_path, temp_path = create_temp_config(param=None, trial_number=i, type="sim")
    
    # Run SUMO simulation
    run_sumo(temp_config_path)

    # Extract simulated traffic volumes
    real_meas = reader.extract_sim_meas(
        ["trial_" + location for location in measurement_locations],
        file_dir=temp_path
    )

    # 'real_meas' is a dictionary with keys "flow", "density", "speed", each holding a 2D array (time x space)
    flow = real_meas["flow"]
    density = real_meas["density"]
    speed = real_meas["speed"]

    # Stack the features (flow, density, speed) along a new dimension to form (time, space, features)
    traffic = np.stack([flow, density, speed], axis=-1)  # Shape: (time, space, features)
    
    # Extract OD matrix (demand)
    demand = extract_od_matrix(temp_path)

    return traffic, demand

def generate_sim_data(num_samples, suffix):
    """
    Generate simulated (fake) traffic data samples using SUMO in parallel.

    Args:
        num_samples (int): The number of simulated data samples to generate.
        suffix (str): The suffix to add to the saved file names.

    Description:
        This function uses the SUMO traffic simulator to generate simulated traffic data. It runs SUMO simulations in parallel using multiprocessing. The SUMO configuration for simulated data generation involves varying vehicle parameters and mildly varying demands. The traffic patterns and demand data are extracted and saved as tensors.

        SUMO Configuration:
        - The route file (.rou.xml) is modified to include randomized vehicle parameters (e.g., maxSpeed, minGap, accel, etc.).
        - Noise is added to the vehicle flow rates using a Gaussian distribution with a standard deviation defined by `FLOW_STD`.

    Returns:
        traffic_patterns_tensor (Tensor): Tensor containing traffic patterns.
        demand_tensor (Tensor): Tensor containing demand data.
    """
    """
    Generate real traffic data samples using SUMO in parallel.
    """

    # Use multiprocessing to parallelize the SUMO runs
    with multiprocessing.Pool(processes=min(num_samples, multiprocessing.cpu_count())) as pool:
        # For each sample, simulate traffic and demand
        results = pool.map(partial(simulate_sim_worker, measurement_locations=measurement_locations), range(num_samples))

    # Separate traffic and demand for saving
    traffic_patterns = [result[0] for result in results]  # Extract traffic data
    demand_data = [result[1] for result in results]      # Extract demand data

    # Convert the list of traffic patterns to a tensor
    traffic_patterns_tensor = torch.tensor(np.array(traffic_patterns), dtype=torch.float32)

    # Convert the list of demand data to a tensor
    demand_tensor = torch.tensor(np.array(demand_data), dtype=torch.float32)

    # Clear temporary directory
    clear_directory("temp")

    # Save both traffic_patterns_tensor and demand_tensor to "/data/SCENARIO/sim"
    save_traffic_path = os.path.abspath(f"../../data/{SCENARIO}/sim/traffic_patterns_{suffix}.pt")
    save_demand_path = os.path.abspath(f"../../data/{SCENARIO}/sim/demand_{suffix}.pt")
    
    print(f"Saving to: {save_traffic_path} and {save_demand_path}")
    os.makedirs(os.path.dirname(save_traffic_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_demand_path), exist_ok=True)
    
    torch.save(traffic_patterns_tensor, save_traffic_path)
    torch.save(demand_tensor, save_demand_path)
    
    print(f"Saved traffic patterns tensor to {save_traffic_path}")
    print(f"Saved demand tensor to {save_demand_path}")

    return traffic_patterns_tensor, demand_tensor

# ================ SUMO Simulation Functions ====================
def run_sumo(sim_config):
    """Run a SUMO simulation with the given configuration."""
    command = ['sumo', '-c', sim_config, '--no-step-log', '--xml-validation', 'never']
    subprocess.run(command, check=True)

def create_temp_config(param, trial_number, type="real"):
    """Create temporary SUMO configuration files for each trial."""
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # Update .rou.xml file with new parameters
    if type == "real":
        # For real data, use the base route file but only modify flow rates
        rou_tree = ET.parse(f"{SCENARIO}.rou.xml")
        rou_root = rou_tree.getroot()
        # Keep original vehicle type parameters for real data
        for flow in rou_root.findall('flow'):
            vph = float(flow.get("vehsPerHour"))
            flow.set("vehsPerHour", str(vph + random.gauss(0, FLOW_STD_REAL)))
            flow.set("type", "realDriver")
            
        # for vtype in rou_root.findall('vType'):
        #     if vtype.get('id') == 'trial':
        #         for key, val in param.items():
        #             vtype.set(key, str(val))
        #         break
                
    else: # change parameters and add noise on flow
        rou_tree = ET.parse(f"{SCENARIO}.rou.xml")
        rou_root = rou_tree.getroot()
        # for vtype in rou_root.findall('vType'):
        #     if vtype.get('id') == 'trial':
        #         for key, val in param.items():
        #             vtype.set(key, str(val))
        #         break
        for flow in rou_root.findall('flow'):
            vph = float(flow.get("vehsPerHour"))
            flow.set("vehsPerHour", str(vph + random.gauss(0, FLOW_STD)))
            flow.set("type", "fakeDriver")
            
    new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
    rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)
    
    # Copy other necessary files
    shutil.copy(f"{SCENARIO}.net.xml", os.path.join(output_dir, f"{trial_number}_{SCENARIO}.net.xml"))
    shutil.copy('detectors.add.xml', os.path.join(output_dir, f"{trial_number}_detectors.add.xml"))
    
    # Update .sumocfg file
    sumocfg_tree = ET.parse(f"{SCENARIO}.sumocfg")
    sumocfg_root = sumocfg_tree.getroot()
    input_element = sumocfg_root.find('input')
    if input_element is not None:
        input_element.find('route-files').set('value', f"{trial_number}_{SCENARIO}.rou.xml")
        input_element.find('net-file').set('value', f"{trial_number}_{SCENARIO}.net.xml")
        input_element.find('additional-files').set('value', f"{trial_number}_detectors.add.xml")
    new_sumocfg_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.sumocfg")
    sumocfg_tree.write(new_sumocfg_file_path, encoding='UTF-8', xml_declaration=True)
    
    return new_sumocfg_file_path, output_dir

def extract_od_matrix(temp_path):
    """
    Extract time-varying OD matrix from the .rou file.
    
    Returns:
        od_matrix (np.ndarray): A 2D array of shape (routes × time).
    """
    if temp_path.endswith(".rou.xml"):
        rou_file_path = temp_path
    else:
        rou_file_path = glob.glob(os.path.join(temp_path, "*.rou.xml"))[0]
    tree = ET.parse(rou_file_path)
    root = tree.getroot()
    
    # Initialize OD matrix (example: 3 routes, 5 time steps)
    num_routes = config[SCENARIO]["N_ROUTES"]
    num_intervals = config[SCENARIO]["N_INTERVALS"]
    demand_matrix = np.zeros((num_routes, num_intervals))
    
    # Extract unique routes from route definitions
    routes = [route.get("id") for route in root.findall("route")]
    route_index = {route: i for i, route in enumerate(routes)}

    # Extract flow data
    for flow in root.findall("flow"):
        route = flow.get("route")
        vehs_per_hour = float(flow.get("vehsPerHour"))
        begin = int(flow.get("begin"))
        interval_idx = begin // 1800  # Assuming time intervals of 1800s

        if route in route_index:
            demand_matrix[route_index[route], interval_idx] = vehs_per_hour
    
    return demand_matrix

def normalize_traffic_patterns(traffic_patterns):
    """Normalize traffic patterns to the range [0, 1]."""
    max_val = torch.max(traffic_patterns)
    min_val = torch.min(traffic_patterns)
    return (traffic_patterns - min_val) / (max_val - min_val)


class Discriminator(nn.Module):
    def __init__(self, traffic_shape, od_shape):
        super(Discriminator, self).__init__()

        self.num_samples, self.space, self.time, self.features = traffic_shape
        self.num_samples_od, self.routes, self.time_intervals = od_shape

        # ==================== Traffic Pattern Branch ====================
        self.traffic_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.features, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=(1, 2, 1)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # Calculate traffic_output_size correctly
        with torch.no_grad():
            dummy_input = torch.randn(1, self.features, self.space, self.time, 1)
            dummy_output = self.traffic_conv(dummy_input)
            self.traffic_output_size = dummy_output.shape[1]

        # ==================== OD Matrix Branch ====================
        self.od_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(in_channels=self.routes, out_channels=32, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # Calculate od_output_size correctly
        with torch.no_grad():
            dummy_od = torch.randn(1, self.routes, self.time_intervals)
            dummy_od_output = self.od_conv(dummy_od)
            self.od_output_size = dummy_od_output.shape[1]

        # ==================== Fully Connected Layers ====================
        self.fc = nn.Sequential(
            nn.Linear(self.traffic_output_size + self.od_output_size, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, traffic_pattern, od_matrix):
        traffic_pattern = traffic_pattern.permute(0, 3, 1, 2).unsqueeze(-1)
        traffic_features = self.traffic_conv(traffic_pattern)
        od_features = self.od_conv(od_matrix)
        combined = torch.cat([traffic_features, od_features], dim=1)
        output = self.fc(combined)
        return output


# ================ Discriminator Training Function ====================
def train_discriminator(discriminator, real_traffic_patterns, real_od_matrices, simulated_traffic_patterns, simulated_od_matrices, 
                        num_epochs=100, batch_size=16, log_file="training_log.txt"):
    """
    Train the discriminator on real and simulated traffic patterns.
    
    Args:
    - discriminator (nn.Module): The discriminator model.
    - real_traffic_patterns (Tensor): Real traffic patterns, shape (sample_size, time, space, features).
    - real_od_matrices (Tensor): Real OD matrices, shape (sample_size, routes, time_intervals).
    - simulated_traffic_patterns (Tensor): Simulated traffic patterns, shape (sample_size, time, space, features).
    - simulated_od_matrices (Tensor): Simulated OD matrices, shape (sample_size, routes, time_intervals).
    - num_epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    - log_file (str): Path to the file where the training log will be saved.
    """
    
    # Combine real and simulated data
    X_traffic = torch.cat([real_traffic_patterns, simulated_traffic_patterns], dim=0)
    X_od = torch.cat([real_od_matrices, simulated_od_matrices], dim=0)
    y = torch.cat([torch.ones(len(real_traffic_patterns)), torch.zeros(len(simulated_traffic_patterns))], dim=0)
    y = y.unsqueeze(1)  # Shape: (16,) -> (16, 1)
    
    # Shuffle the data
    indices = torch.randperm(len(X_traffic))
    X_traffic = X_traffic[indices]
    X_od = X_od[indices]
    y = y[indices]
    
    # Create a DataLoader for batching
    dataset = torch.utils.data.TensorDataset(X_traffic, X_od, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function and optimizer
    lr = 0.0002
    betas = (0.5, 0.999)
    # criterion = nn.BCELoss() # use with sigmoid in fully connected layer (fc)
    criterion = nn.BCEWithLogitsLoss() # for stability
    optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Open the log file to save training progress
    with open(log_file, 'w') as log:
        log.write("Epoch, Step, Loss\n")  # Write headers to log file

        # Training loop
        for epoch in range(num_epochs):
            discriminator.train()  # Set the model to training mode
            running_loss = 0.0
            
            for i, (traffic_batch, od_batch, labels_batch) in enumerate(dataloader):
                # Move data to the GPU if needed
                traffic_batch = traffic_batch.to(device)
                od_batch = od_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = discriminator(traffic_batch, od_batch)
                
                # Calculate the loss
                loss = criterion(outputs, labels_batch)
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0) # gradient clipping
                optimizer.step()
                
                running_loss += loss.item()
            
            # Print the average loss for the epoch
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(dataloader)}')
                log.write(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(dataloader)}\n')
            
            # Optionally save the model after each epoch
            # torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

    print("Training complete!")



def evaluate_discriminator(discriminator, real_traffic_patterns, real_od_matrices, simulated_traffic_patterns, simulated_od_matrices, 
                           batch_size=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the discriminator on testing data.

    Args:
        discriminator (nn.Module): Pretrained discriminator model.
        test_loader (DataLoader): DataLoader for testing data.
        device (str): Device to use for evaluation ("cuda" or "cpu").

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Combine real and simulated data
    X_traffic = torch.cat([real_traffic_patterns, simulated_traffic_patterns], dim=0)
    X_od = torch.cat([real_od_matrices, simulated_od_matrices], dim=0)
    y = torch.cat([torch.ones(len(real_traffic_patterns)), torch.zeros(len(simulated_traffic_patterns))], dim=0)
    y = y.unsqueeze(1)  # Shape: (16,) -> (16, 1)
    
    # Shuffle the data
    indices = torch.randperm(len(X_traffic))
    X_traffic = X_traffic[indices]
    X_od = X_od[indices]
    y = y[indices]
    
    # Create a DataLoader for batching
    dataset = torch.utils.data.TensorDataset(X_traffic, X_od, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    discriminator.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for traffic_patterns, od_matrices, labels in dataloader:
            # Move data to the appropriate device
            traffic_patterns = traffic_patterns.to(device)
            od_matrices = od_matrices.to(device)
            labels = labels.to(device)

            # Get discriminator predictions
            outputs = discriminator(traffic_patterns, od_matrices).squeeze()  # Shape: (batch_size,)
            preds = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    # print(all_preds)
    # print(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    # Return results as a dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    return results

# ================ Main Script ====================
if __name__ == "__main__":
    # ================== Initialize logging ===================
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_dir = '_log'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_file = os.path.join(log_dir, f'{current_time}_optuna_log_{EXP}.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')


    # ================== Generate training and testing data ==================
    clear_directory("temp")

    # Configure parallel processing
    num_train_samples = 2
    num_test_samples = 2
    max_workers = min(multiprocessing.cpu_count(), 8)  # Use at most 8 cores to avoid overloading

    generate_real_data(num_samples=num_train_samples, suffix="train")
    # generate_sim_data(num_samples=num_train_samples, suffix="train")
    # generate_real_data(num_samples=num_test_samples, suffix="test")
    # generate_sim_data(num_samples=num_test_samples, suffix="test")

    # ================== Load data for discriminator training ==================
    # real_traffic_train = torch.load(f"../../data/{SCENARIO}/real/traffic_patterns_train.pt",weights_only=True)
    # real_od_train = torch.load(f"../../data/{SCENARIO}/real/demand_train.pt",weights_only=True)
    # sim_traffic_train = torch.load(f"../../data/{SCENARIO}/sim/traffic_patterns_train.pt",weights_only=True)
    # sim_od_train = torch.load(f"../../data/{SCENARIO}/sim/demand_train.pt",weights_only=True)
    # # Load data for discriminator testing
    # real_traffic_test = torch.load(f"../../data/{SCENARIO}/real/traffic_patterns_test.pt",weights_only=True)
    # real_od_test = torch.load(f"../../data/{SCENARIO}/real/demand_test.pt",weights_only=True)
    # sim_traffic_test = torch.load(f"../../data/{SCENARIO}/sim/traffic_patterns_test.pt",weights_only=True)
    # sim_od_test = torch.load(f"../../data/{SCENARIO}/sim/demand_test.pt",weights_only=True)
    # # Normalize traffic patterns
    # real_traffic_train = normalize_traffic_patterns(real_traffic_train)
    # sim_traffic_train = normalize_traffic_patterns(sim_traffic_train)
    # real_traffic_test = normalize_traffic_patterns(real_traffic_test)
    # sim_traffic_test = normalize_traffic_patterns(sim_traffic_test)

    # # ================== Train the discriminator ==================
    # discriminator = Discriminator(real_traffic_train.shape, real_od_train.shape)
    # train_discriminator(discriminator, real_traffic_train, real_od_train, sim_traffic_train, sim_od_train)
    # torch.save(discriminator.state_dict(), "discriminator.pth")

    # # # ================== Load the pretrained model ==================
    # # discriminator = Discriminator(real_traffic_test.shape, real_od_test.shape)
    # # discriminator.load_state_dict(torch.load("discriminator.pth", weights_only=True))

    # # # ================== Start evaluation (Testing) ==================
    # result = evaluate_discriminator(discriminator, real_traffic_test, real_od_test, sim_traffic_test, sim_od_test)
    # print(result)

    # # ================== plot training samples ==================
    # import matplotlib.pyplot as plt
    # # Extract relevant dimensions
    # sample_size, space, time, features = real_traffic_test.shape
    # assert sim_traffic_test.shape == real_traffic_test.shape, "Real and Sim data must have the same shape"
    # print("sample size ", sample_size, space, time, features)

    # # Select one feature to visualize
    # feature_idx = 1
    # real_traffic_feature = real_traffic_train[:, :, :, feature_idx]  # Shape: (sample_size, space, time, features)
    # sim_traffic_feature = sim_traffic_train[:, :, :, feature_idx]  # Shape: (sample_size, space, time, features)

    # # Create subplots for each space dimension
    # fig, axes = plt.subplots(nrows=space, figsize=(10, 2 * space), sharex=True)

    # if space == 1:
    #     axes = [axes]  # Ensure axes is iterable when space = 1

    # num_plot_samples = 20  # Increase the number of samples for plotting
    # for i in range(space):
    #     ax = axes[i]
        
    #     # Plot all real traffic samples in red
    #     for j in range(min(num_plot_samples, sample_size)):  # Limit to available samples
    #         ax.plot(real_traffic_feature[j, i, :], color="red", alpha=0.5, label="Real" if j == 0 else "")
        
    #     # Plot all simulated traffic samples in blue
    #     for j in range(min(num_plot_samples, sample_size)):  # Limit to available samples
    #         ax.plot(sim_traffic_feature[j, i, :], color="blue", alpha=0.5, label="Sim" if j == 0 else "")
        
    #     ax.set_title(measurement_locations[i])
    #     ax.legend()

    # plt.xlabel("Time")
    # plt.suptitle("Traffic Patterns: Real vs. Simulated")
    # plt.tight_layout()
    # plt.savefig("traffic_comparison.png", dpi=300, bbox_inches="tight")
    # # plt.show()
