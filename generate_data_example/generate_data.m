%% Define constants
define_constants;
grid_cases = ["case_ieee30"]; % ["case_ieee30", "case118", "case300"];
case_names = ["case30"]; % ["case30", "case118", "case300"]; % Rename case_ieee30
% Number of grids
N_tot = 15e3;
N_batch = 1e3;
num_batches = round(N_tot / N_batch);

%% Generate grids by resampling values in reference grid
mpopt = mpoption('verbose', 0, 'out.all', 0);
for i = 1:length(grid_cases)
    DATA_PATH = pwd + "\data\matpower_format\" + case_names(i) + "\";

    if not(isfolder(DATA_PATH))
        mkdir(DATA_PATH)
    end

    mpc_name = "grids_pre_soln";
    acpf_name = "grids_post_acpf";
    dcpf_name = "grids_post_dcpf";

    mpc_ref = loadcase(grid_cases(i));
    n_bus = length(mpc_ref.bus(:,PD));
    n_branch = length(mpc_ref.branch(:, 1));
    n_gen = length(mpc_ref.gen(:, 1));
    n_transformers = sum(mpc_ref.branch(:, TAP) ~= 0);
    % Find bus-number and index for the reference
    ref_num = mpc_ref.bus((mpc_ref.bus(:,2) == 3.0), 1);
    ref_gen_idx = (mpc_ref.gen(:, 1) == ref_num);
    % Find transformer indices
    transformer_idx = mpc_ref.branch(:, TAP) ~=0;
    
    for j = 1:num_batches
        % Initialize cell arrays to hold results
        mpc_array = cell(1, N_batch);
        res_acpf_array = cell(1, N_batch);
        res_dcpf_array = cell(1, N_batch);

        for k = 1:N_batch
           done = false;
           % Count convergence failure
           n_failures = 0;
            while done == false
               lastwarn('') % Clear last warning message
               mpc_temp = mpc_ref;
               
               % Resample load values
               mpc_temp.bus(:, PD) = mpc_temp.bus(:, PD).*(0.5 + 1.0 .* rand(n_bus,1));
               mpc_temp.bus(:, QD) = mpc_temp.bus(:, QD).*(0.5 + 1.0 .* rand(n_bus,1));
               
               % Resample shunt values
               mpc_temp.bus(:, GS) = mpc_temp.bus(:, GS).*(0.75 + 0.5 .* rand(n_bus,1));
               mpc_temp.bus(:, BS) = mpc_temp.bus(:, BS).*(0.75 + 0.5 .* rand(n_bus,1));
               
               % Resample resistance, reactance and charging susceptance
               mpc_temp.branch(:, BR_R) = mpc_temp.branch(:, BR_R).*(0.2 .* rand(n_branch,1)+0.9);
               mpc_temp.branch(:, BR_X) = mpc_temp.branch(:, BR_X).*(0.2 .* rand(n_branch,1)+0.9);
               mpc_temp.branch(:, BR_B) = mpc_temp.branch(:, BR_B).*(0.2 .* rand(n_branch,1)+0.9);
               % Sample transformer tap ratio and shift angle
               mpc_temp.branch(transformer_idx, TAP) = (0.4 .* rand(n_transformers,1)+0.8);
               mpc_temp.branch(transformer_idx, SHIFT) = (0.4 .*rand(n_transformers,1) - 0.2).*180/pi;
               
               % Resample generator active power setpoint
               mpc_temp.gen(:, PG) = min(mpc_temp.gen(:, PG).*(0.75 + 0.5 .* rand(n_gen,1)), mpc_temp.gen(:, PMAX));
               
               %-- Resample voltage setpoint for gens --%
               mpc_temp.gen(:, VG) = (0.15 .* rand(n_gen,1)+0.95);
               for g=1:n_gen
                    gen_idx = mpc_temp.gen(g,1);
                    idx = find(mpc_temp.bus(:,1) == gen_idx);
                    mpc_temp.bus(idx, VM) = mpc_temp.gen(g, VG);
               end
               
               % Apply standard power solver
               respf_temp = runpf(mpc_temp, mpopt);
               if respf_temp.success == 1
                   % Run DCPF solver
                   dcpf_temp = rundcpf(mpc_temp, mpopt);
                   % Catch warnings
                   [warnMsg, warnId] = lastwarn;
                   if isempty(warnMsg)
                        % Exit while loop if there are no warnings
                        done = true;
                   else
                       n_failures = n_failures + 1;
                   end

               else
                   n_failures = n_failures + 1;
               end
            end
            
           mpc_array{k} = mpc_temp;
           res_acpf_array{k} = respf_temp;
           res_dcpf_array{k} = dcpf_temp;
           
           disp(['Batch: ', num2str(j), ' Iteration: ', num2str(k), ' Failures: ', num2str(n_failures)])
        end
        save(DATA_PATH + mpc_name + "_batch" + num2str(j), 'mpc_array')
        save(DATA_PATH + acpf_name + "_batch" + num2str(j), 'res_acpf_array')
        save(DATA_PATH + dcpf_name + "_batch" + num2str(j), 'res_dcpf_array')
    end
end