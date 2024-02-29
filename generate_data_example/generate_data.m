%% Define constants
define_constants;
grid_cases = ["case_ieee30"]; % ["case_ieee30", "case118", "case300"];
case_names = ["case30"]; % ["case30", "case118", "case300"]; % Rename case_ieee30
% Number of grids
N_tot = 15e3;
N_batch = 1e3;
num_batches = round(N_tot / N_batch);

% Set whether to disconnect lines
LINE_DISCONNECT = false;
% Set whether to ensure that the generator on the reference bus must
% output a positive active power
ENSURE_POWER_DEFICIT = true;

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
                ref_num = mpc_temp.bus((mpc_ref.bus(:,2) == 3.0), 1);

                % Disconnect lines/branches in grid
                if LINE_DISCONNECT
                    % Find index for the reference bus
                    n_ref = length(mpc_temp.bus(:,PD));
                    ref_idx = find(mpc_temp.bus(:,2) == 3.0);

                    % Number of branch disconnections
                    num_disconnects = 5 + randsample(15, 1, true);
                    % Indices for branches not connected to slack bus
                    idx_nonref_branches = find((mpc_temp.branch(:, 1) ~= ref_num) & (mpc_temp.branch(:, 2) ~= ref_num));
                    % Disconnect branches
                    mpc_temp.branch(randsample(idx_nonref_branches, num_disconnects, false), :) = [];

                    % Find groupings
                    groups = find_islands(mpc_temp);

                    % Discard groups that don't have the reference bus
                    for g=1:length(groups)
                        if any(groups{g} == ref_idx)
                            islands = extract_islands(mpc_temp, groups);
                            mpc_temp = islands{g};
                            break
                        end
                    end

                    % Discard sample if grid size is much smaller than original
                    if length(mpc_temp.bus(:,PD)) < (n_ref*0.1)
                        continue;
                    end
                end

                n_bus = length(mpc_temp.bus(:,PD));
                n_branch = length(mpc_temp.branch(:, 1));
                n_gen = length(mpc_temp.gen(:, 1));
                n_transformers = sum(mpc_temp.branch(:, TAP) ~= 0);
                % Find transformer indices
                transformer_idx = mpc_temp.branch(:, TAP) ~=0;

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
                
                % Discard sample if power deficit is not guaranteed
                if ENSURE_POWER_DEFICIT
                    ref_gen_idx = (mpc_temp.gen(:, 1) == ref_num);
                    if (sum(mpc_temp.gen(:, PG)) > (sum(mpc_temp.bus(:, PD)) + mpc_temp.gen(ref_gen_idx, PG)))
                        continue;
                    end
                end

                % Resample voltage setpoint for gens
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