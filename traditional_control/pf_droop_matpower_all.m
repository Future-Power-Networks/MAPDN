%% run droop control and no control using matpower
% general test
clear;
close all;

% don't change here
run_specific = containers.Map;
run_specific('case33') = 0;
run_specific('case141') = 1;
run_specific('case322') = 2;

%% change the input here
case_name = 'case33';   
episode_no = 100;            %% average the result of each episode
test_length = 1*20*12-1;   %% 
noise_switch = 1;          %% 0 : without noise measurement          

max_ite = 100;             %% maximum droop control number
gain = 0.1;                %% control the convergent performance of droop controller
load_scale = 1;            %% 1 : default
pv_scale = 1;              %% 1 : defualt
oversize_factor = 1.2;       %% the oversize factor of pv inverter
mpopt = mpoption('verbose', 0, 'out.all', 0);

%% automatic selections for visualization
case_index = run_specific(case_name);            %% change at the same time
if case_index == 0
    reactive_ratio = 1.0;      %% the manually set ratio of the maximum reactive power
elseif case_index == 1
    reactive_ratio = 1.0;
elseif case_index == 2
    reactive_ratio = 1.0;
end

%% load data
load_active = load(sprintf("load_active_%s.mat", case_name));
load_active = load_active.load_active_value;

load_reactive = load(sprintf("load_reactive_%s.mat", case_name));
load_reactive = load_reactive.load_reactive_value;

pv_active = load(sprintf("pv_active_%s.mat", case_name));
pv_active = pv_active.pv_active_value;

load_incidence_matrix = load(sprintf("load_incidence_matrix_%s", case_name));
load_incidence_matrix = load_incidence_matrix.load_incidence_matrix;

PV = pv_active*pv_scale; 
PV_std = std(PV);
PV_mean = mean(PV);

load_active = load_active*load_scale;
load_reactive = load_reactive*load_scale;
load_active_std = std(load_active);
load_reactive_std = std(load_reactive);

%% load network
mpc = loadcase(case_name);                            %% load modified network
pv_bus_index = mpc.gen(2:end,1);                      %% the index of the pv bus

%% analysis data
PV_max = max(PV);
S_rated = PV_max*1.2;                                 %% the capacoty of PV inverter
pv_active_max = max(sum(pv_active,2));
load_active_max = max(sum(load_active,2));
disp(['The rated power is ', num2str(S_rated)])
disp(['The pv_load ratio is ', num2str(pv_active_max/load_active_max)]);
q_max_manual = reactive_ratio * S_rated;

%% summary variable
v_summary = [];
q_summary = [];
p_summary = [];                
loss_summary = [];

v_summary_pf = [];
q_summary_pf = [];
p_summary_pf = [];              
loss_summary_pf = [];

start_index_set = randperm(size(load_active,1)- test_length - 1, episode_no);

% main loop
for episode_index = 1:episode_no
    start_index = start_index_set(episode_index);
    for test_index = start_index:start_index + test_length
        if noise_switch == 1
            % add noise to the pv, load active and reactive
            noise_PV = mvnrnd(zeros(size(PV,2),1),diag(PV_std*0.01).^2)';
            noise_load_active = mvnrnd(zeros(size(load_active_std,2),1),diag(load_active_std*0.01).^2)';
            noise_load_reactive = mvnrnd(zeros(size(load_reactive_std,2),1),diag(load_reactive_std*0.01).^2)';
            PV_now = PV(test_index,:)' + noise_PV;
            load_active_now = load_incidence_matrix*(load_active(test_index,:)' + noise_load_active);
            load_reactive_now = load_incidence_matrix*(load_reactive(test_index,:)' + noise_load_reactive);
        else
            % without noise
            PV_now = PV(test_index,:)';
            load_active_now = load_incidence_matrix*load_active(test_index,:)';
            load_reactive_now = load_incidence_matrix*load_reactive(test_index,:)';
        end
        
        % load
        mpc.bus(:,3) = load_active_now;          %% add the current load active to the env
        mpc.bus(:,4) = load_reactive_now;        %% add the current load reactive to the env
        
        % pv
        mpc.gen(2:end,2) = PV_now;               %% gen_1 is the reference bus
        q_last = zeros(size(PV_now,1),1);        %% default no reactive power
        
        % power flow solution (without control)
        mpc.gen(2:end,3) = q_last;               %% update reactive power
        result = runpf(mpc, mpopt);              %% power flow solution
        
        % summary for pf solution
        v_summary_pf = [v_summary_pf, result.bus(:,8)];
        loss = sum(result.gen(:,2)) - sum(result.bus(:,3));
        loss_summary_pf = [loss_summary_pf,loss];
        q_summary_pf = [q_summary_pf, result.gen(2:end,3)];    %% remove the slack bus
        p_summary_pf = [p_summary_pf, result.gen(2:end,2)];
        
        % droop control loop
        q_new = zeros(size(PV_now,1),1);                       %% define vector size
        pv_bus_voltage_last = 100*ones(size(PV_now,1),1);      %% give a large votlage to pass the breack condition
        % voltage_conv = [];
    
        for i = 1:max_ite
            % disp(i)
            mpc.gen(2:end,3) = q_last;                         %% update reactive power : the reactive power of the first run is zero
            result = runpf(mpc, mpopt);                        %% run power flow for the updated reactive power
            voltage = result.bus(:,8);
        
            pv_bus_voltage = voltage(pv_bus_index);            %% find the voltage on the pv bus
    
            % when the overall voltage change is small, we move to the next
            % control instance
            if norm(pv_bus_voltage_last-pv_bus_voltage,2) < 1e-4
                disp(i)
                break
            end
    
            pv_bus_voltage_last = pv_bus_voltage;
    
            % voltage_conv = [voltage_conv;pv_bus_voltage(2)];
            % disp(pv_bus_voltage(2));
    
            for j = 1:size(PV_now,1)
                % for each of the pv bus
                q_new(j) = droop_control(PV_now(j), S_rated(j), pv_bus_voltage(j), q_max_manual(j));
            end
            q_last = (1-gain) * q_last + gain * q_new;  %% update the last q
            
        end
    
        % error();
        % summary
        v_summary = [v_summary, voltage];
        loss = sum(result.gen(:,2)) - sum(result.bus(:,3));
        loss_summary = [loss_summary,loss];
        q_summary = [q_summary,result.gen(2:end,3)];
        p_summary = [p_summary,result.gen(2:end,2)];
    end
end


%% save droop
fname_1 = sprintf('droop_result_all/v_summary_%s.mat', case_name);
fname_2 = sprintf('droop_result_all/p_summary_%s.mat', case_name);
fname_3 = sprintf('droop_result_all/q_summary_%s.mat', case_name);
fname_4 = sprintf('droop_result_all/loss_summary_%s.mat', case_name);

v_summary = v_summary';
p_summary = p_summary';
q_summary = q_summary';
loss_summary = loss_summary';
save(fname_1, 'v_summary');
save(fname_2, 'p_summary');
save(fname_3, 'q_summary');
save(fname_4, 'loss_summary');

%% save power flow (no control)
fname_1 = sprintf('pf_result_all/v_summary_%s.mat', case_name);
fname_2 = sprintf('pf_result_all/p_summary_%s.mat', case_name);
fname_3 = sprintf('pf_result_all/q_summary_%s.mat', case_name);
fname_4 = sprintf('pf_result_all/loss_summary_%s.mat', case_name);

v_summary_pf = v_summary_pf';
p_summary_pf = p_summary_pf';
q_summary_pf = q_summary_pf';
loss_summary_pf = loss_summary_pf';
save(fname_1, 'v_summary_pf');
save(fname_2, 'p_summary_pf');
save(fname_3, 'q_summary_pf');
save(fname_4, 'loss_summary_pf');

%% droop control function
function q =  droop_control(p_single, S_rated, v, q_max_manual)
    
    % p_single : the active power of a single bus
    % S_rated : the rated power of a single bus
    % v : voltage of a single bus
    
    v_ref = 1.0;  % reference voltage
    va = 0.95;     % saturation low
    vd = 1.05;     % saturation highs
    vb = 1.0;     % deadzone low
    vc = 1.0;     % deadzone high
    % gain = (max_ite/0.8-step_no)/(max_ite/0.8);
    gain = 1;

    q_max = sqrt(S_rated^2 - p_single^2); % the rated reactive power
    q_max = min(q_max, q_max_manual);     % set as the smaller one
       
    if v <= va
        % saturation : low voltage
        q = q_max;
    elseif v > vd
        % saturation : high voltage
        q = -q_max;
    elseif v >= vb && v <=vc
        % dead zone
        q = 0;
    elseif v < vb
        % Kp control : low
        droop_k = (q_max - 0) / (va - vb);
        q = gain * droop_k * (v-vb);
    elseif v > vc
        % Kp control : high
        droop_k = (0-q_max) / (vc - vd);
        q = gain * droop_k * (vc - v);
    end
end
