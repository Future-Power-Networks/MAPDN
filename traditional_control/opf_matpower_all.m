%% run opf using matpower
% don't change here
clear;
close all;

%% change here
case_name = 'case33';
noise_switch = 1;
choice = 0;                %% 0 : summer | 1 : winter    
episode_no = 100;
load_scale = 1;            %% 1 : default (superhard)
pv_scale = 1;              %% 1 : defualt (superhard)          

test_length = 1*20*12-1;       
oversize_factor = 1.2;       %% the oversize factor of pv inverter
mpopt = mpoption('verbose', 0, 'out.all', 0);

%% load data
load_active = load(sprintf("load_active_%s.mat", case_name));
load_active = load_active.load_active_value;

load_reactive = load(sprintf("load_reactive_%s.mat", case_name));
load_reactive = load_reactive.load_reactive_value;

pv_active = load(sprintf("pv_active_%s.mat", case_name));
pv_active = pv_active.pv_active_value;

load_incidence_matrix = load(sprintf("load_incidence_matrix_%s", case_name));
load_incidence_matrix = load_incidence_matrix.load_incidence_matrix;

mpc = loadcase(case_name);                            %% load modified network

PV = pv_active*pv_scale; % reduce the PV a little bit
PV_std = std(PV);
PV_mean = mean(PV);

load_active = load_active*load_scale;
load_reactive = load_reactive*load_scale;
load_active_std = std(load_active);
load_reactive_std = std(load_reactive);

%% analysis data
PV_max = max(PV);
S_rated = PV_max*1.2; % PV capacity
pv_active_max = max(sum(pv_active,2));
load_active_max = max(sum(load_active,2));
disp(['The rated power is ', num2str(S_rated)])
disp(['The pv_load ratio is ', num2str(pv_active_max/load_active_max)]);

%% opf: we should set the constraint of Pmax=Pmin=P_pv, Qmax, and Qmin dependents
v_summary = [];
q_summary = [];
p_summary = [];                %% pv active power summary              
loss_summary = [];

start_index_set = randperm(size(load_active,1)- test_length - 1, episode_no);

for episode_index = 1:episode_no
    disp(episode_index);
    start_index = start_index_set(episode_index);
    for test_index = start_index:start_index+test_length
        if noise_switch == 1
            % add noise to the pv, load active and reactive
            noise_PV = mvnrnd(zeros(size(PV,2),1),diag(PV_std*0.01).^2)';
            noise_load_active = mvnrnd(zeros(size(load_active_std,2),1),diag(load_active_std*0.01).^2)';
            noise_load_reactive = mvnrnd(zeros(size(load_reactive_std,2),1),diag(load_reactive_std*0.01).^2)';
        
            PV_now = (PV(test_index,:)' + noise_PV);
            load_active_now = load_incidence_matrix*(load_active(test_index,:)' + noise_load_active);
            load_reactive_now = load_incidence_matrix*(load_reactive(test_index,:)' + noise_load_reactive);
        else
            % without noise
            PV_now = PV(test_index,:)';
            load_active_now = load_incidence_matrix*load_active(test_index,:)';
            load_reactive_now = load_incidence_matrix*load_reactive(test_index,:)';
        end
        
        Qmax = sqrt(S_rated'.^2-PV_now.^2);     %% the remaining capacity of pv inveter
        Qmin = -sqrt(S_rated'.^2-PV_now.^2);
        % load
        mpc.bus(:,3) = load_active_now;     %% add active load
        mpc.bus(:,4) = load_reactive_now;   %% add ractive load
       
        % optimal power flow
        % active consrataint
        mpc.gen(2:end,9) = PV_now*1.0001;   %% constant constraint
        mpc.gen(2:end,10) = PV_now*0.9999;
        % reactive constraint
        mpc.gen(2:end,4) = Qmax;
        mpc.gen(2:end,5) = Qmin;
        result_opf = runopf(mpc, mpopt);    %% run opf
        
        % summary
        % voltage
        v_summary = [v_summary,result_opf.bus(:,8)];                 %% record all bus voltage
        loss = sum(result_opf.gen(:,2)) - sum(result_opf.bus(:,3));  %% all Pg - all Pl = power loss
        loss_summary = [loss_summary,loss]; 
        q_summary = [q_summary,result_opf.gen(2:end,3)];             %% don't record the reference bus power
        p_summary = [p_summary,result_opf.gen(2:end,2)];
        
    end
end

%% save
fname_1 = sprintf('opf_result_all/v_summary_%s.mat', case_name);
fname_2 = sprintf('opf_result_all/p_summary_%s.mat', case_name);
fname_3 = sprintf('opf_result_all/q_summary_%s.mat', case_name);
fname_4 = sprintf('opf_result_all/loss_summary_%s.mat', case_name);

v_summary = v_summary';
p_summary = p_summary(:,:)';
q_summary_full = q_summary;
q_summary = q_summary(:,:)';
loss_summary = loss_summary';
save(fname_1, 'v_summary');
save(fname_2, 'p_summary');
save(fname_3, 'q_summary');
save(fname_4, 'loss_summary');

%% let's see..........
figure();
for i = 1:size(load_active,2)
    plot(v_summary(:,i));
    hold on
end
grid on
ylim([0.93,1.07])

