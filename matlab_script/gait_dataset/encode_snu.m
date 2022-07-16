function trials = encode_snu(pilot_num, type, override)
% encode all trials of a pilot by steps starting with left heel strike
% result files are used for preparing dataset (prepare_dataset.m)
% type should be one of ['Gait', 'Lunge', 'Squat']

if nargin < 3
    override = false;
end

type = string(type);
assert(ismember(type, ["Gait", "Lunge", "Squat"]), ...
    sprintf("type shold be 'Gait', 'Lunge', or 'Squat': %s", type))

% find file name from pilot_num
if ispc
    root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', type);
elseif isunix
    root_dir = fullfile('/', 'home', 'user', 'Dropbox', 'SNU_DATASET', type);
end
subfolders = cellfun(@string, {dir(root_dir).name});
folder = subfolders(contains(subfolders, sprintf('Pilot%02d', pilot_num)));

% handling empty pilot folder
if isempty(folder)
    warning("no folder of pilot num #%d exists.", pilot_num)
    trials = [];
    return
end

% find all trials of same subject
ex_dir = fullfile(root_dir, folder, 'EDIT_WALK*.XLS');
filenames = dir(ex_dir);

% process contained trials
trial_idx = 1;
trials = {};

for trial = 1:length(filenames)
    filename = filenames(trial).name;
    if regexp(filename, '\(\w*\)')
        warning('File %s looks like an repeatition, ignored.', filename);
        continue
    else
        fprintf('Processing Pilot %d, %s ... \n', pilot_num, filename);
    end
    
    % convert .xls to .xlsx
    [~, name, ~] = fileparts(filename);
    read_file = fullfile(root_dir, folder, name + ".xlsx");
    saved_files = dir(fullfile(root_dir, folder, 'trials*.mat'));
    
    if exist(read_file, 'file')
        delete(read_file)
    end
    
    % check if file exists and override flag
    if isempty({saved_files.name}) || override
        % convert .xls to .xlsx
        hExcel = actxserver('Excel.Application');
        workbooks = hExcel.Workbooks;
        workbook = workbooks.Open(fullfile(root_dir, folder, filename));
        workbook.SaveAs(read_file, 51);
        workbook.Close;
        
        % read excel
        [raw, txt] = xlsread(read_file);
        measure = cellfun(@string, txt(24, :));
        measure = arrayfun(@(x)regexprep(x, ' *', '_'), measure);
        
        % measured data
        mat_data = raw(25:end, :);
        n_data = size(mat_data, 1);
        
        % sort event index
        trans_idx = raw(18:21, 2:end)';
        trans_idx = trans_idx(~isnan(trans_idx));
        trans_idx = reshape(trans_idx, [], 4);
        assert(size(trans_idx, 1) == 2, 'more or less expected number of steps!')
        
        HS = sort(trans_idx(1:4));
        TO = sort(trans_idx(5:8));
        events = sort([HS, TO]);
        n_events = length(events);
        
        % additional data
        ex_data = struct();
        ex_data.Phase = zeros(n_data, 4);
        ex_data.R_contact = zeros(n_data, 1);
        ex_data.L_contact = zeros(n_data, 1);
        
        ex_data.R_HS = raw(18, 2:end);
        ex_data.R_HS = ex_data.R_HS(~isnan(ex_data.R_HS));
        ex_data.L_HS = raw(19, 2:end);
        ex_data.L_HS = ex_data.L_HS(~isnan(ex_data.L_HS));
        ex_data.R_TO = raw(20, 2:end);
        ex_data.R_TO = ex_data.R_TO(~isnan(ex_data.R_TO));
        ex_data.L_TO = raw(21, 2:end);
        ex_data.L_TO = ex_data.L_TO(~isnan(ex_data.L_TO));
        
        % process each phase
        if any(HS(1) == trans_idx(:, 1))
            LHS_start = false;
        else
            LHS_start = true;
        end
        
        for i = 1:n_events+1
            % set phase start/ end index
            if i == 1
                start_idx = 1;
            else
                start_idx = events(i-1);
            end
            
            if i == n_events+1
                end_idx = n_data;
            else
                end_idx = events(i);
            end
            
            %  set phase and contact infos
            phase_mat = zeros(end_idx-start_idx+1, 4);
            phase_mat(:, 4) = 1;    % initial left swing
            phase = circshift(phase_mat, i - 1 + LHS_start * 2, 2);
            ex_data.Phase(start_idx:end_idx, :) = phase;
            ex_data.R_contact(start_idx:end_idx) = sum(phase(:, [1, 2, 3]), 2);
            ex_data.L_contact(start_idx:end_idx) = sum(phase(:, [1, 3, 4]), 2);
        end
        
        % set measured
        for i = 1:length(measure)
            field_name = measure(i);
            
            % change EMG name
            % Rectus -> Vastus_medialis
            % Tibialis_ant -> Vastus_lateralis
            field_name = strrep(field_name, 'RECTUS', 'VASTUS_MEDIALIS');
            field_name = strrep(field_name, 'TIBIALIS_ANT', 'VASTUS_LATERALIS');
            
            field_name = char(lower(field_name));
            field_name(1) = upper(field_name(1));
            
            ex_data.(field_name) = mat_data(:, i);
        end
        
        % write trial data file
        mat_filename = fullfile(root_dir, folder, sprintf('trial%02d', trial_idx));
        save(mat_filename, 'ex_data')
        trials{trial_idx} = ex_data;
        trial_idx = trial_idx + 1;
        
        % mirrored data
        ex_data.Phase = ex_data.Phase(:, [3, 4, 1, 2]);
        fields = fieldnames(ex_data);
        
        % swap two field
        while ~ isempty(fields)
            field_swap_a = fields{1};
            field_swap_a = char(field_swap_a);
            
            if ~ contains(field_swap_a, {'R_', 'L_'})
                fields(1) = [];
                continue
            end
            
            field_swap_b = field_swap_a;
            if field_swap_b(1) == 'R'
                field_swap_b(1) = 'L';
            else
                field_swap_b(1) = 'R';
            end
            
            temp = ex_data.(field_swap_b);
            ex_data.(field_swap_b) = ex_data.(field_swap_a);
            ex_data.(field_swap_a) = temp;
            
            fields(1) = [];
            fields(ismember(fields, field_swap_b)) = [];
        end
        
        % write mirrored trial data file
        mat_filename = fullfile(root_dir, folder, sprintf('trial%02d', trial_idx));
        save(mat_filename, 'ex_data')
        trials{trial_idx} = ex_data;
        trial_idx = trial_idx + 1;
    else
        
        % load data
        for i = 1:length(saved_files)
            saved_filename = saved_files(i).name;
            load(saved_filename, 'ex_data')
            trials{stride_idx} = ex_data;
            trial_idx = trial_idx + 1;
        end
    end
end


