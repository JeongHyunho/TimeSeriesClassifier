function trials = encode_lunge(pilot_num, override)
% encodes all xls files of lunge trials and save is as .mat
% when the right leg is forward, phase is 
% forward swing(1), lean(2), retraction(3), backward swing (4)
% when the left leg is forward, phase is
% forward swing(5), lean(6), retraction(7), backward swing (8)

if nargin < 2
    override = false;
end

% percentage of knee flexion rom to determine if down/up starts/ends
REST_THRESHOLD = 0.05;

% find file name from pilot_num
if ispc
    root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'Lunge');
elseif isunix
    root_dir = fullfile('/', 'home', 'user', 'Dropbox', 'SNU_DATASET', 'Lunge');
end
subfolders = cellfun(@string, {dir(root_dir).name});
folder = subfolders(contains(subfolders, sprintf('Pilot%02d', pilot_num)));
assert(~isempty(folder), sprintf("no foler of pilot num %d exists.", pilot_num));

% delete other motion files
all_files = {dir(fullfile(root_dir, folder)).name};
other_files = regexpi(all_files, '\w*squat\w*\.\w*', 'match');
for i = 1:length(other_files)
    other_file = other_files{i};
    if isempty(other_file)
        continue
    else
        filename = other_file{1};
        warning('File %s is not a squat motion data, so deleted.', filename)
        delete(fullfile(root_dir, folder, filename))
    end
end

% find all trials of same subject
exp_dir = fullfile(root_dir, folder, '*LUNGE*.XLS');
filenames = dir(exp_dir);

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
        delete(read_file)
        measure = cellfun(@string, txt(24, :));
        measure = arrayfun(@(x)regexprep(x, ' *', '_'), measure);
        
        % measured data
        mat_data = raw(25:end, :);
        n_data = size(mat_data, 1);
        
        % determine which leg is on forward
        if isempty(regexpi(name, 'RT', 'match'))
            left_forward = true;
        else
            left_forward = false;
        end
        
        % find heel strike, toe off indices
        % set angle to determine retraction
        if left_forward
            HS = raw(19, 3);
            TO = raw(21, 2);
            knee_ang = mat_data(:, strcmpi(measure, "L_KNEE_FLEX_ANG"));
        else
            HS = raw(18, 3);
            TO = raw(20, 2);
            knee_ang = mat_data(:, strcmpi(measure, "R_KNEE_FLEX_ANG"));
        end
        
        [~, ret_idx] = findpeaks(knee_ang, 'MinPeakHeight', 0.8 * max(knee_ang));
        assert(length(ret_idx) == 1, "Exception on detecting retraction index")
        
        % setup data struct and store phase
        ex_data = struct();
        ex_data.Phase = zeros(n_data, 8);
        ex_data.Phase(1:HS, 1 + left_forward * 4) = 1;          % forward swing
        ex_data.Phase(HS+1:ret_idx, 2 + left_forward * 4) = 1;  % lean
        ex_data.Phase(ret_idx+1:TO, 3 + left_forward * 4) = 1;  % retract
        ex_data.Phase(TO+1:end, 4 + left_forward * 4) = 1;      % backward swing
        
        % set measured
        for i = 1:length(measure)
            field_name = measure(i);
            field_name = char(lower(field_name));
            field_name(1) = upper(field_name(1));
            
            ex_data.(field_name) = mat_data(:, i);
        end
        
        % write trial data file
        mat_filename = fullfile(root_dir, folder, sprintf('trial%02d', trial_idx));
        save(mat_filename, 'ex_data')
        trials{trial_idx} = ex_data;
        trial_idx = trial_idx + 1;
        
        % mirrored data (same phase due to symmetric motion)
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
