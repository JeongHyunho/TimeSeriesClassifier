function trials = encode_squat(pilot_num, override)
% encodes all xls files of squat trials and save is as .mat
% phase is rest(0), down(1), up(2) and determined by knee flexion angles 

if nargin < 2
    override = false;
end

% percentage of knee flexion rom to determine if down/up starts/ends
REST_THRESHOLD = 0.05;

% find file name from pilot_num
if ispc
    root_dir = fullfile('C:', 'Users', 'biomechanics', 'Dropbox', 'SNU_DATASET', 'Squat');
elseif isunix
    root_dir = fullfile('/', 'home', 'user', 'Dropbox', 'SNU_DATASET', 'Squat');
end
subfolders = cellfun(@string, {dir(root_dir).name});
folder = subfolders(contains(subfolders, sprintf('Pilot%02d', pilot_num)));
assert(~isempty(folder), sprintf("no foler of pilot num %d exists.", pilot_num));

% delete other motion files
all_files = {dir(fullfile(root_dir, folder)).name};
other_files = regexpi(all_files, '\w*lunge\w*\.\w*', 'match');
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
exp_dir = fullfile(root_dir, folder, '*SQUAT*.XLS');
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
        
        % find event indices (rest->down->up->rest)
        knee_angs = mat_data(:, contains(measure, 'knee_flex_ang', 'IgnoreCase', true));
        m_knee_ang = mean(knee_angs, 2);
        rest_ang = min(m_knee_ang) + REST_THRESHOLD * (max(m_knee_ang) - min(m_knee_ang));
        knee_ang_over = m_knee_ang - rest_ang;
        rest_idx = find((knee_ang_over(2:end) >= 0) == (knee_ang_over(1:end-1) < 0), 3);
        assert(length(rest_idx) == 2, 'More or less than 2 rest indices. \n')
        
        [~, up_idx] = findpeaks(m_knee_ang, 'MinPeakHeight', max(m_knee_ang) - rest_ang);
        assert(length(up_idx) == 1, 'More than 1 up_idx. \n')
        
        % setup data struct and store phase
        ex_data = struct();
        ex_data.Phase = zeros(n_data, 3);
        ex_data.Phase(1:rest_idx(1), 1) = 1;
        ex_data.Phase(rest_idx(1)+1:up_idx, 2) = 1;
        ex_data.Phase(up_idx+1:rest_idx(2), 3) = 1;
        ex_data.Phase(rest_idx(2)+1:end, 1) = 1;
        
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
