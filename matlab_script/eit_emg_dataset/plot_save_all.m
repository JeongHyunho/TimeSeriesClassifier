% 

cur_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(cur_dir, '..', '..', 'data', 'EIT_EMG_Gait_210127');
data_files = dir(fullfile(root_dir, 'S*.mat'));

for i = 1:length(data_files)
    filename = fullfile(data_files(i).folder, data_files(i).name);
    name_enc = split(data_files(i).name, ["_", "."]);
    sub_id = str2num(name_enc{1}(2:end));
    label = str2num(name_enc{2});
    trial = str2num(name_enc{3});
    
    data = load(filename, 'Data').Data;
    n_frames = size(data, 2);
    
    fh = figure('Visible', 'off');
    fh.Position(3) = 2 * fh.Position(3);
    subplot(1, 2, 1)
    plot(0.01 * (0:n_frames-1), data(1:8, :)')
    title('EMG G' + string(label) + ' S' + string(sub_id) + ' T' + string(trial))
    subplot(1, 2, 2)
    plot(0.01 * (0:n_frames-1), data(9:72, :)')
    title('EIT G' + string(label) + ' S' + string(sub_id) + ' T' + string(trial))
    saveas(fh, filename(1:end-4) + ".png")
    
    close(fh)
    
    fprintf('%d/%d done', i, length(data_files))
end
