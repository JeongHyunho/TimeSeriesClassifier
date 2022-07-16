% Squat

% subject_list = 2:35;
subject_list = 2;

subject_list(subject_list == 20) = [];
subject_list(subject_list == 25) = [];

for idx = subject_list
    data = encode_squat(idx, true);
end