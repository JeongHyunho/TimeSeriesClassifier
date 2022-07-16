
subject_list = 1:35;
% subject_list = 1;

type = "Gait";
% type = "Lunge";
% type = "Squat";


for idx = subject_list
    data = encode_snu(idx, type, true);
end