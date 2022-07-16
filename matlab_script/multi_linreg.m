
pilot_num = 1;

data = encode_snu(pilot_num, true);

% X_field = ["R_RECTUS", "R_MEDIAL_HAM", "R_TIBIALIS_ANT", "R_GASTROCNEMIUS", "L_RECTUS", "L_MEDIAL_HAM", "L_TIBIALIS_ANT", "L_GASTROCNEMIUS"];
% Y_field = "R_ANK_FLEX_ANG";
% 
% X = [];
% for i = 1:length(X_field)
%     X = [X, data.(X_field(i))];
% end
% 
% Y = data.(Y_field);
% 
% model = fitlm(X, Y);
% X_field
% model