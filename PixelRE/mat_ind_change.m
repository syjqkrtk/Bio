%%%%% Distance matrix row-column change
%%%%% Byung Chang Chung, KAIST UMLS
%%%%%
%%%%% input: symmetric distance matrix (dmat_bef), two row indices (ind_t1,
%%%%% ind_t2)
%%%%% output: symmetric distance matrix (dmat_aft)
%%%%%

function [dmat_aft] = mat_ind_change(dmat_bef, ind_t1, ind_t2)

[row col] = size(dmat_bef);

if row ~= col
    dmat_aft = zeros(row,col);
else
    dmat_aft = dmat_bef;
    
    dmat_aft(:,ind_t1) = dmat_bef(:,ind_t2);
    dmat_aft(:,ind_t2) = dmat_bef(:,ind_t1);
    
    temp_1 = zeros(1,col);
    temp_1 = dmat_aft(ind_t1,:);
    dmat_aft(ind_t1,:) = dmat_aft(ind_t2,:);
    dmat_aft(ind_t2,:) = temp_1;
end




end

