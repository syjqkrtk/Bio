function [CM_result N_c Cluster_result Cluster_sn] = clustering_algorithm160909(CM, n_m)
%UNTITLED3 �� �Լ��� ��� ���� ��ġ
% %   �ڼ��� ���� ��ġ
% function [pur2 cluster_CM S2] = clustering_algorithm(CM, n_m)

l = length(n_m);
if l >= 20
    cl_c2 = 20;
else
    cl_c2 = l;
end

for C_n = 2 : cl_c2
% for C_n = 2:2

clearvars '*' -except l n_m cl_c C_n pur2 cluster_CM cluster_info CM_result_p CM cluster_mean inter_cluster_mean pur3
     
    ccc = num2str(C_n);

CM_test = 1-CM;
CM_info = 1 : 1: l;

%% Ŭ������ �߽� ã��
[a1 b1] = max(max(CM_test));
[a2 b2] = max(CM_test(b1,:));

CM_test(b1,b2) = 0;
CM_test(b2,b1) = 0;


%% ù° �Ѷ� �߽�
cl_c_p(1) = b1;
cl_c_p(2) = b2;

%% ������ �߽ɳ��
for k = 3 : l
% for k = 3 : 10
    t = zeros(1,l)+1;

    for i = 1 : length(cl_c_p)
        t = t.*CM_test(cl_c_p(i),:);
    end

    [a3 b3] = max(t);

    for i = 1 : length(cl_c_p)

        CM_test(cl_c_p(i),b3) = 0;
        CM_test(b3,cl_c_p(i)) = 0;
    end

    cl_c_p(k) = b3;

end



for re = 1 : 15

CM_info_p = 1 : 1 : l;

for i = 1 : C_n
    Cluster_p{i}(1) = cl_c_p(i);
end

cl_c_p_s = sort(cl_c_p(1:C_n),'descend');

for i = 1 : C_n
        CM_info_p(cl_c_p_s(i)) = [];
end
% 

c_n = zeros(1,C_n) + 2;
%     
    for i = 1 : length(CM_info_p)
       for k = 1: C_n
          
           a_p(k) = CM(Cluster_p{k}(1),CM_info_p(i));
           
       end
       [m1 m2] = max(a_p);
     
       Cluster_p{m2}(c_n(m2)) = CM_info_p(i);
       c_n(m2) = c_n(m2)+1;

    end
    
    hh2{re} = Cluster_p;
    
    for i = 1 : length(Cluster_p)
        for j = 1 : length(Cluster_p{i})
           for k = 1 : length(Cluster_p{i})
                CM_c_p{i}(j,k) = CM(Cluster_p{i}(j),Cluster_p{i}(k));
           end
        end
        [a CM_c_p_m_s(i)] = max(mean(CM_c_p{i}));
    end
    
    hh{re} = CM_c_p;
    
    for i = 1 : length(CM_c_p_m_s)
        cl_c(i) = Cluster_p{i}(CM_c_p_m_s(i));
    end
    
    cl_c_p = cl_c;
    
    CM_c_p_i{re} = CM_c_p;
    cl_c_p_i{re} = cl_c_p;  
    clear Cluster_p CM_c_p
end
    
%% Ŭ�����͸�
for cl_n = C_n: C_n
    
    
    CM_info = 1 : 1: l;
    c_n = zeros(1,cl_n)+2;
    
    
    %% �Ƹ� ���͵鰣�� �Ÿ��� ��°� ����
    cl_c_n_p = cl_c(1:cl_n);
    
    
    for i = 1 :  cl_n
        for j = 1 : cl_n
            if i ~= j
            CM_cc(i,j) = CM(cl_c_n_p(i),cl_c_n_p(j));
            else
                CM_cc(i,j) = 0;
            end
        end
    end
    
    
    %% Ŭ������ �߽����� ���絵�� ã�Ƽ� Ŭ������ ���� ���ϴ°�
    clxx = zeros(1,cl_n);
    clxx_p = zeros(1,cl_n);
    k = 1;
    k_c = 1;
    for re = 1 : cl_n^2
       k0 = find(clxx == 0);
       
       if length(k0) > 0
        
           [x1 y1] = max(max(CM_cc));
           [x2 y2] = max(CM_cc(y1,:));

           xx1 = find(clxx == y1);
           xx2 = find(clxx == y2);

               if length(xx1) == 0 && length(xx2) == 0 
                   clxx(k) = y1;
                   Cl_cc{k_c}(1) = y1;
                   clxx_p(k) = k_c;

                   k = k + 1;

                   clxx(k) = y2;
                   Cl_cc{k_c}(2) = y2;
                   clxx_p(k) = k_c;

                   k = k + 1;
                   k_c = k_c +1;

                   CM_cc(y1,y2) = 0;
                   CM_cc(y2,y1) = 0; 
               
               elseif length(xx1) == 0 && length(xx2) ~= 0 
                   clxx(k) = y1;
                   Cl_cc{clxx_p(xx2)}(length(Cl_cc{clxx_p(xx2)})+1) = y1;
                   clxx_p(k) = clxx_p(xx2);

                   k = k + 1;



                   CM_cc(y1,y2) = 0;
                   CM_cc(y2,y1) = 0;
                
               elseif length(xx1) ~= 0 && length(xx2) == 0 
                   clxx(k) = y2;
                   Cl_cc{clxx_p(xx1)}(length(Cl_cc{clxx_p(xx1)})+1) = y2;
                   clxx_p(k) = clxx_p(xx1);

                   k = k + 1;



                   CM_cc(y1,y2) = 0;
                   CM_cc(y2,y1) = 0;
               
               else                   
                   CM_cc(y1,y2) = 0;
                   CM_cc(y2,y1) = 0;

               end
       end
    end

    for i = 1 : length(Cl_cc)
        for j = 1 : length(Cl_cc{i})
            cl_c_c{i}(j) = cl_c_n_p(Cl_cc{i}(j));
        end
    end

    k = 1;
    for i = 1 : length(cl_c_c)
        for j = 1 : length(cl_c_c{i})
            cl_c_n(k) = cl_c_c{i}(j);
            k = k + 1;
        end
    end
%     disp(cell2mat(Cl_cc));
%     disp(cl_c_n_p);
%     disp(length(cl_c_n_p));
%     disp(cell2mat(cl_c_c));
    if cl_n>length(cl_c_n)
        cl_n = length(cl_c_n);
    end
    
  %% Ŭ������ ������
    for i = 1 : cl_n
        Cluster{i}(1) = cl_c_n(i);
    end
%     
    cl_c_s = sort(cl_c_n,'descend');
    for i = 1 : cl_n
        CM_info(cl_c_s(i)) = [];
    end
    

    
    for i = 1 : length(CM_info)
       for k = 1: cl_n
          
           a(k) = CM(Cluster{k}(1),CM_info(i));
           
       end
       [m1 m2] = max(a);
     
       Cluster{m2}(c_n(m2)) = CM_info(i);
       c_n(m2) = c_n(m2)+1;

    end
    % ������ Cluster�� �����
    
    % �߽ɳ����� �ڸ����̼� �����ϴµ�
    for i = 1:length(Cluster)
    for j = 1 : length(Cluster{i})
       
        CM_p{i}(j) = CM(Cluster{i}(1),Cluster{i}(j));
    end
    CM_p{i}(1) = 2;
    end
%   
    % �װɷ� ������
    for i = 1:length(CM_p)
        [CM_s_v CM_s_p{i}] = sort(CM_p{i},'descend');
    end
    % ������ ��� 
    for i = 1: length(CM_s_p)
        for j = 1:length(CM_s_p{i})
            CM_p_s_p{i}(j) = Cluster{i}(CM_s_p{i}(j));
        end
    end
%     


    C = CM_p_s_p;
    
    %% �������� ���

    CM_info_re = C{1};
    l1 = length(C{1});

    for i = 2 : length(C) 
        l2 = length(C{i});
        CM_info_re(l1+1:l1+l2) = C{i};
        l1 = l1 + l2; 
    end
    % CM_info_re�� �������� �����
    
   %% ���������� �̿��ؼ� ���ο� �ڸ����̼� ��Ʈ���� ����
    S1 = 1 : 1 : l;
    S2 = CM_info_re;

    CM_re = mat_ind_change(CM, S2, S1);
%     Chart_name = strcat('W=');
%     figure(cl_n)
%     visual_dmat(CM_re,Chart_name)
%     
%     al = 0; 
% % 
% for i = 1 : length(Cluster)
% % for i = 1 : 2
%     l2 = length(Cluster{i});
%     al_t(i) = al;
%     l_t(i) = l2;
% 
%     rectangle('Position',[al al l2 l2],'EdgeColor','r','LineWidth',2);
%     hold on;
% 
% al = al + l2;
% 
% end
% hold on;



    S{cl_n} = S2;
    
    %% �����Լ� ���
    CM_result_p{cl_n} = CM_re;  
    ki = 1 ;
    cluster_element = 0;
    
    for i = 1 : length(C)
       li = length(C{i});
       cluster_element = cluster_element + li*li;
       cluster_CM{cl_n}{i} = CM_re(ki:ki+li-1,ki:ki+li-1);
       cluster_info{cl_n}{i} = CM_info_re(ki:ki+li-1);
       cluster_CM_mean(i) = mean(mean(cluster_CM{cl_n}{i}));
       cluster_CM_sum(i) = sum(sum(cluster_CM{cl_n}{i}));
       ki = ki + li;              
        
    end
    
    cluster_mean(cl_n) = mean(cluster_CM_mean);
    
    cluster_sum = sum(cluster_CM_sum);
    
    total_sum = sum(sum(CM_re));
    total_element = l*l;
    
    inter_cluster_mean(cl_n) = (total_sum-cluster_sum)/(total_element-cluster_element);
    pur2(cl_n) = cluster_mean(cl_n)/inter_cluster_mean(cl_n);
    pur3(cl_n) = cluster_mean(cl_n)-inter_cluster_mean(cl_n);
    pur4(cl_n) = cluster_mean(cl_n)*(1-inter_cluster_mean(cl_n));

    
end
end
   [am bm] = max(pur2);
    N_c = bm;
    Cluster_result = cluster_CM{bm};
    
    for i = 1 : length(cluster_info{bm})
        for j = 1 : length(cluster_info{bm}{i})
                      
            Cluster_sn{i}(j) = n_m(cluster_info{bm}{i}(j));
        end
    end

    CM_result = CM_result_p{bm};




end

